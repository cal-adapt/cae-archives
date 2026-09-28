#!/usr/bin/env python
"""Measure type-hint and NumPy-docstring coverage across the package.

Converting 13,000 lines of working analysis code to full type hints and
NumPy-style docstrings is not a single edit; it is a queue. This reports where
the queue stands so the work can be done a module at a time without losing
track, and so a CI job can stop coverage regressing.

A function counts as **typed** when every parameter other than ``self``/``cls``
carries an annotation and a return annotation is present. It counts as
**documented** when it has a docstring with the NumPy sections appropriate to
its signature: ``Parameters`` if it takes arguments, ``Returns`` if it can
return a value. Private helpers (leading underscore) are reported separately,
since the house rule is that they need hints but only a summary line.

Run
---
``python scripts/audit_docs.py``                      whole package
``python scripts/audit_docs.py --details checks``      per-function detail
``python scripts/audit_docs.py --details grids``      per-function for a module
``python scripts/audit_docs.py --min-coverage 80``    exit 1 below a threshold
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

#: NumPy docstring sections this tool knows how to look for.
NUMPY_SECTIONS: Tuple[str, ...] = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Notes",
    "See Also",
    "Attributes",
)

#: Return-statement-free functions still need Returns if they are generators.
_RETURNING = (ast.Return, ast.Yield, ast.YieldFrom)


@dataclass
class FunctionReport:
    """Coverage of one function.

    Attributes
    ----------
    name : str
        Dotted name within its module, e.g. ``Klass.method``.
    lineno : int
        Line the definition starts on.
    private : bool
        Whether the name begins with a single underscore.
    typed : bool
        Every parameter and the return value carry annotations.
    has_docstring : bool
        A docstring is present and non-empty.
    missing_sections : list of str
        NumPy sections the signature implies but the docstring lacks.
    """

    name: str
    lineno: int
    private: bool
    typed: bool
    has_docstring: bool
    missing_sections: List[str] = field(default_factory=list)

    @property
    def documented(self) -> bool:
        """Whether the docstring is present and has the sections it needs.

        Returns
        -------
        bool
            True when nothing is missing.
        """
        return self.has_docstring and not self.missing_sections

    @property
    def complete(self) -> bool:
        """Whether the function needs no further work.

        Returns
        -------
        bool
            True when typed and documented.
        """
        return self.typed and self.documented


@dataclass
class ModuleReport:
    """Coverage of one module.

    Attributes
    ----------
    path : Path
        File inspected.
    module : str
        Importable module name.
    has_module_docstring : bool
        Whether the file opens with a docstring.
    functions : list of FunctionReport
        One entry per function or method.
    error : str or None
        Parse failure message, if the file could not be read.
    """

    path: Path
    module: str
    has_module_docstring: bool = False
    functions: List[FunctionReport] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def public(self) -> List[FunctionReport]:
        """Public functions only.

        Returns
        -------
        list of FunctionReport
            Functions whose name does not begin with an underscore.
        """
        return [f for f in self.functions if not f.private]

    def counts(self, include_private: bool = False) -> Tuple[int, int, int]:
        """Summarise the module.

        Parameters
        ----------
        include_private : bool, optional
            Count underscore-prefixed functions too. Default False.

        Returns
        -------
        tuple of int
            ``(total, typed, documented)``.
        """
        pool = self.functions if include_private else self.public
        return (
            len(pool),
            sum(1 for f in pool if f.typed),
            sum(1 for f in pool if f.documented),
        )


def _is_typed(node: ast.AST) -> bool:
    """Check whether a function is fully annotated.

    Parameters
    ----------
    node : ast.AST
        A ``FunctionDef`` or ``AsyncFunctionDef`` node.

    Returns
    -------
    bool
        True when every parameter except ``self``/``cls`` is annotated and a
        return annotation is present.
    """
    args = node.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    params = [a for a in params if a.arg not in ("self", "cls")]
    if args.vararg is not None:
        params.append(args.vararg)
    if args.kwarg is not None:
        params.append(args.kwarg)
    return all(a.annotation is not None for a in params) and node.returns is not None


def _takes_arguments(node: ast.AST) -> bool:
    """Whether a function has parameters worth documenting.

    Parameters
    ----------
    node : ast.AST
        A function definition node.

    Returns
    -------
    bool
        True when at least one parameter other than ``self``/``cls`` exists.
    """
    args = node.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    params = [a for a in params if a.arg not in ("self", "cls")]
    return bool(params or args.vararg or args.kwarg)


def _returns_value(node: ast.AST) -> bool:
    """Whether a function can hand a value back to its caller.

    Parameters
    ----------
    node : ast.AST
        A function definition node.

    Returns
    -------
    bool
        True when the body contains a non-bare ``return``, a ``yield``, or the
        signature annotates a return type other than ``None``.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True
    if node.returns is not None:
        text = ast.unparse(node.returns) if hasattr(ast, "unparse") else ""
        if text.strip() not in ("None", ""):
            return True
    return False


def _missing_sections(node: ast.AST, docstring: Optional[str]) -> List[str]:
    """Identify NumPy sections the signature implies but the docstring omits.

    Parameters
    ----------
    node : ast.AST
        A function definition node.
    docstring : str or None
        The function's docstring.

    Returns
    -------
    list of str
        Section names that should be present and are not. Empty when the
        docstring is adequate, or when there is no docstring at all (that is
        reported separately).
    """
    if not docstring:
        return []
    missing: List[str] = []
    if _takes_arguments(node) and "Parameters" not in docstring:
        missing.append("Parameters")
    if _returns_value(node):
        if not any(s in docstring for s in ("Returns", "Yields")):
            missing.append("Returns")
    return missing


def _walk_functions(tree: ast.AST) -> Iterator[Tuple[str, ast.AST]]:
    """Yield every function in a module, qualifying methods with their class.

    Parameters
    ----------
    tree : ast.AST
        Parsed module.

    Yields
    ------
    tuple of (str, ast.AST)
        Dotted name and the function node.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{child.name}", child


def audit_module(path: Path, module: str) -> ModuleReport:
    """Inspect one Python file.

    Parameters
    ----------
    path : Path
        File to read.
    module : str
        Importable module name, used for display.

    Returns
    -------
    ModuleReport
        Coverage for the file. On a parse error the report carries ``error``
        and no functions.
    """
    report = ModuleReport(path=path, module=module)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    report.has_module_docstring = bool(ast.get_docstring(tree))
    for name, node in _walk_functions(tree):
        docstring = ast.get_docstring(node)
        report.functions.append(
            FunctionReport(
                name=name,
                lineno=node.lineno,
                private=name.rsplit(".", 1)[-1].startswith("_"),
                typed=_is_typed(node),
                has_docstring=bool(docstring),
                missing_sections=_missing_sections(node, docstring),
            )
        )
    return report


def audit_package(root: Path, packages: Sequence[str]) -> List[ModuleReport]:
    """Inspect every module in the named packages.

    Parameters
    ----------
    root : Path
        Directory containing the packages, normally ``src``.
    packages : sequence of str
        Package directory names.

    Returns
    -------
    list of ModuleReport
        One report per file, sorted by module name.
    """
    reports: List[ModuleReport] = []
    for package in packages:
        base = root / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            relative = path.relative_to(root).with_suffix("")
            reports.append(audit_module(path, ".".join(relative.parts)))
    return sorted(reports, key=lambda r: r.module)


def format_table(
    reports: Iterable[ModuleReport], include_private: bool = False
) -> str:
    """Render a per-module coverage table.

    Parameters
    ----------
    reports : iterable of ModuleReport
        Reports to summarise.
    include_private : bool, optional
        Count underscore-prefixed functions. Default False.

    Returns
    -------
    str
        A plain-text table with a TOTAL row.
    """
    reports = list(reports)
    width = max((len(r.module) for r in reports), default=10)
    lines = [
        f"{'module':{width}s} {'fns':>5} {'typed':>7} {'documented':>11}  {'mod doc':>7}",
        "-" * (width + 34),
    ]
    totals = [0, 0, 0]
    for report in reports:
        if report.error:
            lines.append(f"{report.module:{width}s}  PARSE FAILED: {report.error}")
            continue
        total, typed, documented = report.counts(include_private)
        totals[0] += total
        totals[1] += typed
        totals[2] += documented
        pct_t = f"{100 * typed / total:.0f}%" if total else "-"
        pct_d = f"{100 * documented / total:.0f}%" if total else "-"
        flag = "yes" if report.has_module_docstring else "NO"
        lines.append(
            f"{report.module:{width}s} {total:5d} {typed:4d} {pct_t:>4s} "
            f"{documented:5d} {pct_d:>5s}  {flag:>7s}"
        )
    total, typed, documented = totals
    pct_t = f"{100 * typed / total:.0f}%" if total else "-"
    pct_d = f"{100 * documented / total:.0f}%" if total else "-"
    lines += [
        "-" * (width + 34),
        f"{'TOTAL':{width}s} {total:5d} {typed:4d} {pct_t:>4s} "
        f"{documented:5d} {pct_d:>5s}",
    ]
    return "\n".join(lines)


def format_details(report: ModuleReport, include_private: bool = False) -> str:
    """Render the per-function gaps for one module.

    Parameters
    ----------
    report : ModuleReport
        Module to detail.
    include_private : bool, optional
        Include underscore-prefixed functions. Default False.

    Returns
    -------
    str
        One line per incomplete function.
    """
    if report.error:
        return f"{report.module}: PARSE FAILED: {report.error}"
    pool = report.functions if include_private else report.public
    outstanding = [f for f in pool if not f.complete]
    if not outstanding:
        return f"{report.module}: complete"
    lines = [f"{report.module}: {len(outstanding)} of {len(pool)} need work"]
    for function in sorted(outstanding, key=lambda f: f.lineno):
        gaps: List[str] = []
        if not function.typed:
            gaps.append("type hints")
        if not function.has_docstring:
            gaps.append("docstring")
        elif function.missing_sections:
            gaps.append("missing " + "/".join(function.missing_sections))
        lines.append(f"  L{function.lineno:<5d} {function.name:40s} {', '.join(gaps)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Report type-hint and NumPy-docstring coverage."
    )
    parser.add_argument(
        "--root", default="src", help="directory holding the packages (default: src)"
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help="package to audit; repeatable (default: data_audit)",
    )
    parser.add_argument(
        "--details", metavar="MODULE", help="list per-function gaps for a module"
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="count underscore-prefixed functions",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help="exit 1 if combined typed+documented coverage falls below this percent",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the audit.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 if ``--min-coverage`` is not met
        or no modules were found.

    Notes
    -----
    Restores the default ``SIGPIPE`` behaviour so that piping the output into
    ``head`` terminates quietly rather than raising ``BrokenPipeError``.
    """
    try:  # not available on Windows
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):  # pragma: no cover
        pass

    args = build_parser().parse_args(argv)
    packages = args.packages or ["data_audit"]
    reports = audit_package(Path(args.root), packages)

    if not reports:
        print(f"no modules found under {args.root}/{{{','.join(packages)}}}")
        return 1

    if args.details:
        matches = [r for r in reports if r.module.endswith(args.details)]
        if not matches:
            print(f"no module matching {args.details!r}")
            return 1
        for report in matches:
            print(format_details(report, args.include_private))
        return 0

    print(format_table(reports, args.include_private))

    totals = [0, 0, 0]
    for report in reports:
        total, typed, documented = report.counts(args.include_private)
        totals[0] += total
        totals[1] += typed
        totals[2] += documented
    total, typed, documented = totals
    if args.min_coverage is not None and total:
        combined = 100.0 * (typed + documented) / (2 * total)
        print(f"\ncombined coverage: {combined:.1f}% (threshold {args.min_coverage}%)")
        if combined < args.min_coverage:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
