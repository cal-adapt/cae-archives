"""
Finding data model shared by every check.

A *Finding* is one observation about one dataset. Checks never raise on bad
data; they emit findings.
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from typing import Any


class Level(IntEnum):
    """
    Severity ordering.

    Higher is worse, so ``max()`` over a dataset's findings gives its grade.

    Attributes
    ----------
    OK : int
        The check passed.
    INFO : int
        Worth recording, not a defect.
    WARN : int
        A deviation from the documented standard that is tolerable or expected.
    ERROR : int
        A defect against the documentation.
    FATAL : int
        The dataset could not be opened or inspected at all.
    """

    OK = 0
    INFO = 10
    WARN = 20
    ERROR = 30
    FATAL = 40

    @property
    def label(self) -> str:
        """
        Return the member name, for use as a column value.

        Returns
        -------
        str
            The severity name, e.g. ``"ERROR"``.
        """
        return self.name


@dataclasses.dataclass
class Finding:
    """
    One QC observation.

    Parameters
    ----------
    code : str
        Stable machine-readable identifier, e.g. ``"units.mismatch"``. Group and
        count on this; never parse ``message``.
    level : Level
        Severity.
    message : str
        Human sentence describing what was seen.
    expected : Any, optional
        The value the documentation calls for, when the check is a comparison.
    actual : Any, optional
        The value observed. Kept in a separate column from ``expected`` so a
        reviewer can eyeball a spreadsheet.
    check : str, optional
        Name of the check function that produced this. Filled in by the runner.
    context : dict, optional
        Facets identifying the dataset (``activity_id``, ``variable_id``, ...).
        Filled in by the runner, not by individual checks.
    """

    code: str
    level: Level
    message: str
    expected: Any | None = None
    actual: Any | None = None
    check: str = ""
    context: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """
        Flatten to a single record for a tidy table.

        Returns
        -------
        dict
            The context facets, followed by ``check``, ``code``, ``level``,
            ``level_num``, ``message``, ``expected`` and ``actual``.
        """
        row: dict[str, Any] = dict(self.context)
        row.update(
            {
                "check": self.check,
                "code": self.code,
                "level": self.level.label,
                "level_num": int(self.level),
                "message": self.message,
                "expected": _stringify(self.expected),
                "actual": _stringify(self.actual),
            }
        )
        return row


def _stringify(value: Any) -> str | None:
    """
    Render a value for a table cell, sorting collections and truncating.

    Parameters
    ----------
    value : Any
        Value to render. Collections are sorted so that two runs comparing the
        same set produce identical text.

    Returns
    -------
    str or None
        ``None`` passes through; anything else becomes a string of at most 500
        characters, truncated with an ellipsis.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = sorted(str(v) for v in value)
        text = ", ".join(items)
    else:
        text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


class FindingList(list[Finding]):
    """
    A list of findings with per-severity constructors.

    Subclasses :class:`list`, so it supports ``+=`` between check functions and
    ordinary iteration by the runner.
    """

    def add(
        self,
        code: str,
        level: Level,
        message: str,
        expected: Any = None,
        actual: Any = None,
    ) -> FindingList:
        """
        Append a finding at an explicit severity.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        level : Level
            Severity.
        message : str
            Human sentence describing what was seen.
        expected : Any, optional
            The documented value.
        actual : Any, optional
            The observed value.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        self.append(
            Finding(
                code=code,
                level=level,
                message=message,
                expected=expected,
                actual=actual,
            )
        )
        return self

    def ok(self, code: str, message: str, **kw: Any) -> FindingList:
        """
        Append a passing finding.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        message : str
            Human sentence describing what was seen.
        **kw : Any
            Passed to :meth:`add`, normally ``expected`` and ``actual``.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        return self.add(code, Level.OK, message, **kw)

    def info(self, code: str, message: str, **kw: Any) -> FindingList:
        """
        Append an informational finding.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        message : str
            Human sentence describing what was seen.
        **kw : Any
            Passed to :meth:`add`.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        return self.add(code, Level.INFO, message, **kw)

    def warn(self, code: str, message: str, **kw: Any) -> FindingList:
        """
        Append a warning.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        message : str
            Human sentence describing what was seen.
        **kw : Any
            Passed to :meth:`add`.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        return self.add(code, Level.WARN, message, **kw)

    def error(self, code: str, message: str, **kw: Any) -> FindingList:
        """
        Append an error.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        message : str
            Human sentence describing what was seen.
        **kw : Any
            Passed to :meth:`add`.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        return self.add(code, Level.ERROR, message, **kw)

    def fatal(self, code: str, message: str, **kw: Any) -> FindingList:
        """
        Append a fatal finding, used when a dataset cannot be inspected.

        Parameters
        ----------
        code : str
            Stable machine-readable identifier.
        message : str
            Human sentence describing what was seen.
        **kw : Any
            Passed to :meth:`add`.

        Returns
        -------
        FindingList
            This list, to allow chaining.
        """
        return self.add(code, Level.FATAL, message, **kw)

    @property
    def worst(self) -> Level:
        """
        Return the highest severity present.

        Returns
        -------
        Level
            The maximum severity, or :attr:`Level.OK` when the list is empty.
        """
        return max((f.level for f in self), default=Level.OK)
