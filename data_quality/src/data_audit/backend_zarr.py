"""
Backend B: read the Zarr stores straight out of ``s3://cadcat``.

This bypasses climakitae entirely, which matters for two reasons:

1. climakitae repairs some metadata on read (it attaches a CRS to cadcat stores
   that lack one). Anything it repairs is invisible to backend A but is still a
   defect in the delivered artifact.
2. Only a direct read can see *store-level* structure: consolidated metadata,
   Zarr format version, chunk shape, compressor, dtype. None of that survives
   into the xarray object.

Everything is anonymous (``anon=True``) — the bucket is public.
"""

from __future__ import annotations

import json
import logging
import posixpath
from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from . import standards as S
from .findings import FindingList

logger = logging.getLogger(__name__)

#: Chunks much outside this band make for slow or memory-hungry reads.
CHUNK_MB_SOFT_MIN = 1.0
CHUNK_MB_SOFT_MAX = 200.0


class ZarrBackend:
    """
    Open cadcat Zarr stores directly.

    Parameters
    ----------
    anon
        Anonymous S3 access. Leave True for the public bucket.
    consolidated
        Passed to ``xr.open_zarr``. ``None`` uses consolidated metadata when it
        exists and falls back to per-array reads, which is what climakitae does
        and what keeps Zarr v2 and v3 stores both working.
    """

    name = "zarr"

    #: Reads one published store per call with no aggregation, so its structure
    #: is the archive's structure.
    authoritative_structure = True

    def __init__(self, anon: bool = True, consolidated: bool | None = None) -> None:
        """
        Configure anonymous access to the public bucket.

        Parameters
        ----------
        anon : bool, optional
            Anonymous S3 access. Leave True for the public ``cadcat`` bucket.
        consolidated : bool, optional
            Passed to ``xr.open_zarr``. ``None`` uses consolidated metadata when it
            exists and falls back to per-array reads, which keeps both Zarr v2 and
            v3 stores working.
        """
        self.anon = anon
        self.consolidated = consolidated
        self._fs = None

    #  filesystem
    @property
    def fs(self) -> Any:
        """
        Return the s3fs filesystem, creating it on first use.

        Returns
        -------
        Any
            An ``s3fs.S3FileSystem``. Deferred so that importing this module does
            not require s3fs.
        """
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=self.anon)
        return self._fs

    #  opening
    def open(self, row: dict[str, Any]) -> tuple[xr.Dataset | None, dict[str, Any]]:
        """
        Return ``(dataset, diagnostics)`` for one catalog row.

        Parameters
        ----------
        row : dict
            Catalog record. ``path`` is used when present, otherwise the path is
            rebuilt from the record's facets.

        Returns
        -------
        dataset : xarray.Dataset or None
            The lazily-opened store, or ``None`` on failure.
        diagnostics : dict
            Always carries ``backend`` and ``path``; carries ``error`` on failure.
        """
        path = row.get("path") or S.expected_s3_path(row)
        diagnostics: dict[str, Any] = {"backend": self.name, "path": path}
        if not path:
            diagnostics["error"] = "no path available for this row"
            return None, diagnostics

        store = _to_s3_key(path)
        try:
            import s3fs

            mapper = s3fs.S3Map(root=store, s3=self.fs, check=False)
            dataset = xr.open_zarr(
                mapper,
                consolidated=self.consolidated,
                decode_times=True,
                mask_and_scale=True,
                chunks={},
            )
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return None, diagnostics
        return dataset, diagnostics

    #  raw store inspection
    def inspect_store(self, row: dict[str, Any]) -> FindingList:
        """
        Structural checks that only a raw read can make.

        Parameters
        ----------
        row : dict
            Catalog record. ``path`` is used when present, otherwise the path is
            rebuilt from the record's facets.

        Returns
        -------
        FindingList
            Findings about consolidated metadata, Zarr format, chunking, dtype and
            compression. A single FATAL finding when the prefix cannot be reached.
        """
        out = FindingList()
        path = row.get("path") or S.expected_s3_path(row)
        if not path:
            out.fatal("store.path.unknown", "No S3 path for this record.")
            return out
        store = _to_s3_key(path)

        try:
            exists = self.fs.exists(store)
        except Exception as exc:
            out.fatal("store.unreachable", f"Could not reach S3: {exc}")
            return out
        if not exists:
            out.fatal(
                "store.missing",
                "Catalog record points at a prefix that does not exist in S3.",
                actual=path,
            )
            return out

        metadata, layout = self._read_store_metadata(store)
        if metadata is None:
            out.error(
                "store.metadata.unreadable",
                "Neither .zmetadata, .zgroup nor zarr.json could be read.",
                actual=path,
            )
            return out

        if layout == "consolidated_v2":
            out.ok("store.consolidated", "Consolidated metadata (.zmetadata) present.")
        elif layout == "v3":
            out.info("store.zarr_v3", "Zarr v3 store (zarr.json).", actual="v3")
        else:
            out.warn(
                "store.not_consolidated",
                "No .zmetadata. Opening this store requires one request per array, "
                "which is slow over S3 and is why intake-esm reads can stall.",
                expected=".zmetadata",
                actual=layout,
            )

        variable_id = row.get("variable_id")
        out += self._check_arrays(metadata, variable_id)
        return out

    #  internals
    def _read_store_metadata(self, store: str) -> tuple[dict[str, Any] | None, str]:
        """
        Return ``(metadata_dict, layout)``.

        ``layout`` is ``consolidated_v2``, ``unconsolidated_v2`` or ``v3``.

        Parameters
        ----------
        store : str
            S3 key of the store prefix.

        Returns
        -------
        metadata : dict or None
            Parsed metadata, or ``None`` when nothing could be read.
        layout : str
            ``"consolidated_v2"``, ``"v3"``, ``"unconsolidated_v2"`` or
            ``"unknown"``.
        """
        for name, layout in (
            (".zmetadata", "consolidated_v2"),
            ("zarr.json", "v3"),
        ):
            key = posixpath.join(store, name)
            try:
                if self.fs.exists(key):
                    raw = json.loads(self.fs.cat(key).decode("utf-8"))
                    if layout == "consolidated_v2":
                        return raw.get("metadata", raw), layout
                    return raw, layout
            except Exception:
                continue

        # Unconsolidated v2: walk one level and read each .zarray
        try:
            metadata: dict[str, Any] = {}
            for entry in self.fs.ls(store, detail=False):
                zarray = posixpath.join(entry, ".zarray")
                zattrs = posixpath.join(entry, ".zattrs")
                base = posixpath.basename(entry.rstrip("/"))
                if self.fs.exists(zarray):
                    metadata[f"{base}/.zarray"] = json.loads(
                        self.fs.cat(zarray).decode("utf-8")
                    )
                if self.fs.exists(zattrs):
                    metadata[f"{base}/.zattrs"] = json.loads(
                        self.fs.cat(zattrs).decode("utf-8")
                    )
            return (metadata or None), "unconsolidated_v2"
        except Exception:
            return None, "unknown"

    def _check_arrays(
        self, metadata: dict[str, Any], variable_id: str | None
    ) -> FindingList:
        """
        Chunking, dtype and compression of the payload array.

        Parameters
        ----------
        metadata : dict
            Store metadata from :meth:`_read_store_metadata`.
        variable_id : str or None
            Name of the payload array; the widest array is used when absent.

        Returns
        -------
        FindingList
            Findings about shape, chunking, dtype and compression.
        """
        out = FindingList()
        arrays = {
            key[: -len("/.zarray")]: value
            for key, value in metadata.items()
            if isinstance(key, str) and key.endswith("/.zarray")
        }
        if not arrays:
            out.info(
                "store.arrays.none", "No v2 .zarray records found (Zarr v3 store?)."
            )
            return out

        target = variable_id if variable_id in arrays else None
        if target is None:
            ranked = sorted(
                arrays.items(),
                key=lambda kv: len(kv[1].get("chunks", []) or []),
                reverse=True,
            )
            target = ranked[0][0] if ranked else None
        if target is None:
            return out

        spec = arrays[target]
        chunks = spec.get("chunks") or []
        shape = spec.get("shape") or []
        dtype = spec.get("dtype")
        compressor = spec.get("compressor")

        out.info(
            "store.array.shape",
            f"{target}: shape {tuple(shape)}, chunks {tuple(chunks)}, dtype {dtype}.",
            actual=f"shape={tuple(shape)} chunks={tuple(chunks)} dtype={dtype}",
        )

        if compressor is None:
            out.warn(
                "store.array.uncompressed",
                f"{target} is stored uncompressed.",
                expected="a compressor",
                actual=None,
            )

        chunk_mb = _chunk_megabytes(chunks, dtype)
        if chunk_mb is not None:
            if chunk_mb > CHUNK_MB_SOFT_MAX:
                out.warn(
                    "store.chunk.too_large",
                    f"Chunks are ~{chunk_mb:.0f} MB; a single element read pulls "
                    "that much across the wire.",
                    expected=f"<{CHUNK_MB_SOFT_MAX:.0f} MB",
                    actual=f"{chunk_mb:.0f} MB",
                )
            elif chunk_mb < CHUNK_MB_SOFT_MIN:
                n_chunks = _chunk_count(shape, chunks)
                out.warn(
                    "store.chunk.too_small",
                    f"Chunks are ~{chunk_mb:.2f} MB across about {n_chunks:,} chunks; "
                    "per-object S3 latency will dominate reads.",
                    expected=f">{CHUNK_MB_SOFT_MIN:.0f} MB",
                    actual=f"{chunk_mb:.2f} MB",
                )

        if shape and chunks and len(shape) == len(chunks):
            if any(c > s for c, s in zip(chunks, shape)):
                out.info(
                    "store.chunk.exceeds_shape",
                    "At least one chunk dimension exceeds the array dimension "
                    "(a single-chunk axis).",
                    actual=f"chunks={tuple(chunks)} shape={tuple(shape)}",
                )
        return out

    #  lifecycle
    def close(self) -> None:
        """
        Release the underlying s3fs session.

        Without this, aiohttp complains about an unclosed client session at
        interpreter exit. Harmless, but it lands on stderr after the report has
        printed and reads like a failure.

        The session belongs to the aiobotocore client that s3fs wraps, not to
        s3fs itself, so it has to be closed through ``S3FileSystem.close_session``
        (the same entry point s3fs registers with ``weakref.finalize``). Older
        and newer releases expose it differently, hence the fallbacks.
        """
        filesystem, self._fs = self._fs, None
        if filesystem is None:
            return

        loop = getattr(filesystem, "loop", None)
        client = getattr(filesystem, "s3", None)

        closer = getattr(type(filesystem), "close_session", None)
        if callable(closer) and client is not None:
            try:
                closer(loop, client)
                self._clear_cache(filesystem)
                return
            except Exception:
                pass

        # Fallback: exit the client's async context directly.
        if client is not None:
            try:
                from fsspec.asyn import sync

                sync(loop, client.__aexit__, None, None, None, timeout=2)
                self._clear_cache(filesystem)
                return
            except Exception:
                pass

        self._clear_cache(filesystem)

    @staticmethod
    def _clear_cache(filesystem: Any) -> None:
        """
        Drop the s3fs instance cache.

        Parameters
        ----------
        filesystem : Any
            The filesystem whose class cache should be cleared.
        """
        try:
            type(filesystem).clear_instance_cache()
        except Exception:
            pass

    def __enter__(self) -> ZarrBackend:
        """
        Enter the context manager.

        Returns
        -------
        ZarrBackend
            This backend.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """
        Close the session on leaving the context manager.

        Parameters
        ----------
        *exc : Any
            Exception triple, ignored; cleanup runs either way.
        """
        self.close()

    #  discovery
    def list_stores(self, prefix: str, max_depth: int = 8) -> list[str]:
        """
        List Zarr store prefixes under ``prefix`` (e.g. ``cadcat/wrf/ucla``).

        A store is any prefix containing ``.zmetadata``, ``.zgroup`` or
        ``zarr.json``. Use this to find stores that exist in the bucket but are
        absent from the intake catalog.

        Parameters
        ----------
        prefix : str
            S3 prefix to walk, e.g. ``"cadcat/wrf/ucla"``.
        max_depth : int, optional
            Recursion limit. Default 8.

        Returns
        -------
        list of str
            Sorted store prefixes. Subtract the catalog's paths from these to find
            stores the catalog does not index.
        """
        found: list[str] = []
        self._walk(prefix.rstrip("/"), 0, max_depth, found)
        return sorted(found)

    def _walk(self, prefix: str, depth: int, max_depth: int, found: list[str]) -> None:
        """
        Recurse into a prefix looking for Zarr stores.

        Parameters
        ----------
        prefix : str
            Prefix to inspect.
        depth : int
            Current recursion depth.
        max_depth : int
            Recursion limit.
        found : list of str
            Accumulator, appended to in place.
        """
        if depth > max_depth:
            return
        try:
            entries = self.fs.ls(prefix, detail=True)
        except Exception:
            return
        names = {posixpath.basename(e["name"].rstrip("/")) for e in entries}
        if names & {".zmetadata", ".zgroup", "zarr.json"}:
            found.append(prefix)
            return
        for entry in entries:
            if entry.get("type") == "directory":
                self._walk(entry["name"].rstrip("/"), depth + 1, max_depth, found)


def _to_s3_key(path: str) -> str:
    """
    ``s3://cadcat/a/b/`` -> ``cadcat/a/b``; leaves bare keys alone.

    Parameters
    ----------
    path : str
        Store path, with or without the ``s3://`` scheme.

    Returns
    -------
    str
        Bucket-qualified key with no trailing slash.
    """
    text = str(path).strip()
    if text.startswith("s3://"):
        text = text[len("s3://") :]
    return text.rstrip("/")


def _chunk_megabytes(chunks: Iterable[int], dtype: str | None) -> float | None:
    """
    Compute the size of one chunk.

    Parameters
    ----------
    chunks : iterable of int
        Chunk shape.
    dtype : str or None
        NumPy dtype string.

    Returns
    -------
    float or None
        Chunk size in MiB, or ``None`` when the dtype cannot be interpreted.
    """
    chunks = list(chunks or [])
    if not chunks or not dtype:
        return None
    try:
        itemsize = np.dtype(dtype.lstrip("<>|=")).itemsize
    except Exception:
        try:
            itemsize = np.dtype(dtype).itemsize
        except Exception:
            return None
    total = itemsize
    for c in chunks:
        total *= int(c)
    return total / (1024.0**2)


def _chunk_count(shape: Iterable[int], chunks: Iterable[int]) -> int:
    """
    Count the chunks in an array.

    Parameters
    ----------
    shape : iterable of int
        Array shape.
    chunks : iterable of int
        Chunk shape.

    Returns
    -------
    int
        Number of chunks, or 0 when the shapes do not correspond.
    """
    shape, chunks = list(shape or []), list(chunks or [])
    if not shape or len(shape) != len(chunks):
        return 0
    total = 1
    for s, c in zip(shape, chunks):
        total *= max(1, -(-int(s) // int(c)))
    return total
