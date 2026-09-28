#!/usr/bin/env python
"""
Publish the analysis outputs in data/ to S3 as zarr.

    python publish_to_s3.py --dry-run          # list what would happen
    python publish_to_s3.py                    # convert and upload
    python publish_to_s3.py --overwrite        # replace what is already there

What it does:
  * every .nc becomes a .zarr group under the destination prefix
  * existing .zarr stores are copied as-is (no round-trip through xarray)
  * other files (.csv, ...) are uploaded unchanged

Requires: s3fs, zarr, xarray, netCDF4

Credentials come from the usual boto3 chain (env vars, ~/.aws/credentials,
instance role). Nothing is read from the script.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

DEFAULT_SRC = "data"
DEFAULT_DEST = "s3://cadcat-tmp/data_quality"

# zarr v2: v3 has no consolidated metadata and is stricter about codecs, and v2
# is what the HDP store already uses.
ZARR_FORMAT = 2


def clean_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Make a Dataset writable to zarr.

    Two fixes, both learned the hard way:
      * encoding inherited from the source file (netCDF chunking, compressors,
        fill values) conflicts with the zarr writer -- drop it
      * numpy 2 StringDType and fixed-width unicode are not writable; object
        dtype becomes a zarr vlen string
    """
    ds = ds.copy()
    for name in list(ds.variables):
        ds[name].encoding = {}
        if ds[name].dtype.kind in ("T", "U"):
            ds[name] = ds[name].astype(object)
    return ds


def sanitize_chunks(ds: xr.Dataset) -> xr.Dataset:
    """Give every array a single chunk per dimension if it is not chunked.

    Small analysis outputs do not benefit from chunking, and unchunked writes
    avoid the "inconsistent chunks" errors that appear when variables in one
    file were chunked differently.
    """
    if not ds.chunks:
        return ds
    return ds.chunk({d: -1 for d in ds.dims})


def _exists(fs, target: str, dry_run: bool) -> bool:
    """Existence check that never fails a dry run on credentials."""
    if dry_run:
        try:
            return fs.exists(target)
        except Exception:
            return False
    return fs.exists(target)


def convert_netcdf(path: pathlib.Path, dest: str, fs, overwrite: bool,
                   dry_run: bool, storage_options: dict | None = None) -> str:
    target = f"{dest}/{path.stem}.zarr"
    if _exists(fs, target, dry_run) and not overwrite:
        print(f"  SKIP  {path.name} -> {target} (exists; --overwrite to replace)")
        return "skipped"
    if dry_run:
        print(f"  would convert  {path.name} -> {target}")
        return "dry"

    if not target.startswith("s3://"):
        pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(path)
    try:
        ds = sanitize_chunks(clean_for_zarr(ds))
        # Pass the URL rather than a mapper: fsspec then creates intermediate
        # paths, which get_mapper does not do for a local destination.
        ds.to_zarr(target, mode="w", consolidated=True,
                   zarr_format=ZARR_FORMAT,
                   storage_options=storage_options or None)
    finally:
        ds.close()
    print(f"  wrote  {path.name} -> {target}")
    return "converted"


def copy_zarr(path: pathlib.Path, dest: str, fs, overwrite: bool,
              dry_run: bool) -> str:
    """Copy an existing zarr store verbatim.

    Re-opening and rewriting would be slower and could silently change chunking
    or dtypes, so the bytes are copied directly.
    """
    target = f"{dest}/{path.name}"
    if _exists(fs, target, dry_run):
        if not overwrite:
            print(f"  SKIP  {path.name}/ -> {target} (exists)")
            return "skipped"
        if not dry_run:
            fs.rm(target, recursive=True)
    files = [f for f in path.rglob("*") if f.is_file()]
    n = len(files)
    if dry_run:
        print(f"  would copy  {path.name}/ ({n} files) -> {target}")
        return "dry"

    # Copy file by file with explicit relative paths. fsspec's recursive put
    # differs between backends about whether the source directory name is
    # appended to the destination, which silently nests the store one level.
    for i, f in enumerate(files, 1):
        rel = f.relative_to(path).as_posix()
        dst = f"{target}/{rel}"
        if not target.startswith("s3://"):
            pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        fs.put(str(f), dst)
        if n > 200 and i % 200 == 0:
            print(f"    {i}/{n}")
    print(f"  copied  {path.name}/ ({n} files) -> {target}")
    return "copied"


def copy_plain(path: pathlib.Path, dest: str, fs, overwrite: bool,
               dry_run: bool) -> str:
    target = f"{dest}/{path.name}"
    if _exists(fs, target, dry_run) and not overwrite:
        print(f"  SKIP  {path.name} -> {target} (exists)")
        return "skipped"
    if dry_run:
        print(f"  would upload  {path.name} -> {target}")
        return "dry"
    fs.put(str(path), target)
    print(f"  uploaded  {path.name} -> {target}")
    return "copied"


def report_duplicates(paths) -> None:
    """Flag files whose names differ only by hyphen/underscore.

    data/ currently holds both metrics_wrf-gcm_daily.nc and
    metrics_wrf_gcm_daily.nc, which are almost certainly the same run written
    under two naming conventions. They would land as two separate stores.
    """
    seen: dict[str, list[str]] = {}
    for p in paths:
        key = p.stem.replace("-", "_")
        seen.setdefault(key, []).append(p.name)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print("\nWARNING: names differing only by - vs _ :")
        for k, v in sorted(dupes.items()):
            print(f"  {k}: {v}")
        print("  These upload as separate stores. Delete the stale one first "
              "if they are duplicates.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", default=None, help="AWS profile name")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only these file names")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    if not src.is_dir():
        print(f"no such directory: {src}")
        return 1

    dest = a.dest.rstrip("/")

    # A non-s3 destination writes to local disk, which is the way to rehearse
    # the whole conversion (string dtypes, encoding, chunking) before touching
    # the bucket: --dest /tmp/publish_test
    if dest.startswith("s3://"):
        import s3fs
        fs = (s3fs.S3FileSystem(profile=a.profile) if a.profile
              else s3fs.S3FileSystem())
    else:
        import fsspec
        fs = fsspec.filesystem("file")
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

    entries = sorted(src.iterdir())
    if a.only:
        entries = [p for p in entries if p.name in set(a.only)]

    ncs = [p for p in entries if p.suffix == ".nc"]
    zarrs = [p for p in entries if p.suffix == ".zarr" and p.is_dir()]
    others = [p for p in entries
              if p not in ncs and p not in zarrs and p.is_file()]

    print(f"source      : {src.resolve()}")
    print(f"destination : {dest}")
    print(f"mode        : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}")
    print(f"found       : {len(ncs)} netCDF, {len(zarrs)} zarr, {len(others)} other\n")

    report_duplicates(ncs)

    tally: dict[str, int] = {}

    def bump(k):
        tally[k] = tally.get(k, 0) + 1

    so = {"profile": a.profile} if (dest.startswith("s3://") and a.profile) else None

    print("netCDF -> zarr")
    for p in ncs:
        try:
            bump(convert_netcdf(p, dest, fs, a.overwrite, a.dry_run,
                                storage_options=so))
        except Exception as e:                      # keep going on one bad file
            print(f"  FAILED {p.name}: {type(e).__name__}: {e}")
            bump("failed")

    if zarrs:
        print("\nzarr stores (copied verbatim)")
        for p in zarrs:
            try:
                bump(copy_zarr(p, dest, fs, a.overwrite, a.dry_run))
            except Exception as e:
                print(f"  FAILED {p.name}: {type(e).__name__}: {e}")
                bump("failed")

    if others:
        print("\nother files")
        for p in others:
            try:
                bump(copy_plain(p, dest, fs, a.overwrite, a.dry_run))
            except Exception as e:
                print(f"  FAILED {p.name}: {type(e).__name__}: {e}")
                bump("failed")

    print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    if not a.dry_run:
        if dest.startswith("s3://"):
            print(f"\nverify with:  aws s3 ls {dest}/")
        else:
            print(f"\nverify with:  ls {dest}/")
    return 1 if tally.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
