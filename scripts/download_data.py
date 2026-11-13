#!/usr/bin/env python3
"""Download the Learned Hands label files into data/.

Input:  configs/default.yaml (data.base_url, data.files, paths.data_dir).
Output: one CSV per entry of data.files, written into the data folder.

Each download goes to a temporary file, is checked (header and row count against
the config), and only then renamed into place, so a failed or partial download
never leaves a bad CSV behind. A file that is already present with the same size
as the remote copy is skipped.

Usage:
    python scripts/download_data.py            # download what is missing
    python scripts/download_data.py --dry-run  # list files and remote sizes only
    python scripts/download_data.py --force    # download everything again

The data is CC BY-NC-SA 4.0 and must not be redistributed, so this script refuses
to write anything unless git says the data folder is ignored.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
USER_AGENT = "listen-download-script"  # generic on purpose: no personal details
N_CATEGORIES = 20  # label columns after the id and text columns
TIMEOUT_SECONDS = 60

LICENSE_NOTICE = (
    "Learned Hands data: CC BY-NC-SA 4.0 (Suffolk LIT Lab and Stanford Legal Design Lab).\n"
    "Keep it local and out of git. Do not redistribute it."
)


def load_config():
    """Read configs/default.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_ignores(path):
    """Return True if git ignores path, False if it does not, None if git can't tell.

    None covers three cases: git is not installed, ROOT is not a git repository
    (for example a zip download), or the path lies outside the repository.
    """
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return None
    probe = relative / "probe.csv"  # a path inside the folder, so a "/data/" rule matches
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", str(probe)],
            capture_output=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def remote_size(url):
    """Return the Content-Length of url from a HEAD request, or None if absent."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def check_csv(path, expected_rows, id_column, text_column):
    """Raise ValueError if the CSV's header or row count is not what the config expects."""
    csv.field_size_limit(2**31 - 1)  # posts are long; the default field limit is small
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    problems = []
    if header[:2] != [id_column, text_column]:
        problems.append(f"first columns are {header[:2]}, expected {[id_column, text_column]}")
    if len(header) != N_CATEGORIES + 2:
        problems.append(f"{len(header)} columns, expected {N_CATEGORIES + 2}")
    if rows != expected_rows:
        problems.append(f"{rows} rows, expected {expected_rows}")
    if problems:
        raise ValueError("; ".join(problems))


def download(url, dest, expected_rows, id_column, text_column):
    """Fetch url into a temporary file, validate it, then rename it to dest."""
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            request, timeout=TIMEOUT_SECONDS
        ) as response:
            shutil.copyfileobj(response, out)
        check_csv(tmp, expected_rows, id_column, text_column)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Download the Learned Hands label files.")
    parser.add_argument("--dry-run", action="store_true", help="list files and remote sizes; write nothing")
    parser.add_argument("--force", action="store_true", help="download even if the file is already present")
    args = parser.parse_args()

    config = load_config()
    data_config = config["data"]
    data_dir = ROOT / config["paths"]["data_dir"]

    # Guard: never write licensed data into a folder git could add.
    ignored = git_ignores(data_dir)
    if ignored is False:
        message = f"{data_dir.relative_to(ROOT)}/ is not gitignored. Add it to .gitignore first."
        if not args.dry_run:
            sys.exit(f"Refusing to download: {message}")
        print(f"WARNING: {message}")
    elif ignored is None:
        print("Note: could not check whether the data folder is gitignored.")

    print(LICENSE_NOTICE)
    if not args.dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for key, spec in data_config["files"].items():
        url = urllib.parse.urljoin(data_config["base_url"], spec["name"])
        dest = data_dir / spec["name"]
        try:
            size = remote_size(url)
            size_text = f"{size / 1e6:.1f} MB" if size else "size unknown"
            if args.dry_run:
                state = "present" if dest.exists() else "missing"
                print(f"{key}: {spec['name']} ({size_text}), locally {state}")
            elif dest.exists() and size is not None and dest.stat().st_size == size and not args.force:
                print(f"{key}: {spec['name']} already present, skipping")
            else:
                print(f"{key}: downloading {spec['name']} ({size_text})")
                download(url, dest, spec["rows"], data_config["id_column"], data_config["text_column"])
                print(f"{key}: ok, {spec['rows']} rows")
        except (urllib.error.URLError, ValueError, OSError) as error:
            failed += 1
            print(f"{key}: FAILED: {error}", file=sys.stderr)

    if failed:
        sys.exit(f"{failed} file(s) failed.")


if __name__ == "__main__":
    main()
