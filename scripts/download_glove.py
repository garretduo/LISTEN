#!/usr/bin/env python3
"""Download the GloVe 50-dimension word vectors that the embedding models use.

Input:  none (the download URLs are below).
Output: glove.6B.50d.txt in the data folder (paths.data_dir in configs/default.yaml).

The source is glove.6B.zip (about 860 MB). Only the 50d file (about 170 MB) is kept, and the
zip is deleted. The file is written only after the download succeeds, and an existing file
is left alone. `make results` runs this when the file is missing; `make glove` runs it alone.
"""

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
NAME = "glove.6B.50d.txt"
URLS = [  # tried in order
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
]


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data_dir = ROOT / yaml.safe_load(f)["paths"]["data_dir"]
    dest = data_dir / NAME
    if dest.exists():
        print(f"{dest.relative_to(ROOT)} already present, skipping")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:  # same folder, so the final rename is safe
        archive = Path(tmp) / "glove.6B.zip"
        for url in URLS:
            print(f"Downloading {url} (about 860 MB; only {NAME} is kept)", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "listen-download-script"})
                with urllib.request.urlopen(request, timeout=60) as response, open(archive, "wb") as out:
                    shutil.copyfileobj(response, out)
                break
            except OSError as error:
                print(f"Failed: {error}", file=sys.stderr)
        else:
            sys.exit("Could not download GloVe from any source.")

        with zipfile.ZipFile(archive) as zf:
            zf.extract(NAME, tmp)
        os.replace(Path(tmp) / NAME, dest)
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
