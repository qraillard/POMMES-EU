#!/usr/bin/env python3
"""Download the POMMES-EU input data from Zenodo into data/.

    python fetch_data.py                    download every file
    python fetch_data.py --list             list the files of the record
    python fetch_data.py --only demand.csv  download only the given files

Files already in place with the expected MD5 checksum are skipped and
interrupted downloads are resumed. Only the standard library is needed.
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Version of the Zenodo record holding the input data matching this code.
DOI = "10.5281/zenodo.23662766"
RECORD_ID = DOI.rsplit(".", 1)[1]
ZENODO_URL = os.environ.get("ZENODO_URL", "https://zenodo.org")  # overridden in tests only
CHUNK = 1 << 20
RETRIES = 4


def open_url(url, headers=None):
    """Open a URL, retrying on server errors and connection problems."""
    request = urllib.request.Request(url, headers=headers or {})
    for attempt in range(RETRIES):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as err:
            if (err.code < 500 and err.code != 429) or attempt == RETRIES - 1:
                raise
        except OSError:
            if attempt == RETRIES - 1:
                raise
        time.sleep(2 * 2**attempt)


def md5sum(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(record_id):
    """Return the title of the record and the description of its files."""
    with open_url(f"{ZENODO_URL}/api/records/{record_id}") as response:
        record = json.load(response)
    entries = record.get("files")
    if not isinstance(entries, list) or not entries:
        sys.exit(f"No file found in Zenodo record {record_id}.")
    files = []
    for entry in entries:
        name = entry["key"]
        if Path(name).name != name:
            sys.exit(f"Refusing suspicious file name in the record: {name!r}")
        url = (entry.get("links") or {}).get("self") or (
            f"{ZENODO_URL}/records/{record_id}/files/{urllib.parse.quote(name)}?download=1"
        )
        files.append(
            {
                "name": name,
                "size": entry.get("size"),
                "md5": entry.get("checksum", "").removeprefix("md5:"),
                "url": url,
            }
        )
    return record.get("metadata", {}).get("title", ""), files


def is_complete(path, item):
    """Whether path already holds the file described by item."""
    if not path.exists() or (item["size"] and path.stat().st_size != item["size"]):
        return False
    return not item["md5"] or md5sum(path) == item["md5"]


def download(item, dest):
    target = dest / item["name"]
    size = item["size"]
    megabytes = f"{size / 1e6:,.1f} MB" if size else "unknown size"
    if is_complete(target, item):
        print(f"  {item['name']}: already present, skipped")
        return
    print(f"  {item['name']} ({megabytes})")
    part = target.with_name(target.name + ".part")
    for attempt in range(RETRIES):
        done = part.stat().st_size if part.exists() else 0
        try:
            with open_url(item["url"], {"Range": f"bytes={done}-"} if done else None) as response:
                if done and response.status != 206:  # the server ignored the Range header
                    done = 0
                with open(part, "ab" if done else "wb") as out:
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
            break
        except urllib.error.HTTPError as err:
            if err.code != 416:
                raise
            part.unlink(missing_ok=True)  # nothing to resume from: start over
        except (OSError, http.client.HTTPException):
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * 2**attempt)  # connection cut: resume from what we have
    if (size and part.stat().st_size != size) or (item["md5"] and md5sum(part) != item["md5"]):
        part.unlink()
        sys.exit(f"Checksum mismatch for {item['name']}: the partial file was removed, run the script again.")
    part.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the files of the record and exit")
    parser.add_argument("--only", nargs="+", metavar="FILE", help="download only these files")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "data", help="destination folder (default: data/ next to this script)")
    parser.add_argument("--record", default=RECORD_ID, help="Zenodo record id (default: %(default)s, doi:" + DOI + ")")
    args = parser.parse_args()

    title, files = list_files(args.record)
    print(f"Zenodo record {args.record}: {title}")
    if args.only:
        unknown = sorted(set(args.only) - {f["name"] for f in files})
        if unknown:
            sys.exit("Not in the record: " + ", ".join(unknown))
        files = [f for f in files if f["name"] in args.only]
    if args.list:
        for item in files:
            print(f"  {item['name']:45s} {(item['size'] or 0) / 1e6:>10,.1f} MB")
        return
    args.dest.mkdir(parents=True, exist_ok=True)
    for item in files:
        download(item, args.dest)
    print(f"Done: {len(files)} file(s) in {args.dest}")


if __name__ == "__main__":
    main()
