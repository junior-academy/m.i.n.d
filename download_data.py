"""
Robust downloader/helper for:
- BCI Competition IV Dataset 2a
- BCI Competition III Dataset IIIa

Important:
1. Some sources may require registration or manual download.
2. This script works best when you provide direct URL templates or a public page
   that exposes actual file links.
3. If protected downloads block automation, use this script to verify folder
   structure and skip already-downloaded files.

Example usage:
    python download_bci.py --dataset iv2a --mode direct \
        --url-template "https://example.com/path/{subject}.gdf"

    python download_bci.py --dataset iiia --mode scrape \
        --page-url "https://example.com/downloads.html"

    python download_bci.py --dataset all --mode manual-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path("data")
ROOT.mkdir(exist_ok=True)

DATASETS = {
    "iv2a": {
        "name": "BCI Competition IV Dataset 2a",
        "folder": ROOT / "bci_competition_iv_2a",
        "subjects": [
            "A01T", "A01E",
            "A02T", "A02E",
            "A03T", "A03E",
            "A04T", "A04E",
            "A05T", "A05E",
            "A06T", "A06E",
            "A07T", "A07E",
            "A08T", "A08E",
            "A09T", "A09E",
        ],
        "extensions": [".gdf", ".mat", ".zip"],
    },
    "iiia": {
        "name": "BCI Competition III Dataset IIIa",
        "folder": ROOT / "bci_competition_iii_iiia",
        "subjects": ["k3", "k6", "l1"],
        "extensions": [".gdf", ".mat", ".zip"],
    },
}

USER_AGENT = "Mozilla/5.0 (compatible; Kevin-BCI-Downloader/1.0)"

def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def session_with_headers() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def filename_from_response(url: str, response: requests.Response, fallback: str) -> str:
    cd = response.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    if m:
        return m.group(1)
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or fallback

def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Skipping existing file: {dest.name}")
        return True

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        print(f"  Downloaded: {dest.name}")
        return True
    except Exception as e:
        print(f"  Failed: {url}")
        print(f"    Error: {e}")
        return False

def normalize_subject(subject: str) -> str:
    return subject.lower().replace("_", "").replace("-", "")

def link_matches_subject(link_text: str, href: str, subject: str, exts: list[str]) -> bool:
    blob = f"{link_text} {href}".lower()
    subject_norm = normalize_subject(subject)
    blob_norm = normalize_subject(blob)
    has_subject = subject_norm in blob_norm
    has_ext = any(ext in href.lower() or ext in blob for ext in exts)
    return has_subject and has_ext

def scrape_links(page_url: str, subjects: list[str], exts: list[str]) -> dict[str, str]:
    s = session_with_headers()
    r = s.get(page_url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    subject_links = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)
        full_url = urljoin(page_url, href)

        for subject in subjects:
            if subject not in subject_links and link_matches_subject(text, href, subject, exts):
                subject_links[subject] = full_url

    return subject_links

def direct_links_from_template(url_template: str, subjects: list[str]) -> dict[str, str]:
    return {subject: url_template.format(subject=subject) for subject in subjects}

def choose_extension(url: str, default_ext: str = ".gdf") -> str:
    path = urlparse(url).path.lower()
    for ext in [".gdf", ".mat", ".zip"]:
        if path.endswith(ext):
            return ext
    return default_ext

def save_manifest(dataset_key: str, records: list[dict]):
    out = DATASETS[dataset_key]["folder"] / "download_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved manifest: {out}")

def run_dataset(dataset_key: str, mode: str, url_template: str | None, page_url: str | None):
    cfg = DATASETS[dataset_key]
    folder = cfg["folder"]
    ensure_folder(folder)

    print("\n" + "=" * 72)
    print(f"{cfg['name']}")
    print("=" * 72)
    print(f"Destination: {folder}")

    subjects = cfg["subjects"]
    exts = cfg["extensions"]

    if mode == "manual-check":
        found = []
        for subject in subjects:
            matched = list(folder.glob(f"{subject}.*")) + list(folder.glob(f"{subject.lower()}.*"))
            if matched:
                found.extend(matched)
        print(f"Found {len(found)} existing files in {folder}")
        for f in sorted(found):
            print(f"  {f.name}")
        missing = [s for s in subjects if not any(folder.glob(f"{s}.*")) and not any(folder.glob(f"{s.lower()}.*"))]
        if missing:
            print("\nMissing subjects/files:")
            for s in missing:
                print(f"  {s}")
        return

    if mode == "direct":
        if not url_template:
            raise ValueError("direct mode requires --url-template")
        subject_links = direct_links_from_template(url_template, subjects)

    elif mode == "scrape":
        if not page_url:
            raise ValueError("scrape mode requires --page-url")
        subject_links = scrape_links(page_url, subjects, exts)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    manifest = []
    success_count = 0

    for subject in subjects:
        url = subject_links.get(subject)
        if not url:
            print(f"  No link found for {subject}")
            manifest.append({
                "subject": subject,
                "status": "missing_link",
                "url": None,
                "saved_as": None,
            })
            continue

        ext = choose_extension(url, default_ext=".gdf")
        dest = folder / f"{subject}{ext}"

        ok = download_file(url, dest)
        manifest.append({
            "subject": subject,
            "status": "downloaded" if ok else "failed",
            "url": url,
            "saved_as": str(dest),
        })
        if ok:
            success_count += 1
        time.sleep(0.5)

    print(f"\nCompleted {dataset_key}: {success_count}/{len(subjects)} successful")
    save_manifest(dataset_key, manifest)

def parse_args():
    p = argparse.ArgumentParser(description="Downloader/helper for BCI competition datasets.")
    p.add_argument(
        "--dataset",
        choices=["iv2a", "iiia", "all"],
        required=True,
        help="Which dataset to process.",
    )
    p.add_argument(
        "--mode",
        choices=["direct", "scrape", "manual-check"],
        required=True,
        help="direct = use URL template, scrape = parse links from page, manual-check = verify local files",
    )
    p.add_argument(
        "--url-template",
        default=None,
        help='Example: "https://host/path/{subject}.gdf"',
    )
    p.add_argument(
        "--page-url",
        default=None,
        help="Public HTML page containing actual file links to scrape.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    targets = ["iv2a", "iiia"] if args.dataset == "all" else [args.dataset]
    for dataset_key in targets:
        try:
            run_dataset(
                dataset_key=dataset_key,
                mode=args.mode,
                url_template=args.url_template,
                page_url=args.page_url,
            )
        except Exception as e:
            print(f"\nError while processing {dataset_key}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()