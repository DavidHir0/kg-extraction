"""Download the 73 benchmark figures from arXiv.

The figures are not redistributed with the labels. This script downloads each
source paper from arXiv, runs pdffigures2 at 300 DPI and keeps the figures
that have a golden label, in the folder the benchmark reads by default.

    scripts/fetch_assets.sh jar                    # pdffigures2, once
    uv run python scripts/fetch_golden_figures.py

Needs Java (pdffigures2) and pdftoppm from poppler (one figure pdffigures2
does not detect is cut from its page instead). Checked on 2026-09-28: all 73
rebuilt figures are pixel-identical to the benchmark images except three that
come out at a higher resolution.

With ``--compare DIR`` it also checks every rebuilt image against an existing
copy (same size, pixel difference) and writes ``compare.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import PIL.Image

from kg_pipeline.pdf_figures.pdf_extractor import extract_figures
from kg_pipeline.settings import get_settings

GOLDEN_DIR = os.path.join("data", "golden_dataset", "ground_truth")
USER_AGENT = "kg-extraction golden-figure fetcher (research use)"

# The labels' filenames say v1, but these figures were taken from a later
# version of the paper (checked pixel-identical against arXiv v2).
VERSION_OVERRIDE = {"1612.00593v1": "1612.00593v2", "1706.03762v1": "1706.03762v2"}

# pdffigures2 does not detect these figures. Cut them from the page instead:
# 1-based page number and box in PDF points (x0, y0, x1, y1, origin top-left).
MANUAL_CROPS = {
    "2207.02390v1_Swin Deformable Attention U-Net Transformer SDAUT for Explainable Fast MRI-Figure1-1":
        (2, (144.2, 394.8, 484.6, 567.6)),
}


def paper_of(stem: str) -> str:
    """Golden stem ``<paper>-Figure<N>-<M>`` -> ``<paper>`` (the PDF basename)."""
    return stem.split("-Figure")[0]


def arxiv_id(paper: str) -> str:
    """``2111.10591v1_AGA-GAN ...`` -> ``2111.10591v1`` (or its pinned version)."""
    aid = paper.split("_", 1)[0]
    return VERSION_OVERRIDE.get(aid, aid)


def crop_page(pdf: str, page: int, box: tuple, dpi: int, dest: str) -> None:
    x0, y0, x1, y1 = (round(v * dpi / 72) for v in box)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        "-x", str(x0), "-y", str(y0), "-W", str(x1 - x0), "-H", str(y1 - y0),
                        pdf, os.path.join(tmp, "crop")], check=True)
        shutil.move(os.path.join(tmp, "crop.png"), dest)


def download(aid: str, dest: str, attempts: int = 4) -> None:
    req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError as e:  # timeouts and dropped connections; arXiv is sometimes slow
            if attempt == attempts:
                raise RuntimeError(f"{aid}: download failed {attempts} times ({e}); re-run to resume") from e
            print(f"    {aid}: {e}; retrying in {10 * attempt} s")
            time.sleep(10 * attempt)
    with open(dest + ".part", "rb") as f:
        if f.read(5) != b"%PDF-":
            raise RuntimeError(f"{aid}: response is not a PDF")
    os.replace(dest + ".part", dest)


def compare(a: str, b: str) -> dict:
    with PIL.Image.open(a) as ia, PIL.Image.open(b) as ib:
        if ia.size != ib.size:
            return {"same_size": False, "new": ia.size, "old": ib.size}
        xa = np.asarray(ia.convert("RGB"), dtype=np.int16)
        xb = np.asarray(ib.convert("RGB"), dtype=np.int16)
    d = np.abs(xa - xb)
    return {"same_size": True, "identical": bool(d.max() == 0), "mean_abs_diff": float(d.mean()),
            "pct_pixels_changed": float((d.max(axis=2) > 0).mean() * 100)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("data", "golden_dataset", "images_300dpi"),
                    help="figures are written to <out>/images/")
    ap.add_argument("--golden", default=GOLDEN_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cache", default=os.path.join("data", "pdf_cache"), help="downloaded PDFs")
    ap.add_argument("--compare", help="existing image folder to check the rebuild against")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{args.golden}/*.json"))
    papers = sorted({paper_of(s) for s in stems})
    print(f"{len(stems)} golden figures across {len(papers)} papers")

    os.makedirs(args.cache, exist_ok=True)
    for i, paper in enumerate(papers, 1):
        pdf = os.path.join(args.cache, arxiv_id(paper) + ".pdf")
        if os.path.isfile(pdf):
            continue
        print(f"  [{i}/{len(papers)}] downloading {arxiv_id(paper)}")
        download(arxiv_id(paper), pdf)
        time.sleep(3)  # arXiv asks for one request every 3 s

    # pdffigures2 names figures after the PDF basename, so stage each PDF under
    # the paper name the golden stems use.
    work = os.path.join(args.cache, "pdffigures2")
    with tempfile.TemporaryDirectory() as pdf_dir:
        for paper in papers:
            os.symlink(os.path.abspath(os.path.join(args.cache, arxiv_id(paper) + ".pdf")),
                       os.path.join(pdf_dir, paper + ".pdf"))
        print(f"Running pdffigures2 @ {args.dpi} DPI")
        extract_figures(pdf_dir, work, str(get_settings().pdffigures2_jar), dpi=args.dpi)

    images = os.path.join(args.out, "images")
    os.makedirs(images, exist_ok=True)
    report, missing = {}, []
    for stem in stems:
        src = os.path.join(work, "images", stem + ".png")
        if stem in MANUAL_CROPS:
            page, box = MANUAL_CROPS[stem]
            src = os.path.join(work, stem + ".manual.png")
            crop_page(os.path.join(args.cache, arxiv_id(paper_of(stem)) + ".pdf"), page, box, args.dpi, src)
        if not os.path.isfile(src):
            missing.append(stem)
            continue
        shutil.copy(src, os.path.join(images, stem + ".png"))
        if args.compare and os.path.isfile(os.path.join(args.compare, stem + ".png")):
            report[stem] = compare(src, os.path.join(args.compare, stem + ".png"))

    print(f"\nRebuilt {len(stems) - len(missing)}/{len(stems)} figures → {images}")
    for s in missing:
        print(f"  missing: {s}")
    if args.compare:
        same = [s for s, r in report.items() if r.get("identical")]
        close = [s for s, r in report.items() if r["same_size"] and not r["identical"]]
        diff = [s for s, r in report.items() if not r["same_size"]]
        print(f"vs {args.compare}: {len(same)} pixel-identical, {len(close)} same size but differ, "
              f"{len(diff)} different size")
        for s in close:
            print(f"  differs: {s[:70]}  ({report[s]['pct_pixels_changed']:.2f}% px)")
        for s in diff:
            print(f"  size:    {s[:70]}  new {report[s]['new']} old {report[s]['old']}")
        json.dump({"missing": missing, "figures": report}, open(os.path.join(args.out, "compare.json"), "w"), indent=1)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
