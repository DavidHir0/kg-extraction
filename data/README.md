# The golden dataset

73 neural-architecture figures, each with a hand-made knowledge graph.

```
ground_truth/                73 labels
images_300dpi/images/        73 figures at 300 DPI (not included; see below)
SOURCES.md                   citations and arXiv licences of the 45 source papers
CITATIONS.bib                BibTeX for those papers
```

Labels and figures share a filename stem, so `ground_truth/<stem>.json`
describes `images_300dpi/images/<stem>.png`.

## Getting the figures

The figures are not in this repository. Many of the source papers carry
arXiv's default licence, which does not allow redistribution. Download them
from arXiv instead:

```bash
scripts/fetch_assets.sh jar                    # pdffigures2, once
uv run python scripts/fetch_golden_figures.py  # 45 PDFs -> 73 figures
```

This needs Java and `pdftoppm` (from poppler). It downloads one PDF every 3
seconds, as arXiv asks, into `data/pdf_cache/`, and writes the figures to
`images_300dpi/images/`.

Three details, so the result is not a surprise:

- For *PointNet* and *Attention Is All You Need* the figures come from arXiv
  **v2**, although the filenames say v1. The two versions draw different
  figures, and the labels describe the v2 ones.
- pdffigures2 does not detect SDAUT Figure 1. The script cuts it from page 2
  of the PDF with a fixed box.
- AGA-GAN Figure 1 and SDAUT Figures 1 and 2 come out at a higher resolution
  than the images the published scores were measured on. The drawing is the
  same. The other 70 figures are pixel-identical to the benchmark images.

## What the labels encode

The target is **what the figure depicts**, not the architecture the figure came
from. A figure that does not draw its activations gets no activation nodes.

These are the corrected labels. An earlier label set differed in two ways: 9
nodes had the wrong type (an 'Attention Gate' is a `LAYER`), and 59 ReLU nodes
were recorded in GoogLeNet Fig 3 that the figure never draws. The two sets
differ on 4 of the 73 figures. Numbers measured on the earlier set are
systematically lower and are not comparable with numbers measured on these.

## Resolution

The benchmark is defined at **300 DPI**. Running the same pipeline on the same
figures at 150 DPI once silently changed our results.
`scripts/benchmark_golden.py` defaults to the correct directory, so don't
override `--images` unless you mean to.

## Scoring

```bash
uv run python main.py evaluate -r outputs/your_run
```

Defaults to these labels, the propagation matcher, and a 0.35 similarity
threshold. The propagation matcher is the trusted one; the anchor matcher stalls
on ResNet-style repeated blocks.

Report **strict triplet F1** first. Node F1 flatters a system: an unconnected
node raises node F1 while moving strict F1 by exactly zero.

## Known label quirks

- **11% of components share a drawn rectangle** in the source annotations: an
  activation drawn on top of its convolution, or one box holding a sequence that
  was split into several components. This affects detection work, not graph
  scoring.
