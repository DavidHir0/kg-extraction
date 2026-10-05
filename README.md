# kg-extraction

A pipeline that finds neural network architecture diagrams in research papers
and converts each one into a **knowledge graph**: the layers, data and
operations the figure draws become typed nodes, and the arrows between them
become directed edges.

It ships with a benchmark of 73 hand-labelled figures. With the two models listed
in `.env.example`, the pipeline scores **strict triplet F1 0.527 ± 0.038** and
node F1 0.796 ± 0.021 (mean of 3 runs). Other models will score differently.

## What it does

You can start from either:

- **PDF papers.** `extract` pulls every figure out of the PDFs
  ([pdffigures2](https://github.com/allenai/pdffigures2)), `classify` sorts the
  figures into 18 kinds (plots, tables, architecture diagrams, ...) with a
  ResNet50 and keeps the architecture diagrams, and `generate` turns each
  diagram into a graph.
- **Figure images you already have.** `generate` takes them directly. The
  benchmark works this way on its 73 figures.

For each figure, `generate` does this:

1. A **vision model** reads the image and writes down what is drawn: boxes and
   their text, arrows, the legend, and boxes that group other boxes.
2. **Rule-based steps** tidy that up: empty boxes that only pass an arrow
   through are removed, and a legend's meaning is attached to the boxes drawn in
   its style.
3. A **text model** fits it to the schema: every box gets a class and a type,
   and every arrow a role.
4. More rule-based steps unify wording (e.g. "Conv" and "conv2d") and split a
   box holding a chain like "Conv → BN → ReLU" into separate nodes.
5. A **validator** checks the graph against the schema. If it finds errors, the
   text model is asked to fix exactly those (up to 3 times).
6. The graph is saved as JSON.

A figure normally costs two model calls: one vision, one text.

## Output format

The full schema is in [`prompts/schema_v5.8.md`](prompts/schema_v5.8.md). In
short, a graph has:

- **nodes**, each with a `class`:
  `DATA` (inputs, outputs, feature maps), `LAYER` (convolution, linear,
  attention, ...), `MODIFIER` (activation, normalization, pooling, ...),
  `JUNCTION` (add, concat, ...) or `MACRO` (a named block); a `sub_type` within
  that class; `raw_text`, the text exactly as written in the figure; and
  `properties` such as `kernel` or `channels`, filled only when the figure shows
  them.
- **edges** from `source` to `target`, with a `role`: `forward`, `skip`,
  `condition` or `backward`.
- **groups**: drawn boxes or labelled regions, listing the nodes inside them.

**Example.** For panel (a) of Figure 1 in
[*ResNet in ResNet*](https://arxiv.org/abs/1603.08029v1) (a basic residual
block), the pipeline produced this part of the graph (the full graph of all four
panels has 36 nodes, 36 edges and 8 groups):

```json
{
  "nodes": [
    {"id": "input_a", "class": "DATA", "sub_type": "input", "raw_text": "",
     "properties": {"note": "implicit"}},
    {"id": "node_a_conv1", "class": "LAYER", "sub_type": "convolution",
     "raw_text": "conv", "style_id": "style_conv", "properties": {}},
    {"id": "node_a_conv2", "class": "LAYER", "sub_type": "convolution",
     "raw_text": "conv", "style_id": "style_conv", "properties": {}},
    {"id": "node_a_plus", "class": "JUNCTION", "sub_type": "add",
     "raw_text": "$\\oplus$", "style_id": "style_plus", "properties": {}}
  ],
  "edges": [
    {"source": "input_a", "target": "node_a_conv1", "role": "forward", "properties": {"labels": []}},
    {"source": "node_a_conv1", "target": "node_a_conv2", "role": "forward", "properties": {"labels": []}},
    {"source": "node_a_conv2", "target": "node_a_plus", "role": "forward", "properties": {"labels": []}},
    {"source": "input_a", "target": "node_a_plus", "role": "skip", "properties": {"labels": []}}
  ],
  "groups": [
    {"id": "group_a", "type": "container", "label": "(a)", "boundary_type": "implicit_header",
     "member_node_ids": ["node_a_conv1", "node_a_conv2", "node_a_plus"]}
  ]
}
```

## Install

Needs **Python 3.13+** and an API key for one vision-capable model. No GPU: the
pipeline is API calls and deterministic Python, and `torch` is not in the base
install.

With [uv](https://docs.astral.sh/uv/). This reproduces the exact environment the
benchmark numbers were measured in, because `uv.lock` pins every package and
`.python-version` pins the interpreter:

```bash
git clone https://github.com/DavidHir0/kg-extraction.git
cd kg-extraction
uv sync
cp .env.example .env          # then put a key in it, see below
```

With plain pip, you need **Python 3.13 or newer** yourself (`python3 --version`
to check; `pip install` refuses older versions):

```bash
git clone https://github.com/DavidHir0/kg-extraction.git
cd kg-extraction
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest            # only needed to run the tests
cp .env.example .env
```

pip ignores `uv.lock`, so it installs the newest compatible package versions
rather than the pinned ones. That is fine for using the pipeline; to reproduce a
benchmark number exactly, prefer uv.

**The commands below are written for uv.** With pip, activate the virtualenv
(`source .venv/bin/activate`) and leave out `uv run`: `uv run python main.py
check` becomes `python main.py check`, and `uv run pytest` becomes `pytest`. The
one exception is `uv sync --extra classify`, which is
`pip install -e ".[classify]"`.

## Configure a model

Two agents do the LLM work: one **vision** call that reads the figure, and one
**text** call that normalizes the result onto the schema. They can point at
different models.

**Any OpenAI-compatible endpoint** works: OpenAI, OpenRouter, Together, vLLM,
Ollama, LM Studio, a university endpoint. In `.env`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1    # or any compatible endpoint
KG_VISION_MODEL=your-vision-model            # must accept images
KG_TEXT_MODEL=your-text-model
```

**Both model names are required**, since which models exist depends on your
endpoint. The vision model matters most, because it does the actual reading of
the figure. `.env.example` lists the two models the benchmark number was
measured with.

Confirm the setup before spending a real call:

```bash
uv run python main.py check          # what is configured, and where the key goes
uv run python main.py check --ping   # one tiny request, proving it works
```

`check` prints **which endpoint your key will be sent to**. An OpenAI-compatible
client posts your key wherever `OPENAI_BASE_URL` points, so check that line.

You can also override the models per run without touching `.env`:

```bash
uv run python scripts/benchmark_golden.py \
  --provider openai --model your-vision-model \
  --text-provider openai --text-model your-text-model \
  --workers 4
```

To check the install without an API key:

```bash
uv run python main.py diagram        # renders the pipeline topology
uv run pytest -q                     # deterministic tests, no network
```

## Run it on figures

```bash
uv run python main.py generate -i path/to/figures/ -o outputs/my_run
```

`-i` must be a **folder**; every `.png` and `.jpg` in it is processed. Passing a
single image file finds nothing, so put even one figure in a folder of its own.
The finished graph is written to
`outputs/my_run/final_graphs/<figure-stem>.json`, and the raw vision output
is kept beside it in `debug_outputs/` so you can see what each stage did.

## Run it on papers

`generate` starts from figures you already have. To start from PDFs, two
upstream stages find the figures and keep only the architecture diagrams. They
need binary assets that are too large for git, attached to the
[`assets-v1` release](https://github.com/DavidHir0/kg-extraction/releases/tag/assets-v1)
and downloaded on demand:

```bash
scripts/fetch_assets.sh                    # both assets, into models/
scripts/fetch_assets.sh jar                # or one at a time
uv sync --extra classify                   # torch, for the classifier only
```

Then chain the three commands:

```bash
uv run python main.py extract  -i papers/ -o data/extracted       # PDFs -> figures
uv run python main.py classify -i data/extracted -o data/figures  # keep diagrams
uv run python main.py generate -i data/figures -o outputs/my_run  # figures -> graphs
```

- `extract` runs [pdffigures2](https://github.com/allenai/pdffigures2) and needs
  a Java runtime (`java -version` to check).
- `classify` runs a ResNet50 fine-tuned on the ACL figures dataset. It sorts
  figures into the 18 classes in `models/classification/classes.txt` and keeps
  the architecture diagrams, since most figures in a paper are plots and tables.
- `classify` reads from `<-i>/images/`, the layout `extract` writes. Pointing it
  at a bare folder of PNGs fails with `Input folder missing: .../images`; put
  them in an `images/` subdirectory.
- **`classify` deletes its `-o` folder before writing**, so point it at a new
  folder, never at one holding anything you want to keep.

## Run the benchmark

The 73 figures are not included in this repository. Download them from arXiv
first (details in `data/README.md`):

```bash
scripts/fetch_assets.sh jar
uv run python scripts/fetch_golden_figures.py
```

Then:

```bash
uv run python scripts/benchmark_golden.py --limit 3 --workers 4 -o outputs/bench_test  # 3 figures
uv run python scripts/benchmark_golden.py --workers 4 -o outputs/bench_r1              # all 73
uv run python main.py evaluate -r outputs/bench_r1                                     # re-score a run
```

Run `--limit 3` once before a full run. The script generates the graphs, scores
them and prints a table; the full report is written to
`<-o>/evaluation/REPORT.md`. `--workers` is how many figures run in parallel;
lower it if your endpoint returns HTTP 429 (rate limited).

**Reading the table**

- **Strict triplet F1** is the main metric: an edge counts only if both of its
  boxes and its role are right. Node F1 checks only the boxes, so it is higher
  and says less about the graph.
- **micro** pools the edges of all figures, so large figures weigh more.
  **macro** scores each figure on its own and then averages, so every figure
  weighs the same.
- **A figure without a graph counts as a total miss.** If a figure fails, for
  example on the request timeout (`KG_REQUEST_TIMEOUT`), re-running with the
  same `-o` skips finished figures and retries the rest. `evaluate` always
  scores all 73 figures, so on a `--limit 3` run it shows a score near zero;
  the benchmark script's own table covers only the figures it ran.

**How the 0.527 was measured:** strict triplet F1, micro, as the mean of 3 full
runs with all 73 figures completed. It uses the 300 DPI images and the corrected
labels in this repo (see `data/README.md`), with the models in `.env.example`.
Model outputs vary between identical runs, which is why it is a mean with a
spread.

## Credits and licences

- **Benchmark figures.** The 73 figures come from 45 arXiv papers. They are not
  distributed here; `scripts/fetch_golden_figures.py` downloads them from arXiv.
  `data/SOURCES.md` gives the full citation and arXiv licence of each paper, and
  `data/CITATIONS.bib` has the BibTeX entries. Copyright in the figures remains
  with the papers' authors.
- **pdffigures2** (`extract`). © the Allen Institute for AI, Apache-2.0. Please
  cite Clark and Divvala, *PDFFigures 2.0: Mining Figures from Research Papers*,
  JCDL 2016. The jar is built unmodified from source; it bundles other
  libraries, listed with their licences in `models/pdffigures2/NOTICE.md`.
- **Figure classifier** (`classify`). A ResNet-50 (He et al., CVPR 2016) from
  torchvision, started from torchvision's ImageNet-1K weights and fine-tuned on
  the ACL-Fig dataset (Karishma et al., 2023). Details and citations in
  `models/classification/NOTICE.md`.
- **Python packages** are installed from PyPI under their own licences.

## Licence

The code and the labels are licensed under the
[Apache License 2.0](LICENSE). Copyright 2026 David Hiroki Ortelt.

## Acknowledgements

Developed during a research internship at the National Institute of Informatics
(NII), Tokyo, supervised by Prof. Dr. Hideaki Takeda (NII) and Dr. Daniel
Kurzawe (University of Göttingen).
