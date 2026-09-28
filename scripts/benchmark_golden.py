"""End-to-end golden-dataset benchmark.

Stages the golden figures' images, runs the KG-extraction pipeline over
them, scores every output against `data/golden_dataset/ground_truth/`, and
writes the full evaluation + error-analysis reports:

    <out>/inputs/                 staged images (symlinks)
    <out>/final_graphs/           pipeline outputs (+ .critique.json when unvalidated)
    <out>/perf_log.json           per-figure pipeline status/timings
    <out>/evaluation/REPORT.md    human-readable benchmark report
    <out>/evaluation/error_analysis.json, summary.json, figures/...

Usage:
    uv run python scripts/benchmark_golden.py --limit 3          # smoke run
    uv run python scripts/benchmark_golden.py --model your-vision-model \
        --text-model your-text-model
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv

# The benchmark is defined at 300 DPI. Running the same pipeline on lower-
# resolution copies of the figures changes the results, so never point this
# anywhere but the canonical set.
DEFAULT_IMAGES_DIR = os.path.join(
    "data", "golden_dataset", "images_300dpi", "images"
)
DEFAULT_GOLDEN_DIR = os.path.join("data", "golden_dataset", "ground_truth")


def stage_images(images_dir: str, golden_dir: str, inputs_dir: str, limit: int | None) -> list[str]:
    """Symlinks the image of every golden figure into ``inputs_dir``.

    Only figures with BOTH a golden graph and an image are benchmarked;
    anything one-sided is reported and skipped.
    """
    os.makedirs(inputs_dir, exist_ok=True)
    stems = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(golden_dir, "*.json"))
    )
    if limit:
        stems = stems[:limit]

    staged, missing = [], []
    for stem in stems:
        for ext in (".png", ".jpg"):
            src = os.path.join(images_dir, stem + ext)
            if os.path.isfile(src):
                link = os.path.join(inputs_dir, stem + ext)
                if not os.path.lexists(link):
                    os.symlink(os.path.abspath(src), link)
                staged.append(stem)
                break
        else:
            missing.append(stem)

    if missing:
        print(f"WARNING: no image found for {len(missing)} golden figures:")
        for stem in missing:
            print(f"  {stem}")
    return staged


def run_meta_from_perf(perf_log: list[dict], settings, duration: float) -> dict:
    statuses = Counter(item["status"] for item in perf_log)
    times = [item["inference_time"] for item in perf_log if item["inference_time"]]
    return {
        "date": f"{datetime.now():%Y-%m-%d %H:%M}",
        "vision model": f"{settings.vision_provider}/{settings.vision_model}",
        "text model": f"{settings.text_provider}/{settings.text_model}",
        "figures": len(perf_log),
        "pipeline statuses": ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())),
        "correction retries (total)": sum(item.get("correction_retries", 0) for item in perf_log),
        "mean inference time": f"{sum(times) / len(times):.1f}s" if times else "n/a",
        "total wall time": f"{duration / 60:.1f} min",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the pipeline on the golden dataset")
    parser.add_argument("--images", default=DEFAULT_IMAGES_DIR, help="Directory with the golden figures' images")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("-o", "--output", default=None, help="Benchmark dir (default: outputs/benchmark_<timestamp>)")
    parser.add_argument("--provider", choices=["openai"], default=None)
    parser.add_argument("--model", default=None, help="Vision-agent model override")
    parser.add_argument("--text-provider", choices=["openai"], default=None)
    parser.add_argument("--text-model", default=None, help="Text-agent model override")
    # propagation is the matcher every published number uses; anchor stalls on
    # repeated blocks (ResNet stages) and is kept only for comparison.
    parser.add_argument("--matcher", choices=["propagation", "anchor", "both"], default="propagation")
    parser.add_argument("--workers", type=int, default=1,
                        help="Run N figures concurrently. 4 is the tested value for a full "
                             "run; the endpoint rate-limits on request RATE, not worker "
                             "count, so short text-only calls need a lower number than "
                             "long vision calls")
    parser.add_argument("--limit", type=int, default=None, help="Benchmark only the first N figures (smoke run)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Re-evaluate an existing benchmark dir without re-running the pipeline")
    args = parser.parse_args()

    load_dotenv()

    from kg_pipeline.eval.evaluate import evaluate_run
    from kg_pipeline.eval.report import format_summary_table
    from kg_pipeline.workflow.build import build_pipeline
    from kg_pipeline.runner import run_batch, run_batch_parallel
    from kg_pipeline.registry import build_deps
    from kg_pipeline.settings import get_settings

    settings = get_settings()
    overrides = {}
    if args.provider:
        overrides["vision_provider"] = args.provider
        overrides["text_provider"] = args.text_provider or args.provider
    elif args.text_provider:
        overrides["text_provider"] = args.text_provider
    if args.model:
        overrides["vision_model"] = args.model
    if args.text_model:
        overrides["text_model"] = args.text_model
    if overrides:
        settings = settings.model_copy(update=overrides)

    out_dir = args.output or os.path.join(
        "outputs", f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_meta = {}

    if args.skip_generation:
        perf_path = os.path.join(out_dir, "perf_log.json")
        if os.path.isfile(perf_path):
            with open(perf_path) as f:
                run_meta = run_meta_from_perf(json.load(f), settings, 0.0)
                run_meta.pop("total wall time", None)
    else:
        staged = stage_images(args.images, args.golden, os.path.join(out_dir, "inputs"), args.limit)
        if not staged:
            sys.exit("Nothing to benchmark: no golden figure has an image.")

        try:
            deps = build_deps(settings)
        except ValueError as e:
            sys.exit(str(e))  # missing key or model; the message says what to set
        pipeline = build_pipeline(deps)

        print(
            f"Benchmarking {len(staged)} figures: vision={settings.vision_provider}/"
            f"{settings.vision_model}, text={settings.text_provider}/{settings.text_model}"
            f" -> {out_dir}"
        )
        # Machine-readable run metadata, written up-front so live tooling (the
        # web overview, bench-status) can show the models while the run is
        # still generating.
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(
                {
                    "started": f"{datetime.now():%Y-%m-%d %H:%M}",
                    "vision_model": f"{settings.vision_provider}/{settings.vision_model}",
                    "text_model": f"{settings.text_provider}/{settings.text_model}",
                    "figures_staged": len(staged),
                },
                f,
                indent=2,
            )
        t0 = time.time()
        inputs_dir = os.path.join(out_dir, "inputs")
        perf_log = (
            run_batch_parallel(pipeline, inputs_dir, out_dir, args.workers)
            if args.workers > 1
            else run_batch(pipeline, inputs_dir, out_dir)
        )
        duration = time.time() - t0

        with open(os.path.join(out_dir, "perf_log.json"), "w") as f:
            json.dump(perf_log, f, indent=2)
        run_meta = run_meta_from_perf(perf_log, settings, duration)
        print(f"Generation done in {duration / 60:.1f} min: {run_meta['pipeline statuses']}")

    matchers = ("propagation", "anchor") if args.matcher == "both" else (args.matcher,)
    # With --limit, restrict scoring to the staged figures via a filtered
    # golden view; otherwise missing predictions for unstaged goldens would
    # (correctly but unhelpfully) drown the numbers.
    golden_dir = args.golden
    if args.limit:
        staged_stems = {
            os.path.splitext(f)[0] for f in os.listdir(os.path.join(out_dir, "inputs"))
        }
        golden_dir = os.path.join(out_dir, "golden_subset")
        os.makedirs(golden_dir, exist_ok=True)
        for stem in staged_stems:
            src = os.path.abspath(os.path.join(args.golden, stem + ".json"))
            link = os.path.join(golden_dir, stem + ".json")
            if os.path.isfile(src) and not os.path.lexists(link):
                os.symlink(src, link)

    summaries = evaluate_run(
        out_dir,
        golden_dir=golden_dir,
        matchers=matchers,
        out_dir=os.path.join(out_dir, "evaluation"),
        run_meta=run_meta,
    )

    print()
    print(format_summary_table(summaries))
    print(f"\nReport -> {os.path.join(out_dir, 'evaluation', 'REPORT.md')}")


if __name__ == "__main__":
    main()
