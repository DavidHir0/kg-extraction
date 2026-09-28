import argparse
import json
import os
import sys

from dotenv import load_dotenv

from kg_pipeline.settings import get_settings

def cmd_extract(args, settings):
    from kg_pipeline.pdf_figures.pdf_extractor import extract_figures

    jar = settings.pdffigures2_jar
    if not os.path.exists(jar):  # a 19 MB binary: release asset, not in git
        raise SystemExit(
            "The extract command needs the pdffigures2 jar, which is not in "
            "this repository (it is a 19 MB binary).\n"
            "  scripts/fetch_assets.sh jar\n"
            f"downloads it to {jar}.\n"
            "It also needs a Java runtime on PATH (`java -version`).\n"
            "The benchmark does not use this command -- the 73 figures are "
            "already extracted under data/golden_dataset/."
        )

    result = extract_figures(args.input, args.output, str(jar), args.dpi)
    print(f"Extraction finished in {result['duration']:.2f}s -> {args.output}")


def cmd_classify(args, settings):
    try:
        from kg_pipeline.classification.classifier import classify_and_filter
    except ImportError as exc:  # torch is an optional extra
        raise SystemExit(
            "The classify command needs torch, which is not in the base install.\n"
            "  uv sync --extra classify      (or: pip install 'torch' 'torchvision')\n"
            f"[{exc}]"
        ) from exc

    if not os.path.exists(settings.classifier_weights):
        raise SystemExit(
            "The classify command needs the ResNet checkpoint, which is not in "
            "this repository (it is a 91 MB binary).\n"
            "  scripts/fetch_assets.sh weights\n"
            f"downloads it to {settings.classifier_weights}.\n"
            "The benchmark does not use this command."
        )

    perf_log = classify_and_filter(
        args.input,
        args.output,
        str(settings.classifier_weights),
        str(settings.classifier_classes),
        use_padding=args.padding,
    )
    print(f"Classified {len(perf_log)} images -> {args.output}")


def _settings_with_overrides(args, settings):
    """CLI flags override env/.env-provided pipeline model selection."""
    overrides = {}
    if args.provider:
        overrides["vision_provider"] = args.provider
        # Text agents follow the vision provider unless --text-provider is given.
        overrides["text_provider"] = args.text_provider or args.provider
    elif args.text_provider:
        overrides["text_provider"] = args.text_provider
    if args.model:
        overrides["vision_model"] = args.model
        overrides["text_model"] = args.text_model or args.model
    elif args.text_model:
        overrides["text_model"] = args.text_model
    return settings.model_copy(update=overrides) if overrides else settings


def cmd_generate(args, settings):
    from kg_pipeline.workflow.build import build_pipeline
    from kg_pipeline.runner import new_run_dir, run_batch
    from kg_pipeline.registry import build_deps

    settings = _settings_with_overrides(args, settings)
    try:
        deps = build_deps(settings)
    except ValueError as e:  # missing key or model; the message says what to set
        raise SystemExit(str(e)) from None
    graph = build_pipeline(deps)

    output_dir = args.output or new_run_dir()
    print(
        f"Starting KG generation: vision={settings.vision_provider}, "
        f"text={settings.text_provider} -> {output_dir}"
    )
    perf_log = run_batch(graph, args.input, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    perf_log_path = os.path.join(output_dir, "perf_log.json")
    with open(perf_log_path, "w") as f:
        json.dump(perf_log, f, indent=2)

    succeeded = sum(1 for item in perf_log if item["status"] == "success")
    print(f"Done: {succeeded}/{len(perf_log)} images validated clean. Log -> {perf_log_path}")


def cmd_diagram(args, settings):
    """Renders the pipeline topology without needing API keys."""
    from kg_pipeline.workflow.build import build_pipeline
    from kg_pipeline.registry import build_deps

    graph = build_pipeline(build_deps(settings, require_keys=False)).get_graph()

    if args.png:
        # Rendered via the mermaid.ink web service; needs internet access.
        with open(args.output or "pipeline_diagram.png", "wb") as f:
            f.write(graph.draw_mermaid_png())
        print(f"Wrote {args.output or 'pipeline_diagram.png'}")
        return

    mermaid = graph.draw_mermaid()
    out_path = args.output or "pipeline_diagram.mmd"
    with open(out_path, "w") as f:
        f.write(mermaid)
    print(mermaid)
    print(f"\nWrote {out_path} (paste into https://mermaid.live to render).")


def cmd_evaluate(args, settings):
    from kg_pipeline.eval.evaluate import evaluate_run
    from kg_pipeline.eval.matching import MatchConfig
    from kg_pipeline.eval.report import format_summary_table

    matchers = ("propagation", "anchor") if args.matcher == "both" else (args.matcher,)
    cfg = MatchConfig(threshold=args.threshold)
    out_dir = args.output or os.path.join(args.run, "evaluation")

    summaries = evaluate_run(
        args.run, golden_dir=args.golden, matchers=matchers, cfg=cfg, out_dir=out_dir
    )

    # The warning goes *above* the table: scoring covers the whole golden set,
    # so a partial run reads as a catastrophic score unless you see this first.
    first = summaries[matchers[0]]
    missing = first.get("missing_predictions")
    if missing:
        print(
            f"WARNING: {missing}/{first['figures']} golden figures have no prediction "
            "in this run and are counted as all-false-negatives.\n"
            "         The scores below are for the whole benchmark, not for the "
            "figures you ran.\n"
            "         For a partial run use: scripts/benchmark_golden.py --limit N\n"
        )
    print(format_summary_table(summaries))
    print(f"Full reports -> {out_dir}")


def cmd_check(args, settings):
    """Report whether this machine is configured to run the pipeline.

    Setting up an API key is the one step where a newcomer has no feedback
    until they spend a real vision call, so this answers the question directly:
    which key was found, in which variable, and WHICH ENDPOINT it will be sent
    to. That last one matters -- an OpenAI-compatible client will happily post
    your key wherever `base_url` points.
    """
    from kg_pipeline.settings import KEY_ENV_VARS

    ok = True
    print("Configuration\n")
    for role in ("vision", "text"):
        provider = getattr(settings, f"{role}_provider") or settings.vision_provider
        model = getattr(settings, f"{role}_model")
        key = settings.api_key_for(provider)
        source = next(
            (v for v in KEY_ENV_VARS[provider] if os.environ.get(v)), None
        )
        where = f" (from {source})" if source else " (from .env)" if key else ""
        print(f"  {role:<7} model: {model or f'MISSING (set KG_{role.upper()}_MODEL)'}")
        print(f"          key: {'found' + where if key else 'MISSING'}")
        print(f"          endpoint: {settings.openai_base_url}")
        ok = ok and bool(key) and bool(model)

    print()
    if not ok:
        from kg_pipeline.llm.factory import missing_key_message, missing_model_message

        if not settings.api_key_for(settings.vision_provider):
            print(missing_key_message(settings.vision_provider))
        else:
            role = "vision" if not settings.vision_model else "text"
            print(missing_model_message(role))
        raise SystemExit(1)

    if args.ping:
        print("Sending one tiny request to check the key and endpoint work...")
        from kg_pipeline.llm.factory import get_chat_model

        try:
            reply = get_chat_model(settings.vision_provider, settings).invoke("Say OK.")
            text = (reply.content or "").strip().replace("\n", " ")[:60]
            print(f"  reply: {text!r}\n\nReady.")
        except Exception as exc:  # noqa: BLE001 -- the whole point is to report it
            raise SystemExit(
                f"  the call failed: {type(exc).__name__}: {exc}\n\n"
                "A key that is present but rejected usually means it does not "
                "belong to the endpoint above."
            ) from exc
    else:
        print("Keys are present. Add --ping to spend one tiny request "
              "confirming they work.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal Knowledge Graph Extraction Pipeline (LangChain/LangGraph)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract", help="Extract figures from PDFs")
    p_extract.add_argument("-i", "--input", required=True)
    p_extract.add_argument("-o", "--output", required=True)
    p_extract.add_argument("-d", "--dpi", type=int, default=200)
    p_extract.set_defaults(func=cmd_extract)

    p_classify = subparsers.add_parser("classify", help="Classify and filter extracted figures")
    p_classify.add_argument("-i", "--input", required=True)
    p_classify.add_argument("-o", "--output", required=True)
    p_classify.add_argument("--padding", action="store_true")
    p_classify.set_defaults(func=cmd_classify)

    p_generate = subparsers.add_parser("generate", help="Run KG generation over filtered images")
    p_generate.add_argument("-i", "--input", dest="input", required=True)
    p_generate.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        help="Output directory (default: outputs/run_<timestamp>_<id>, a fresh dir per run)",
    )
    p_generate.add_argument(
        "--provider",
        choices=["openai"],
        default=None,
        help="Vision-agent provider (default: KG_VISION_PROVIDER env or 'openai')",
    )
    p_generate.add_argument("--model", default=None, help="Vision-agent model override")
    p_generate.add_argument(
        "--text-provider",
        choices=["openai"],
        default=None,
        help="Text-agent (normalize/correct) provider; defaults to --provider",
    )
    p_generate.add_argument(
        "--text-model", default=None, help="Text-agent model override"
    )
    p_generate.set_defaults(func=cmd_generate)

    p_evaluate = subparsers.add_parser(
        "evaluate", help="Score a generation run against the golden dataset"
    )
    p_evaluate.add_argument(
        "-r",
        "--run",
        required=True,
        help="Run directory (uses its final_graphs/) or a directory of graph JSONs",
    )
    p_evaluate.add_argument(
        "-g",
        "--golden",
        default=os.path.join("data", "golden_dataset", "ground_truth"),
        help="Golden graph directory (default: data/golden_dataset/ground_truth)",
    )
    p_evaluate.add_argument(
        "--matcher",
        choices=["propagation", "anchor", "both"],
        default="propagation",
        help="Node-matching algorithm (default: propagation — it won the "
        "self-eval comparison); 'both' also writes matcher_disagreement.json",
    )
    p_evaluate.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum similarity for a node match (default: 0.35)",
    )
    p_evaluate.add_argument(
        "-o",
        "--output",
        default=None,
        help="Report directory (default: <run>/evaluation)",
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_diagram = subparsers.add_parser(
        "diagram", help="Render the pipeline topology (Mermaid; no API keys needed)"
    )
    p_diagram.add_argument("-o", "--output", default=None, help="Output file path")
    p_diagram.add_argument(
        "--png", action="store_true", help="Render PNG via mermaid.ink (needs internet)"
    )
    p_diagram.set_defaults(func=cmd_diagram)

    p_check = subparsers.add_parser(
        "check", help="Report whether API keys and models are configured"
    )
    p_check.add_argument("--ping", action="store_true",
                         help="Also send one tiny request to prove the key works")
    p_check.set_defaults(func=cmd_check)

    return parser


def main():
    load_dotenv()
    settings = get_settings()

    parser = build_parser()
    argv = sys.argv[1:] or ["--help"]
    args = parser.parse_args(argv)
    args.func(args, settings)


if __name__ == "__main__":
    main()
