"""Batch driver: runs the compiled pipeline over a directory of images.

All per-image persistence (final graph, critiques, debug topology) is done by
the pipeline's own ``export`` node; the runner only orchestrates the loop and
collects the performance log.
"""

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

TARGET_CATEGORIES = ["neural_networks", "architecture_diagram"]

# Run outputs live under outputs/, not data/ -- data/ holds pipeline *inputs*
# (papers, extracted figures, model weights).
DEFAULT_RUNS_BASE = "outputs"

# The correction loop revisits 4 nodes up to MAX_CORRECTION_RETRIES times;
# LangGraph's default recursion limit of 25 leaves little headroom, so give
# batch runs a comfortable ceiling.
RECURSION_LIMIT = 50


def _ts() -> str:
    """Wall-clock HH:MM:SS stamp for progress lines (readable in a tailed log)."""
    return datetime.now().strftime("%H:%M:%S")


def new_run_dir(base_dir: str = DEFAULT_RUNS_BASE) -> str:
    """A fresh, uniquely named output directory for one experiment/batch run.

    Timestamp for human sorting, short uuid suffix so parallel runs started in
    the same second can't collide.
    """
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    return os.path.join(base_dir, run_id)


def collect_images(input_dir: str) -> list[str]:
    """Scans known category subfolders first, falling back to a flat directory scan."""
    tasks: list[str] = []
    found_categories = False

    for category in TARGET_CATEGORIES:
        cat_path = os.path.join(input_dir, category)
        if os.path.isdir(cat_path):
            found_categories = True
            tasks.extend(
                os.path.join(cat_path, f)
                for f in os.listdir(cat_path)
                if f.lower().endswith((".png", ".jpg"))
            )

    if not found_categories and os.path.isdir(input_dir):
        tasks.extend(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg"))
        )

    return sorted(tasks)


def _run_one(graph, image_path: str, output_dir: str) -> dict:
    """Runs the pipeline for one image and returns its perf-log entry."""
    filename = os.path.basename(image_path)
    item = {
        "filename": filename,
        "status": "failed",
        "nodes_found": 0,
        "edges_found": 0,
        "correction_retries": 0,
        "inference_time": 0.0,
    }
    t0 = time.time()
    try:
        result = graph.invoke(
            {"image_path": image_path, "output_dir": output_dir},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as e:  # noqa: BLE001 -- one bad image must not kill the batch
        item["inference_time"] = time.time() - t0
        item["error"] = str(e)
        return item

    item["inference_time"] = time.time() - t0
    item["status"] = result.get("status", "failed")
    item["correction_retries"] = result.get("retry_count", 0)
    if result.get("error"):
        item["error"] = result["error"]
    final_graph = result.get("graph")
    if final_graph is not None:
        item["nodes_found"] = len(final_graph.nodes)
        item["edges_found"] = len(final_graph.edges)
    return item


def run_batch_parallel(graph, input_dir: str, output_dir: str, workers: int) -> list[dict]:
    """`run_batch` over a thread pool.

    Safe because figures are independent: each invoke carries its own state,
    and every artifact is written per-figure by the export node. Threads (not
    processes) because the work is entirely network wait. How many workers an
    endpoint tolerates before rate-limiting depends on the endpoint.
    """
    images = collect_images(input_dir)
    final_dir = os.path.join(output_dir, "final_graphs")
    todo = [
        p for p in images
        if not os.path.exists(
            os.path.join(final_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
        )
    ]
    skipped = len(images) - len(todo)
    total = len(images)
    if skipped:
        print(f"{_ts()} SKIP {skipped} figures already done", flush=True)
    print(f"{_ts()} running {len(todo)} figures on {workers} workers", flush=True)

    perf_log: list[dict] = []
    lock = threading.Lock()
    done = skipped

    def work(image_path: str) -> dict:
        nonlocal done
        item = _run_one(graph, image_path, output_dir)
        with lock:
            done += 1
            perf_log.append(item)
            err = f" -- {item['error'][:70]}" if item.get("error") else ""
            print(
                f"[{done}/{total}] {_ts()} DONE ({item['status']}) in "
                f"{item['inference_time']:.0f}s: {item['filename']} "
                f"[{item['nodes_found']}n/{item['edges_found']}e, "
                f"retries={item['correction_retries']}]{err}",
                flush=True,
            )
        return item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    return perf_log


def run_batch(graph, input_dir: str, output_dir: str) -> list[dict]:
    """Invokes the pipeline once per image and returns a per-item perf log.

    Progress is logged per figure (with timestamps + elapsed) so a long-running
    batch can be tailed live to see exactly which figure is stalling. Figures
    whose final graph already exists in ``output_dir/final_graphs`` are skipped,
    so a batch that was killed part-way can be resumed by re-running it.
    """
    perf_log = []
    images = collect_images(input_dir)
    total = len(images)
    final_dir = os.path.join(output_dir, "final_graphs")

    for idx, image_path in enumerate(images, start=1):
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]
        item = {
            "filename": filename,
            "status": "failed",
            "nodes_found": 0,
            "edges_found": 0,
            "correction_retries": 0,
            "inference_time": 0.0,
        }

        # Resume support: don't redo figures already written on a prior run.
        if os.path.exists(os.path.join(final_dir, stem + ".json")):
            print(f"[{idx}/{total}] {_ts()} SKIP (already done): {filename}", flush=True)
            continue

        print(f"[{idx}/{total}] {_ts()} START: {filename}", flush=True)
        t0 = time.time()
        try:
            result = graph.invoke(
                {"image_path": image_path, "output_dir": output_dir},
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except Exception as e:  # noqa: BLE001 -- one bad image must not kill the batch
            item["inference_time"] = time.time() - t0
            item["error"] = str(e)
            perf_log.append(item)
            print(
                f"[{idx}/{total}] {_ts()} ERROR after {item['inference_time']:.0f}s: "
                f"{filename} -- {e}",
                flush=True,
            )
            continue
        item["inference_time"] = time.time() - t0

        item["status"] = result.get("status", "failed")
        item["correction_retries"] = result.get("retry_count", 0)
        if result.get("error"):
            item["error"] = result["error"]

        final_graph = result.get("graph")
        if final_graph is not None:
            item["nodes_found"] = len(final_graph.nodes)
            item["edges_found"] = len(final_graph.edges)

        perf_log.append(item)
        print(
            f"[{idx}/{total}] {_ts()} DONE ({item['status']}) in "
            f"{item['inference_time']:.0f}s: {filename} "
            f"[{item['nodes_found']}n/{item['edges_found']}e, "
            f"retries={item['correction_retries']}]",
            flush=True,
        )

    return perf_log
