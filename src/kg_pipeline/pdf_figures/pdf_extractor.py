import os
import shutil
import subprocess
import time


def check_java():
    """Verifies Java is installed."""
    if not shutil.which("java"):
        raise EnvironmentError("Java is not found in PATH. Please install Java.")


def extract_figures(input_dir: str, output_dir: str, jar_path: str, dpi: int = 200):
    """Runs pdffigures2 to extract figure images + metadata from a folder of PDFs.

    Saves images to ``<output_dir>/images/`` and JSONs to ``<output_dir>/data/``.
    Returns execution stats for benchmarking.
    """
    t_start = time.time()

    abs_input = os.path.abspath(input_dir)
    abs_jar = os.path.abspath(jar_path)
    abs_output = os.path.abspath(output_dir)

    data_dir = os.path.join(abs_output, "data")
    img_dir = os.path.join(abs_output, "images")

    check_java()
    if not os.path.exists(abs_jar):
        raise FileNotFoundError(f"JAR not found: {abs_jar}")
    if not os.path.exists(abs_input):
        raise FileNotFoundError(f"Input dir not found: {abs_input}")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    command = [
        "java",
        "-jar",
        abs_jar,
        abs_input,
        "-d",
        os.path.join(data_dir, ""),
        "-m",
        os.path.join(img_dir, ""),
        "-i",
        str(dpi),
    ]

    subprocess.run(command, check=True)
    t_end = time.time()

    return {
        "status": "success",
        "duration": t_end - t_start,
        "input_dir": abs_input,
    }
