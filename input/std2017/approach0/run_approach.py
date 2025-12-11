#!/usr/bin/env python3
"""
pipeline.py

A Python replacement for the original Bash pipeline.
Can be run from the command line or imported as a module.

Usage (CLI):
    python pipeline.py --root /path/to/root

Usage (Python module):
    from pipeline import run_pipeline
    run_pipeline("/path/to/root")
"""

import subprocess
import shutil
from pathlib import Path
import argparse

# ------------------------------------------------------------
# Helper to run commands
# ------------------------------------------------------------
def run(cmd, cwd=None):
    """Run a command with error checking."""
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


# ------------------------------------------------------------
# Main pipeline function
# ------------------------------------------------------------
def run_fortran_pipeline(root):
    """
    Execute the full Fortran/GMAP/DATP pipeline.

    Parameters
    ----------
    root : str or Path
        Path to the directory where the original bash script expects to run.
        All paths inside the pipeline are resolved relative to this root.
    """
    root = Path(root).resolve()
    print(f"\n=== Running pipeline in root: {root} ===\n")

    # Directory setup
    dirs = [
        "01_reduction",
        "02_evaluation",
        "03_reduction",
        "03_reduction_py",
        "04_evaluation",
        "bin"
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    bin_dir = root / "bin"

    # --------------------------------------------------------
    # Compile Fortran sources
    # --------------------------------------------------------
    run([
        "gfortran", "-std=legacy",
        str(root / "../../../deps/gmap-fortran/source/GMAP.FOR"),
        "-o", str(bin_dir / "gmap")
    ])

    run([
        "gfortran", "-std=legacy",
        str(root / "../../../deps/datp-fortran/DATP.FOR"),
        "-o", str(bin_dir / "datp")
    ])

    # --------------------------------------------------------
    # Copy initial input files into 01_reduction
    # --------------------------------------------------------
    shutil.copy(root / "../../input_std2017.crd",
                root / "01_reduction/GMDATA.CRD")

    shutil.copy(root / "../../../input/DAT.INP",
                root / "01_reduction/DAT.INP")

    # --------------------------------------------------------
    # Run DATP in 01_reduction
    # --------------------------------------------------------
    run(["../bin/datp"], cwd=root / "01_reduction")

    # --------------------------------------------------------
    # Move results into 02_evaluation
    # --------------------------------------------------------
    shutil.copy(root / "01_reduction/DAT.RES",
                root / "02_evaluation/data.gma")

    # --------------------------------------------------------
    # Modify MODE line (sed -i equivalent)
    # --------------------------------------------------------
    gma_path = root / "02_evaluation/data.gma"
    lines = gma_path.read_text().splitlines()

    new_lines = [
        "MODE     3    0    0    3    1    0    0    0"
        if l.startswith("MODE") else l
        for l in lines
    ]
    gma_path.write_text("\n".join(new_lines) + "\n")

    # --------------------------------------------------------
    # Run GMAP in 02_evaluation
    # --------------------------------------------------------
    run(["../bin/gmap"], cwd=root / "02_evaluation")

    # --------------------------------------------------------
    # gawk transformation (manual redirection)
    # --------------------------------------------------------
    gawk_output = subprocess.check_output([
        "gawk", "-f", str(root / "../../../replace_values.awk"),
        str(root / "02_evaluation/gma.res"),
        str(root / "01_reduction/DAT.INP")
    ])

    (root / "03_reduction/DAT.INP").write_bytes(gawk_output)

    # --------------------------------------------------------
    # Copy CRD file
    # --------------------------------------------------------
    shutil.copy(root / "01_reduction/GMDATA.CRD",
                root / "03_reduction/GMDATA.CRD")

    shutil.copy(root / "01_reduction/GMDATA.CRD",
                root / "03_reduction_py/GMDATA.CRD")

    # --------------------------------------------------------
    # Run DATP in 03_reduction
    # --------------------------------------------------------
    run(["../bin/datp"], cwd=root / "03_reduction")

    # reduce with Python datpy code
    run(["python", "-m", "datpy.datpy", "--legacy"], cwd= root / "03_reduction_py")
    run(["python", "-m", "datpy.datpy", "--legacy", "--output", "../04_evaluation/data.json"], cwd= root / "03_reduction_py")

    # --------------------------------------------------------
    # Move results into 04_evaluation
    # --------------------------------------------------------
    shutil.copy(root / "03_reduction/DAT.RES",
                root / "04_evaluation/data.gma")

    # --------------------------------------------------------
    # Modify MODE line (sed -i equivalent)
    # --------------------------------------------------------
    gma_path = root / "04_evaluation/data.gma"
    lines = gma_path.read_text().splitlines()

    new_lines = [
        "MODE     3    0    0    3    1    0    0    0"
        if l.startswith("MODE") else l
        for l in lines
    ]
    gma_path.write_text("\n".join(new_lines) + "\n")

    # --------------------------------------------------------
    # Run GMAP in 04_evaluation
    # --------------------------------------------------------
    run(["../bin/gmap"], cwd=root / "04_evaluation")

    print("\n=== Pipeline completed successfully ===\n")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the GMAP/DATP processing pipeline."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory where the original script would run (default: current directory)."
    )
    args = parser.parse_args()
    run_fortran_pipeline(args.root)


if __name__ == "__main__":
    main()

