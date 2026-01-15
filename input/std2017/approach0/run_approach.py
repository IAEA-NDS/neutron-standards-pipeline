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

    num_iters=30
    num_inner_iters=10

    num_inner_iters_str = '{:03d}'.format(num_inner_iters)

    # Directory setup
    dirs = [
        "001_reduction",
        "002_evaluation",
    ]

    for i in range(0, num_iters):
        red_idx = '{:03d}'.format(i * 2 + 1)
        eval_idx = '{:03d}'.format(i * 2 + 2)
        dirs.append(f'{red_idx}_reduction')
        dirs.append(f'{red_idx}_reduction_py')
        dirs.append(f'{eval_idx}_evaluation')

    dirs.append('bin')
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
                root / "001_reduction/GMDATA.CRD")

    shutil.copy(root / "../../../input/DAT.INP",
                root / "001_reduction/DAT.INP")

    # --------------------------------------------------------
    # Run DATP in 01_reduction
    # --------------------------------------------------------
    run(["../bin/datp"], cwd=root / "001_reduction")

    # --------------------------------------------------------
    # Move results into 02_evaluation
    # --------------------------------------------------------
    shutil.copy(root / "001_reduction/DAT.RES",
                root / "002_evaluation/data.gma")

    # --------------------------------------------------------
    # Modify MODE line (sed -i equivalent)
    # --------------------------------------------------------
    gma_path = root / "002_evaluation/data.gma"
    lines = gma_path.read_text().splitlines()

    new_lines = [
        # "MODE     3    0    0    3    1    0    0    0"
        f"MODE     3    0    0  {num_inner_iters_str}    1    0    0    0"
        if l.startswith("MODE") else l
        for l in lines
    ]
    gma_path.write_text("\n".join(new_lines) + "\n")

    # --------------------------------------------------------
    # Run GMAP in 02_evaluation
    # --------------------------------------------------------
    run(["../bin/gmap"], cwd=root / "002_evaluation")

    for i in range(1, num_iters):

        print(f'Perform reduction iteration {i}...')

        old_red_idx = '{:03d}'.format((i-1) * 2 + 1)
        old_eval_idx = '{:03d}'.format((i-1) * 2 + 2)

        new_red_idx = '{:03d}'.format(i * 2 + 1)
        new_eval_idx = '{:03d}'.format(i * 2 + 2)

        # --------------------------------------------------------
        # gawk transformation (manual redirection)
        # --------------------------------------------------------
        gawk_output = subprocess.check_output([
            "gawk", "-f", str(root / "../../../replace_values.awk"),
            str(root / f"{old_eval_idx}_evaluation/gma.res"),
            str(root / f"{old_red_idx}_reduction/DAT.INP")
        ])

        (root / f"{new_red_idx}_reduction/DAT.INP").write_bytes(gawk_output)

        shutil.copy(root / f"{new_red_idx}_reduction" / "DAT.INP",
                    root / f"{new_red_idx}_reduction_py" / "DAT.INP")

        # --------------------------------------------------------
        # Copy CRD file
        # --------------------------------------------------------
        shutil.copy(root / f"{old_red_idx}_reduction/GMDATA.CRD",
                    root / f"{new_red_idx}_reduction/GMDATA.CRD")

        shutil.copy(root / f"{old_red_idx}_reduction/GMDATA.CRD",
                    root / f"{new_red_idx}_reduction_py/GMDATA.CRD")

        # --------------------------------------------------------
        # Run DATP in 03_reduction
        # --------------------------------------------------------
        run(["../bin/datp"], cwd=root / f"{new_red_idx}_reduction")

        # reduce with Python datpy code
        run(["python", "-m", "datpy.datpy", "--legacy"], cwd= root / f"{new_red_idx}_reduction_py")
        run(["python", "-m", "datpy.datpy", "--legacy", "--output", f"../{new_eval_idx}_evaluation/data.json"], cwd= root / f"{new_red_idx}_reduction_py")

        # --------------------------------------------------------
        # Move results into 04_evaluation
        # --------------------------------------------------------
        shutil.copy(root / f"{new_red_idx}_reduction/DAT.RES",
                    root / f"{new_eval_idx}_evaluation/data.gma")

        # --------------------------------------------------------
        # Modify MODE line (sed -i equivalent)
        # --------------------------------------------------------
        gma_path = root / f"{new_eval_idx}_evaluation/data.gma"
        lines = gma_path.read_text().splitlines()

        new_lines = [
            # "MODE     3    0    0    3    1    0    0    0"
            f"MODE     3    0    0  {num_inner_iters_str}    1    0    0    0"
            if l.startswith("MODE") else l
            for l in lines
        ]
        gma_path.write_text("\n".join(new_lines) + "\n")

        # --------------------------------------------------------
        # Run GMAP in 04_evaluation
        # --------------------------------------------------------
        run(["../bin/gmap"], cwd=root / f"{new_eval_idx}_evaluation")

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

