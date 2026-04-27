"""
run_mapping.py
==============
Renders Jinja2-templated workload + architecture YAMLs and invokes
the Timeloop mapper for the TinyYOLOv2 workload on the custom
edge accelerator.

Usage:
    python scripts/run_mapping.py [options]

Examples:
    # Default design point
    python scripts/run_mapping.py

    # Custom GLB and array size
    python scripts/run_mapping.py --glb-kb 512 --fanout-x 16 --fanout-y 16

    # Sweep multiple GLB sizes
    python scripts/run_mapping.py --sweep-glb 64 128 256 512

    # Dry-run: just render the YAMLs, don't invoke Timeloop
    python scripts/run_mapping.py --dry-run
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import BaseLoader, Environment

# ---------------------------------------------------------------------------
# Paths (all relative to new_workspace/, which is two levels up from scripts/)
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
WORKSPACE    = SCRIPT_DIR.parent
WORKLOAD_SRC = WORKSPACE / "workloads" / "tiny_yolo_test.yaml"
ARCH_SRC     = WORKSPACE / "arches"    / "custom_accelerator_sweep.yaml"
OUTPUT_ROOT  = WORKSPACE / "outputs"

# ---------------------------------------------------------------------------
# Default design-point parameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    GLB_KB   = 256,       # on-chip SRAM size in KB
    FANOUT_X = 16,        # PE array width  (input reuse dimension)
    FANOUT_Y = 16,        # PE array height (output reuse dimension)
    FREQ_GHZ = 1.0,       # clock frequency in GHz (1 MAC/cycle)
    MAC_TPT  = 1.0,       # kept for back-compat with un-migrated arch YAMLs
    BATCH_SIZE = 1,       # inference batch size
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_yaml(src: Path, params: dict) -> str:
    """
    Load a YAML file as a Jinja2 template, render it with *params*, then
    return the rendered string.  Both {{ VAR }} substitution and
    {% set ... %} / {% if ... %} directives are supported.
    """
    template_str = src.read_text()
    env = Environment(loader=BaseLoader(), variable_start_string="{{",
                      variable_end_string="}}")
    rendered = env.from_string(template_str).render(**params)
    return rendered


def make_run_dir(params: dict) -> Path:
    """Return (and create) a unique output directory for this design point."""
    tag = (
        f"glb{params['GLB_KB']}kb"
        f"_x{params['FANOUT_X']}"
        f"_y{params['FANOUT_Y']}"
        f"_f{params['FREQ_GHZ']}ghz"
        f"_b{params['BATCH_SIZE']}"
    )
    run_dir = OUTPUT_ROOT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_rendered_yamls(run_dir: Path, params: dict) -> tuple[Path, Path]:
    """Render both templates and write them into run_dir."""
    workload_out = run_dir / "workload.yaml"
    arch_out     = run_dir / "arch.yaml"

    workload_out.write_text(render_yaml(WORKLOAD_SRC, params))
    arch_out.write_text(render_yaml(ARCH_SRC,     params))

    return workload_out, arch_out


def run_timeloop(run_dir: Path, workload_yaml: Path, arch_yaml: Path,
                 extra_args: list[str]) -> int:
    """
    Invoke `timeloop-mapper` (or `timeloop-model` if --model flag is set).
    Returns the process return-code.
    """
    cmd = [
        "timeloop-mapper",
        str(arch_yaml),
        str(workload_yaml),
        *extra_args,
    ]
    print(f"\n[run_mapping] Running: {' '.join(cmd)}")
    print(f"[run_mapping] CWD:     {run_dir}\n")

    result = subprocess.run(cmd, cwd=run_dir)
    return result.returncode


def parse_stats(run_dir: Path) -> dict | None:
    """
    Try to read the Timeloop stats YAML that the mapper writes on success.
    Returns a dict of key metrics, or None if the file isn't found.
    """
    stats_file = run_dir / "timeloop-mapper.stats.yaml"
    if not stats_file.exists():
        # Older Timeloop versions write .txt, not .yaml
        txt = run_dir / "timeloop-mapper.stats.txt"
        if txt.exists():
            print(f"[run_mapping] Stats written to: {txt}")
        return None

    with stats_file.open() as f:
        data = yaml.safe_load(f)

    # Pull the top-level summary fields (structure varies by Timeloop version)
    summary = {}
    for key in ("energy", "cycles", "utilization", "energy_per_mac"):
        if key in data:
            summary[key] = data[key]

    return summary or None


def print_summary(params: dict, stats: dict | None) -> None:
    print("\n" + "=" * 60)
    print("  DESIGN POINT SUMMARY")
    print("=" * 60)
    print(f"  GLB size    : {params['GLB_KB']} KB")
    print(f"  PE array    : {params['FANOUT_X']} x {params['FANOUT_Y']}"
          f"  ({params['FANOUT_X'] * params['FANOUT_Y']} PEs)")
    print(f"  Clock       : {params['FREQ_GHZ']} GHz")
    print(f"  Batch size  : {params['BATCH_SIZE']}")
    if stats:
        print()
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
    else:
        print("\n  (No parsed stats — check run directory for raw output)")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map TinyYOLOv2 workload onto the custom edge accelerator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single design-point knobs
    p.add_argument("--glb-kb",    type=int,   default=DEFAULTS["GLB_KB"],
                   metavar="KB",   help="Global buffer size in KB")
    p.add_argument("--fanout-x",  type=int,   default=DEFAULTS["FANOUT_X"],
                   metavar="N",    help="PE array fanout in X (input reuse)")
    p.add_argument("--fanout-y",  type=int,   default=DEFAULTS["FANOUT_Y"],
                   metavar="N",    help="PE array fanout in Y (output reuse)")
    p.add_argument("--freq-ghz",  type=float, default=DEFAULTS["FREQ_GHZ"],
                   metavar="GHZ",  help="Clock frequency in GHz")
    p.add_argument("--batch-size",type=int,   default=DEFAULTS["BATCH_SIZE"],
                   metavar="B",    help="Inference batch size")

    # Sweep mode: each flag accepts a list of values
    p.add_argument("--sweep-glb",    nargs="+", type=int,   metavar="KB",
                   help="Sweep over multiple GLB sizes (overrides --glb-kb)")
    p.add_argument("--sweep-fanout", nargs="+", type=int,   metavar="N",
                   help="Sweep over fanout values applied to BOTH X and Y")
    p.add_argument("--sweep-freq",   nargs="+", type=float, metavar="GHZ",
                   help="Sweep over clock frequencies")

    # Timeloop pass-through
    p.add_argument("--mapper-args", nargs=argparse.REMAINDER, default=[],
                   metavar="ARG",
                   help="Extra arguments forwarded verbatim to timeloop-mapper")

    # Utility flags
    p.add_argument("--dry-run",  action="store_true",
                   help="Render YAMLs but do not invoke Timeloop")
    p.add_argument("--clean",    action="store_true",
                   help="Delete existing output directories before running")

    return p


def build_param_grid(args: argparse.Namespace) -> list[dict]:
    """
    Build a list of parameter dicts to evaluate.
    Sweep args take priority over single-value args.
    """
    glb_values    = args.sweep_glb    or [args.glb_kb]
    fanout_values = args.sweep_fanout or [args.fanout_x]   # applied to both X/Y
    freq_values   = args.sweep_freq   or [args.freq_ghz]

    grid = []
    for glb, fanout, freq in itertools.product(glb_values, fanout_values, freq_values):
        grid.append({
            **DEFAULTS,
            "GLB_KB":    glb,
            "FANOUT_X":  fanout,
            "FANOUT_Y":  fanout,
            "FREQ_GHZ":  freq,
            "MAC_TPT":   freq,        # keep in sync
            "BATCH_SIZE": args.batch_size,
        })
    return grid


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Sanity-check that source files exist
    for path, label in [(WORKLOAD_SRC, "Workload"), (ARCH_SRC, "Architecture")]:
        if not path.exists():
            print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    param_grid = build_param_grid(args)
    print(f"[run_mapping] {len(param_grid)} design point(s) to evaluate.\n")

    all_stats = []

    for i, params in enumerate(param_grid, 1):
        print(f"--- Design point {i}/{len(param_grid)} ---")

        run_dir = make_run_dir(params)

        if args.clean and run_dir.exists():
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

        workload_yaml, arch_yaml = write_rendered_yamls(run_dir, params)
        print(f"[run_mapping] Rendered YAMLs written to: {run_dir}")

        if args.dry_run:
            print("[run_mapping] --dry-run set, skipping Timeloop invocation.")
            print_summary(params, stats=None)
            continue

        rc = run_timeloop(run_dir, workload_yaml, arch_yaml, args.mapper_args)

        if rc != 0:
            print(f"[WARNING] timeloop-mapper exited with code {rc} "
                  f"for design point {i}. Check {run_dir} for logs.")

        stats = parse_stats(run_dir)
        print_summary(params, stats)
        all_stats.append({"params": params, "stats": stats, "run_dir": str(run_dir)})

    # If we swept multiple points, write a consolidated summary CSV
    if len(param_grid) > 1:
        import csv
        summary_csv = OUTPUT_ROOT / "sweep_summary.csv"
        fieldnames  = list(DEFAULTS.keys()) + ["energy", "cycles",
                                                "utilization", "run_dir"]
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in all_stats:
                row = {**entry["params"], **(entry["stats"] or {}),
                       "run_dir": entry["run_dir"]}
                writer.writerow(row)
        print(f"[run_mapping] Sweep summary written to: {summary_csv}")


if __name__ == "__main__":
    main()