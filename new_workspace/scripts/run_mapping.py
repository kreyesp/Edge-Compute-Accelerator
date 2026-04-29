"""
run_mapping.py
==============
Sweeps design points for the TinyYOLOv2 workload on the custom edge
accelerator using the accelforge Python API. No intermediate files
are written to disk. Results are ranked by energy and printed.

Usage:
    python scripts/run_mapping.py [options]

Examples:
    python scripts/run_mapping.py
    python scripts/run_mapping.py --sweep-glb 64 128 256 512 --sweep-fanout 8 16 32
"""

import argparse
import itertools
import sys
from pathlib import Path

import accelforge as af

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
WORKSPACE    = SCRIPT_DIR.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
# ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep.yaml")


# ---------------------------------------------------------------------------
# Default design-point parameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # GLB_KB     = 1024,
    GLB_KB     = 256,
    FANOUT_X   = 64,
    FANOUT_Y   = 64,
    FREQ_GHZ   = 1.0,
    BATCH_SIZE = 1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_param_grid(args: argparse.Namespace) -> list[dict]:
    glb_values    = args.sweep_glb    or [args.glb_kb]
    fanout_values = args.sweep_fanout or [args.fanout_x]
    freq_values   = args.sweep_freq   or [args.freq_ghz]

    grid = []
    for glb, fanout, freq in itertools.product(glb_values, fanout_values, freq_values):
        grid.append({
            **DEFAULTS,
            "GLB_KB":     glb,
            "FANOUT_X":   fanout,
            "FANOUT_Y":   fanout,
            "FREQ_GHZ":   freq,
            "MAC_TPT":    freq,
            "BATCH_SIZE": args.batch_size,
        })
    return grid


def extract_energy(mappings) -> float | None:
    """
    Pull total energy from the mappings object returned by
    spec.map_workload_to_arch(). Tries common accelforge attribute
    patterns — adjust if your version exposes a different field name.
    """
    # Try direct attribute access (most common in recent accelforge versions)
    for attr in ("energy", "total_energy", "best_energy"):
        if hasattr(mappings, attr):
            val = getattr(mappings, attr)
            if val is not None:
                return float(val)

    # Try treating mappings as iterable of mapping objects and take the best
    try:
        energies = [m.energy for m in mappings if hasattr(m, "energy")]
        if energies:
            return min(energies)
    except TypeError:
        pass

    # Try dict-like access
    try:
        return float(mappings["energy"])
    except (KeyError, TypeError):
        pass

    return None


def label(params: dict) -> str:
    return (f"GLB={params['GLB_KB']}KB  "
            f"PEs={params['FANOUT_X']}x{params['FANOUT_Y']}  "
            f"f={params['FREQ_GHZ']}GHz")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep TinyYOLOv2 design space via accelforge and report energy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--glb-kb",       type=int,   default=DEFAULTS["GLB_KB"])
    p.add_argument("--fanout-x",     type=int,   default=DEFAULTS["FANOUT_X"])
    p.add_argument("--fanout-y",     type=int,   default=DEFAULTS["FANOUT_Y"])
    p.add_argument("--freq-ghz",     type=float, default=DEFAULTS["FREQ_GHZ"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULTS["BATCH_SIZE"])
    p.add_argument("--sweep-glb",    nargs="+",  type=int,   metavar="KB")
    p.add_argument("--sweep-fanout", nargs="+",  type=int,   metavar="N")
    p.add_argument("--sweep-freq",   nargs="+",  type=float, metavar="GHZ")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser    = build_parser()
    args      = parser.parse_args()
    param_grid = build_param_grid(args)
    n          = len(param_grid)

    print(f"Sweeping {n} design point(s)...\n")
    results = []   # list of (energy, params)

    for i, params in enumerate(param_grid, 1):
        print(f"[{i}/{n}] {label(params)}", end="  ", flush=True)

        try:
            spec = af.Spec.from_yaml(
                ARCH_SRC,
                WORKLOAD_SRC,
                jinja_parse_data=params,
            )
            mappings = spec.map_workload_to_arch()
            energy   = extract_energy(mappings)

        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if energy is None:
            print("FAILED (could not parse energy from mappings)")
            print("  >> mappings object:", type(mappings), dir(mappings))
            continue

        print(f"energy = {energy:.4e} pJ")
        results.append((energy, params))

    if not results:
        print("\nNo successful runs to report.")
        sys.exit(1)

    results.sort(key=lambda x: x[0])

    print("\n" + "=" * 70)
    print(f"  SWEEP RESULTS  ({len(results)}/{n} succeeded, ranked by energy)")
    print("=" * 70)
    print(f"  {'Rank':<5}  {'Energy (pJ)':<16}  {'Design Point'}")
    print(f"  {'-'*5}  {'-'*16}  {'-'*40}")
    for rank, (energy, params) in enumerate(results, 1):
        marker = "  <-- BEST" if rank == 1 else ""
        print(f"  {rank:<5}  {energy:<16.4e}  {label(params)}{marker}")
    print("=" * 70)

    best_energy, best_params = results[0]
    print(f"\nBest energy : {best_energy:.4e} pJ")
    print(f"Best config : {label(best_params)}\n")


if __name__ == "__main__":
    main()
