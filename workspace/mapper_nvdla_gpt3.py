"""
Mapper: NVDLA architecture + GPT-3 6.7B KV-Cache workload.

Runs the accelforge FFM mapper over the dense workload and prints
per-einsum EDP results.
"""

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics

ARCH_FILE     = 'workspace/arches/nvdla.yaml'
WORKLOAD_FILE = 'workspace/workloads/gpt3_6.7B_kv_cache.yaml'

# Jinja2 template variables for the workload.
# Keep sizes small enough to run in reasonable time; adjust as needed.
JINJA_DATA = {
    'BATCH_SIZE':    1,
    'N_TOKENS':      128,
    'N_NEW_TOKENS':  1,
}

def min_edp_filter(data):
    """Return the index of the mapping with minimum energy-delay product."""
    best_idx = 0
    best_edp = float('inf')
    for i, result in enumerate(data):
        try:
            edp = float(result.energy()) * float(result.latency())
            if edp < best_edp:
                best_edp = edp
                best_idx = i
        except Exception:
            continue
    return best_idx


def main():
    print('Building spec: NVDLA arch + GPT-3 6.7B KV-Cache workload...')
    spec = af.Spec.from_yaml(
        ARCH_FILE,
        WORKLOAD_FILE,
        jinja_parse_data=JINJA_DATA,
    )

    # Optimise for both energy and latency (EDP).
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY

    print('Running mapper (this may take a while)...')
    all_mappings = spec.map_workload_to_arch()

    best = all_mappings[min_edp_filter(all_mappings.data)]
    total_energy  = float(best.energy())
    total_latency = float(best.latency())
    total_edp     = total_energy * total_latency

    print(f'\n{"Metric":<20} {"Value":>20}')
    print('-' * 42)
    print(f'{"Total Energy (J)":<20} {total_energy:>20.4e}')
    print(f'{"Total Latency (s)":<20} {total_latency:>20.4e}')
    print(f'{"EDP (J·s)":<20} {total_edp:>20.4e}')

    # Per-einsum breakdown
    print('\nPer-einsum energy breakdown:')
    print(f'  {"Einsum":<30} {"Energy (J)":>15}')
    print('  ' + '-' * 47)
    try:
        per_einsum = best.energy(per_einsum=True)
        for einsum_name, energy_val in per_einsum.items():
            print(f'  {einsum_name:<30} {float(energy_val):>15.4e}')
    except Exception as e:
        print(f'  (per-einsum breakdown unavailable: {e})')

    print('\n=== Best Mapping (YAML) ===')
    print(best.mapping().to_yaml())

    


if __name__ == '__main__':
    main()
