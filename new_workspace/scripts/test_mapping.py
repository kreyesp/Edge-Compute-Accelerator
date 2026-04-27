import accelforge as af
from pathlib import Path

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep.yaml")

spec     = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                              jinja_parse_data={"GLB_KB"   :1024,
                                                "FREQ_GHZ" : 1.0,
                                                "FANOUT_X" : 64,
                                                "FANOUT_Y" : 64,})
mappings = spec.map_workload_to_arch()
print(mappings.data.columns.tolist())
print(mappings)
