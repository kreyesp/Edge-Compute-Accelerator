import accelforge as af
from pathlib import Path
import os

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
# ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "tpuv4_accelerator.yaml")

spec     = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                              jinja_parse_data={})
mappings = spec.map_workload_to_arch()
print(spec.arch)
# print(mappings.data.columns.tolist())
# print(mappings)


# print(dir(af))
# List available component models
# print("simba")
# with open("/Users/kreyesp/Desktop/6.5930/Final Project/Edge-Compute-Accelerator/.venv/lib/python3.13/site-packages/examples/arches/simba.yaml") as f:
#     print(f.read())

# print("eyeriss")
# with open("/Users/kreyesp/Desktop/6.5930/Final Project/Edge-Compute-Accelerator/.venv/lib/python3.13/site-packages/examples/arches/eyeriss.yaml") as f:
#     print(f.read())

# print("component annotated")
# with open("/Users/kreyesp/Desktop/6.5930/Final Project/Edge-Compute-Accelerator/.venv/lib/python3.13/site-packages/examples/misc/component_annotated.yaml") as f:
#     print(f.read())
