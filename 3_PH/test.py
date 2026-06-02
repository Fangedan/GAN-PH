import homcloud.interface as hc
import os

header = "../data/persistent_diagram/phase_Ni"

for dim in [0, 1, 2]:
    print(f"Testing dim {dim}...")
    try:
        pd = hc.PDList(os.path.join(header, "PD_1.pdgm")).dth_diagram(dim)
        print(f"  dim {dim} OK: {pd}")
    except Exception as e:
        print(f"  dim {dim} FAILED: {e}")