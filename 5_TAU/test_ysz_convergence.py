"""
Task 1 decision gate: test whether highest tau_YSZ values are solver artifacts
or genuine near-percolation values.

Re-solves the 5 worst structures with stricter convergence settings and
compares. If values drop significantly, labels need recomputing.
"""
import sys, math, warnings
sys.path.insert(0, 'c:/Users/alin2/GAN-PH/4_CNNCT')
from analyze import load_structure, compute_phase_connectivity, YSZ_VAL
from pathlib import Path
import numpy as np
import taufactor as tau

worst = [
    'structure_0027',  # tau_YSZ=214.33
    'structure_0084',  # tau_YSZ=213.90
    'structure_0072',  # tau_YSZ=178.98
    'structure_0070',  # tau_YSZ=154.77
    'structure_0046',  # tau_YSZ=125.31
]
data = Path('c:/Users/alin2/GAN-PH/real_data')

print('Task 1 — convergence sensitivity test on worst tau_YSZ structures')
print(f'Default: iter_limit=10000  conv_crit=0.01')
print(f'Strict:  iter_limit=100000 conv_crit=0.001')
print()

for name in worst:
    sd = data / name
    vol = load_structure(sd)
    binary = (vol == YSZ_VAL).astype(np.float32)
    conn = compute_phase_connectivity(vol, YSZ_VAL)

    # Default settings
    try:
        s1 = tau.Solver(binary)
        s1.solve(iter_limit=10000, conv_crit=0.01, verbose=False)
        v_default = float(np.asarray(s1.tau).flat[0])
    except Exception as e:
        v_default = float('nan')

    # Strict settings
    try:
        s2 = tau.Solver(binary)
        s2.solve(iter_limit=100000, conv_crit=0.001, verbose=False)
        v_strict = float(np.asarray(s2.tau).flat[0])
    except Exception as e:
        v_strict = float('nan')

    if not math.isnan(v_default) and v_default > 0:
        pct = abs(v_strict - v_default) / v_default * 100
        verdict = 'GENUINE' if v_strict > 15 else 'SOLVER ARTIFACT (drops to single digits)'
    else:
        pct = float('nan')
        verdict = 'NaN'

    print(f'{name}: conn_YSZ={conn:.3f}  default={v_default:.2f}  strict={v_strict:.2f}  '
          f'delta={pct:.1f}%  → {verdict}')
    sys.stdout.flush()
