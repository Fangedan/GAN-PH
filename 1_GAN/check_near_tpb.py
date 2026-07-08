"""Measure near_tpb proxy on softmax outputs to calibrate TPB loss target."""
import sys, torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import models, load

device = torch.device('cpu')
gen = models.Generator(latent_size=100).to(device)
gen.load_state_dict(torch.load('save_model/Generator_050epoch.pth', map_location=device))
gen.eval()

data_dir = Path('../real_data')
n_struc = sum(1 for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('structure_'))
x_train = load.load_structure(n_struc, 64, data_dir)
y_train = load.get_label(n_struc, data_dir)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train), batch_size=8, shuffle=False)

vals = []
with torch.no_grad():
    for i, (_, labels) in enumerate(loader):
        if i >= 8: break
        vf  = labels[:, :3, :, :, :]
        ssa = labels[:, 3:, :, :, :]
        noise = torch.randn(labels.size(0), 100, 4, 4, 4)
        g = gen(noise, vf, ssa)
        near_tpb = (g[:,0] * g[:,1] * g[:,2]).mean().item()
        vals.append(near_tpb)
        print(f'batch {i}: near_tpb={near_tpb:.6f}  (g min={g.min():.3f} max={g.max():.3f})')

print(f'\nmean near_tpb across batches: {np.mean(vals):.6f}')
print(f'init value (all probs=1/3):   {(1/3)**3:.6f}')
print(f'suggested target_tpb:         {np.mean(vals) * 1.5:.6f}  (50% above current)')
