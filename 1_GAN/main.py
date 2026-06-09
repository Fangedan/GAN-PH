# Main script for WGAN-GP training
# Generates 3D microstructures of SOC electrodes
#
# MODIFICATION: b_size changed from 64 to 4 for CPU training.

import load
import models
import training
import os
from pathlib import Path
import torch
import torch.optim as optim

# --------- Hyperparameters --------- #
n_struc = 50      # number of training structures
n_size  = 64      # voxel size
seed    = 42      # random seed
b_size  = 4       # batch size  <- was 64; reduced for CPU stability
epochs  = 50      # number of training epochs
lr      = 0.0001  # learning rate

Input_header = Path("../synthetic_data")

# --------- Main --------- #
def main():
    load.torch_fix_seed(seed)

    # ---- Load training data ---- #
    x_train = load.load_structure(n_struc, n_size, Input_header)
    y_train = load.get_label(n_struc, Input_header)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=b_size, shuffle=True
    )

    # ---- Device ---- #
    ngpu   = torch.cuda.device_count()
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, "will be used.")

    # ---- Load pretrained SSA estimator (frozen) ---- #
    estimator = models.Estimator(in_ch=1, ndf=32).to(device)
    estimator.load_state_dict(
        torch.load("../2_CNN/save_model/model_200epoch.pth", map_location=device)
    )
    estimator.eval()

    # ---- Build Generator and Critic ---- #
    generator = models.Generator(latent_size=100).to(device)
    generator.apply(models.weights_init)

    critic = models.Critic().to(device)
    critic.apply(models.weights_init)

    # ---- Optimizers ---- #
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_c = optim.Adam(critic.parameters(),    lr=lr, betas=(0.5, 0.9))

    # ---- Train ---- #
    trainer = training.Trainer(
        model_g    = generator,
        optim_g    = opt_g,
        model_c    = critic,
        optim_c    = opt_c,
        model_e    = estimator,
        epochs     = epochs,
        device     = device,
        dataloader = train_loader,
        in_header  = str(Input_header)
    )
    trainer.train()


if __name__ == "__main__":
    main()
