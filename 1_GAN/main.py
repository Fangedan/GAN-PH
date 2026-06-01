# Code for training of WGAN-gp
# Wasserstein GAN with gradient penalty

import os 
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim

import analysis
import load
import models
import training

def main():

    # ----- 00 Device check ----- #
    print("Device check start!!")
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda" if(torch.cuda.is_available()and ngpu>0) else "cpu")
    print(device, " will be used.")
    print(ngpu, " : Available gpu number")

    # ----- 01 Hyper Parameters ----- #
    b_size = 64  # batch size
    epochs = 50  # the number of epochs
    latent_size = 100  # the size of latent vector
    n_struc = 50  # the number of structure data
    in_header = "../synthetic_data"

    # ----- 02 Load Training Data ----- #
    load.torch_fix_seed(seed=500)
    label = load.get_label(n_struc=n_struc, Input_header=in_header)
    path_td = os.path.join(in_header, "x_train.pt")
    if os.path.isfile(path_td) == True:
        print("Load 3D micro structure")
        x_train = torch.load(path_td)
    else:
        x_train = load.load_structure(n_struc=n_struc,n_size=64,Input_header=in_header)
    dataset = torch.utils.data.TensorDataset(x_train, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True)

    
    # ----- 03 Construct Neural Network ----- #
    # Generator
    model_g = models.Generator().to(device)
    model_g.apply(models.weights_init)
    print(summary(model_g,[(latent_size,4,4,4),(3,4,4,4),(3,4,4,4)]))

    # Critic
    model_c = models.Critic().to(device)
    model_c.apply(models.weights_init)
    print(summary(model_c,(3,64,64,64)))

    # Estimator
    model_e = models.Estimator(in_ch=1, ndf=32).to(device)
    path = "../2_CNN/save_model/model_200epoch.pth"
    load_weight = torch.load(path)
    model_e.load_state_dict(load_weight)
    model_e = model_e.eval()

    # ----- 04 Training ----- #
    optimizer_c = optim.Adam(model_c.parameters(),
                        lr = 0.0002,
                        betas = (0.5,0.999)
                        )

    optimizer_g = optim.Adam(model_g.parameters(),
                            lr = 0.0002,
                            betas = (0.5,0.999)
                            )
    
    trainer = training.Trainer(model_g=model_g, optim_g=optimizer_g, model_c=model_c, optim_c=optimizer_c,
                               model_e=model_e, epochs=epochs, device=device,
                               dataloader=dataloader, in_header=in_header)
    trainer.train()

    # ----- 05 PLot Images ----- #
    plot_image = analysis.Images()
    plot_image.plot_save()

if __name__ == "__main__":
    main()
