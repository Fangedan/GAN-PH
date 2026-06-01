# Surrogate model for estimating the specific surface area

# -------- import library -------- #
import load
import model
import trainer
import analysis
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchsummary import summary
import pandas as pd

# --------- hyperparameters --------- #
n_struc = 50  # number of structures
n_size = 64     # voxel size of the structure
seed = 42       # random seed
b_size = 128    # batch size
epochs = 200    # number of epochs

Input_header = Path(
    "../synthetic_data"
    )  # path to the training data
# ----------------------------------- #

# -------- main function -------- #
def main():
    # -------- fix random seed -------- #
    load.torch_fix_seed(seed)

    # -------- load the training data -------- #
    # structure data
    x_train = load.load_structure(
        n_struc,n_size,Input_header
        )
    # label data
    y_train = load.get_ssalabel(
        n_struc,Input_header
        )

    print("structure data shape: ", x_train.shape)
    print("label data shape: ", y_train.shape)

    structure_ex, label_ex = load.Extract_structure(x_train, y_train)
    print("structure_ex.shape --> ", structure_ex.shape)
    print("label_ex.shape --> ", label_ex.shape)

    # split the data into training and validation
    x_train, x_val, y_train, y_val = train_test_split(
        structure_ex, label_ex, test_size=0.2, random_state=seed
        )
    print("x_train.shape --> ", x_train.shape)
    print("y_train.shape --> ", y_train.shape)
    print("x_val.shape --> ", x_val.shape)
    print("y_val.shape --> ", y_val.shape)

    # create the dataset
    ds_train = torch.utils.data.TensorDataset(
        x_train, y_train
    )
    ds_val = torch.utils.data.TensorDataset(
        x_val, y_val
    )
    # create the dataloader
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=b_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=b_size, shuffle=False
    )
    # ----------------------------------- #

    # -------- create the model -------- #
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda" if(torch.cuda.is_available()and ngpu>0) else "cpu")
    print(device, " will be used.")
    print(ngpu, " : Available gpu number")

    net = model.Net(
        in_ch=1,ndf=32
    )
    net = net.apply(model.weights_init).to(device)
    print(summary(net, (1, 64, 64, 64)))
    # ----------------------------------- #

    # -------- Train -------- #
    criterion = nn.MSELoss()

    # ------- Optimizer ------- #
    optimizer = optim.Adam(net.parameters(),
                            lr = 0.001,
                            betas = (0.5,0.999)
                            )
    print(optimizer)

    _trainer = trainer.Trainer(
        model=net, 
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        dl_train=train_loader,
        dl_valid=val_loader
    )

    _trainer.train()
    # ----------------------------------- #

    # -------- plot loss -------- #
    data = pd.read_table("log.dat", sep="\s+", header=None)
    analysis.plot_moving_ave(data)

if __name__ == "__main__":
    main()
