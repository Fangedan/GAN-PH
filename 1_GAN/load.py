# Wasserstein GAN with gradient penalty
# Load structure data

import numpy as np
import pandas as pd
import os
import random
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


def torch_fix_seed(seed):
    """ 
    :Function of fixing random seed for torch & numpy
    :param seed -> random seed [int]
    """
    # -------- Python random -------- #
    random.seed(seed)

    # -------- Numpy random -------- #
    np.random.seed(seed)

    # -------- Pytorch random -------- #
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    return 

def load_structure(n_struc,n_size,Input_header):
    """ 
    : Function of laoding 3D structuers

    :param n_struc : the number of structure data
    :param n_size :  the size of structure data
    :param Input_header : the path of structure data
    """
    x_train = torch.empty([n_struc,3,n_size,n_size,n_size],dtype=torch.float32)

    for i in tqdm(range(n_struc)):
        structure = np.zeros([n_size,n_size,n_size]).astype(np.float32)

        # step.1 Load data
        for j in range(n_size):
            path = os.path.join(Input_header, "structure_{:04}".format(i+1), "slice_{:04}.bmp".format(j))
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
            structure[j] = img
        
        # step.2 Convert One-Hot
        return_struc = np.zeros([3,n_size,n_size,n_size]).astype(np.float32)

        Ni = np.array([structure == 255]).squeeze()
        return_struc[0] = Ni.astype(np.float32)

        YSZ = np.array([structure == 127]).squeeze()
        return_struc[1] = YSZ.astype(np.float32)

        Pore = np.array([structure == 0]).squeeze()
        return_struc[2] = Pore.astype(np.float32)

        return_struc = torch.as_tensor(return_struc.astype(np.float32)).clone()
        x_train[i] = return_struc

    return x_train

def get_label(n_struc, Input_header):
    """ 
    Function of loading label data
    this function depends on the format of dat file which describes the structure data
    :param n_struc : the number of structure data
    :param Input_header : the path of structure data
    """

    # ---------- Load parameter of training data ---------- #
    path = os.path.join(Input_header,"results.dat")
    df = pd.read_table(path, sep="\s+")

    # ---------- Standardize the number of specific surface area ---------- #
    mm = preprocessing.MinMaxScaler()
    df_ssa = pd.DataFrame(
        mm.fit_transform(df[["SV0","SV1","SV2"]]), columns=["SV0","SV1","SV2"]
        )

    label = torch.zeros([n_struc,6,4,4,4], dtype=torch.float32)
    for i in range(n_struc):
        label[i,0,:,:,:] = df.loc[i,"VF0"] * 0.01    # VF -> Volume fraction
        label[i,1,:,:,:] = df.loc[i,"VF1"] * 0.01    # 0: phase Ni, 1: phase YSZ, 2: phase Pore
        label[i,2,:,:,:] = df.loc[i,"VF2"] * 0.01
        label[i,3,:,:,:] = df_ssa.loc[i,"SV0"]       # SV -> Specific surface area
        label[i,4,:,:,:] = df_ssa.loc[i,"SV1"]       # 0: phase Ni, 1: phase YSZ, 2: phase Pore 
        label[i,5,:,:,:] = df_ssa.loc[i,"SV2"]

    print("label.shape --> ", label.shape, "\n")
    print(label[:5,:,0,0,0],"\n etc ...")
    
    return label
    