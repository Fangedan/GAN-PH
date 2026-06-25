# load training data
# Surrogate model for estimating the specific surface area

# -------- import library -------- #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
import random

# -------- Functions -------- #
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

    :param n_struc : the number of structures to load
    :param n_size :  the size of the structure
    :param Input_header : Path to training data
    """
    x_train = torch.empty([n_struc,3,n_size,n_size,n_size],dtype=torch.float32)

    for i in tqdm(range(n_struc)):
        structure = np.zeros([n_size,n_size,n_size]).astype(np.float32)

        # step.1 Load data
        for j in range(n_size):
            path = os.path.join(Input_header, "structure_{:04}".format(i+1), "slice_{:04}.bmp".format(j))
            # path = Input_header.joinpath("structure_{:04}\out_{:04}.bmp".format(i+1,j))
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
            # img = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) 
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

def get_ssalabel(n_struc, Input_header):
    """ 
    Function to load the ssa label of the training data

    :param n_struc : Number of structures to load
    :param Input_header : Path to training
    """

    # ---------- Load parameter of training data ---------- #
    path = os.path.join(Input_header,"results.dat")
    df = pd.read_table(path, sep="\s+")

    ssa_label = torch.zeros([n_struc,3])
    for i in range(n_struc):
        ssa_label[i,0] = df.loc[i,"SV0"]    # sv -> specific surface area [um^2/um^3]
        ssa_label[i,1] = df.loc[i,"SV1"]    
        ssa_label[i,2] = df.loc[i,"SV2"]
    
    return ssa_label

def Phase_pickup(images, phase):
    if phase == 'Ni':
        imgs = images[:,0]
    elif phase == 'YSZ':
        imgs = images[:,1]
    elif phase == 'Pore':
        imgs = images[:,2]
    imgs = imgs.reshape(imgs.shape[0],1,imgs.shape[1],imgs.shape[2],imgs.shape[3])
    return imgs

def Extract_structure(structure,label):
    """  
    Function to extract the structure of each phase and the label data of each phase
    
    :param structure : 3D structure data
    :param label : ssa label data
    """

    # Extract the structure of each phase
    _Ni = Phase_pickup(structure,'Ni')
    _YSZ = Phase_pickup(structure,'YSZ')
    _Pore = Phase_pickup(structure,'Pore')

    structure_ex = torch.cat((_Ni,_YSZ,_Pore), dim=0)

    # load the label data of each phase
    _ni = label[:,0]
    _ysz = label[:,1]
    _pore = label[:,2]
      
    label_ex = torch.cat((_ni, _ysz, _pore),dim=0)

    return structure_ex, label_ex