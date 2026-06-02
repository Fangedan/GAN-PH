# convert persistence diagram to persistence image

import cv2
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import homcloud.interface as hc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm

plt.rcParams['font.family'] ='Times New Roman'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10.5 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

def load_PD(header, dim, n_sample=None):
    """ 
    Function to load the persistence diagrams
    :param header: str -> path to the persistence diagrams
    :param dim: int -> dimension of the persistence diagrams
    :param n_sample: int -> number of samples
    """
    pd_list = []
    if n_sample is None:
        n_sample = len([item for item in os.listdir(header) if os.path.isfile(os.path.join(header, item)) and item.endswith('.pdgm')])
    else:
        n_sample = n_sample
    for i in range(n_sample):
        _path_pd = os.path.join(header, "PD_{}.pdgm".format(i+1))
        pd_list.append(hc.PDList(_path_pd).dth_diagram(dim))

    return pd_list

def get_pis(header, out_header, label_RF, label_vf, max_dim=3, x_range=(-18.5,18.5)):

    """ 
    Functions to get persistence images from the persistence diagrams
    :param header: str  -> path to the persistence diagrams
    :param out_folder: str -> path to save the persistence images
    :param label_RF: int -> label for real or fake
    :param label_vf: int -> label for volume fraction
    :param max_dim: int -> maximum dimension of the persistence diagrams
    :param x_range: tuple -> range of the persistence diagrams
    """
    os.makedirs(out_header, exist_ok=True)
    # step.1 Load persistence diagram
    pds = {}
    for i in range(max_dim):
        pds["Dim_{}".format(i)] = load_PD(header, dim=i)

    # step.2 Vectorize
    vector_spec = hc.PIVectorizeSpec(
        (x_range[0],x_range[1]), 
        int(x_range[1]-x_range[0]), 
        sigma=0.002, 
        weight=("atan", 0.01, 3)
    )

    pis = {}
    for key in pds.keys():
        pis[key] = np.vstack([
            vector_spec.vectorize(_pd) for _pd in pds[key]
        ])
    pis["all"] = np.concatenate([pis[key] for key in pis.keys()],axis=1)

    # step.3 Create DataFrame
    pi_dfs = {}
    for key in pis.keys():
        df = pd.DataFrame(pis[key])
        df["label_RF"] = label_RF
        df["label_vf"] = label_vf
        pi_dfs[key] = df

    return pi_dfs

def main():
    header_pd = "../data/persistent_diagram/phase_Ni"
    header_pi = "../data/persistent_images/phase_Ni"

    df_pi = get_pis(
        header_pd,
        header_pi,
        label_RF="real",
        label_vf="Ni",
        max_dim=3,
        x_range=(-18.5,18.5)
    )

    for key in df_pi.keys():
        df_pi[key].to_csv(os.path.join(header_pi, "P_images_{}.csv".format(key)), index=False)

        with open(os.path.join(header_pi, "P_images_{}.pkl".format(key)), "wb") as f:
            pickle.dump(df_pi[key], f)

if __name__ == "__main__":
    main()