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
import seaborn as sns

plt.rcParams['font.family'] ='Times New Roman'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10.5 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

class AnodesData:
    """
    Class to handle the data of persistence images
    """

    def __init__(self,pd_vects):
        self.pd_vects = pd_vects
        self.label_vf = self.pd_vects.loc[:,"label_vf"]
        self.label_RF = self.pd_vects.loc[:,"label_RF"]
        self.data = self.pd_vects.iloc[:,:-2]
        self.data_dict = {}

        for i in range(self.pd_vects.shape[0]):
            if self.pd_vects.loc[i,"label_vf"] not in self.data_dict:
                self.data_dict[self.pd_vects.loc[i,"label_vf"]] = [self.pd_vects.iloc[i,:-2]]
            else:
                self.data_dict[self.pd_vects.loc[i,"label_vf"]].append(self.pd_vects.iloc[i,:-2])

    def getByLabel(self, label_vf):
        """ 
        Function to get the data by label
        :param label_vf: int -> label for volume fraction
        """
        return np.array(self.data_dict[label_vf])
    
    def getLabel_vf(self):
        """ 
        Function to get the label for volume fraction
        :return: np.array -> label for volume fraction
        """
        return np.array(self.label_vf)
    
    def getLabel_RF(self):
        """ 
        Function to get the label for real or fake
        :return: np.array -> label for real or fake
        """
        return np.array(self.label_RF)
    
    def getData(self):
        """ 
        Function to get the data
        """
        return np.array(self.data)
    

def calc_pca(ad, n_components=2, pca_reducer=None):
    """
    Function to calculate principal component analysis
      -> applying normalization to the data and then PCA via sklearn.decomposition.PCA
    :param ad: AnodesData object
    :param pca_reducer: PCA object for fake data
    """

    ## ------ PCA for real data ------ ##
    if  pca_reducer is None:

        # Normalize data by dividing by max value (method to homcloud tutorial)
        X = ad.getData()
        X_norm = X / X.max()   # normalize data to be analyzed

        # ------ PCA ------ #
        pca_reducer = PCA(n_components=n_components)   # PCA object
        pca_reducer.fit(X_norm)
        X_pca = pca_reducer.transform(X_norm)   # reduced data
        print("Cumulative explained variance ratio: {:.03f}%".format(np.sum(pca_reducer.explained_variance_ratio_)*100))  # check the cumulative explained variance ratio
        print("Change of data shape: {} ---> {}".format(X_norm.shape, X_pca.shape))

    ## ------ PCA for fake data ------ ##
    else:
        # Normalize data by dividing by max value
        X = ad.getData()
        X_norm = X / X.max()   # normalize data to be analyzed

        # ------ PCA ------ #
        X_pca = pca_reducer.transform(X_norm)   # reduced data
        print("Change of data shape: {} ---> {}".format(X_norm.shape, X_pca.shape))
        
    # ------ output resulted dataframe ------ #
    y_vf = ad.getLabel_vf().reshape(-1,1)
    y_RF = ad.getLabel_RF().reshape(-1,1)
    _columns = ["PC {}".format(i+1) for i in range(n_components)]
    _columns.extend(["label_vf", "label_RF"])
    df_pca = pd.DataFrame(np.concatenate([X_pca, y_vf, y_RF], axis=1),
                            columns=_columns).astype({"label_vf":str, "label_RF":str}) 

    return df_pca, pca_reducer

def cat_dfs(header, phase, dim):
    """ 
    Function to load saved persistence images as a pickle file and concatenate them.
    header: str, path to the folder where the files are saved.
    phase: str, "Ni", "YSZ", or "Pore".
    dim: str, Dim 0~2, and all.
    """
    vfs = [item for item in os.listdir(header) if os.path.isdir(os.path.join(header, item))]

    dfs = []
    for vf in vfs:
        _path = os.path.join(header, vf, "phase_{}/P_images_{}.pkl".format(phase, phase))
        with open (_path, "rb") as f:
            pis = pickle.load(f)
        # print("Keys of the dictionary: ", pis.keys())
        df = pis[dim]
        dfs.append(df)
    
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def main():
    ## ------ Load the data ------ ##
    phase = "Ni"  # "Ni", "YSZ", or "Pore"
    dim = "Dim_0"  # "Dim_0", "Dim_1", "Dim_2", or "all"

    # Load directly from our persistence images folder
    pi_path = "../data/persistent_images/phase_{}/P_images_{}.pkl".format(phase, dim)
    with open(pi_path, "rb") as f:
        pd_vects = pickle.load(f)

    ad = AnodesData(pd_vects)

    # Calculate PCA
    n_components = 2
    df_pca, pca_reducer = calc_pca(ad, n_components=n_components)

    # ------ Plot the data ------ #
    os.makedirs("../data/pca_results", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(x="PC 1", y="PC 2", hue="label_vf", style="label_RF", data=df_pca, ax=ax)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA of Persistence Images ({} {})".format(phase, dim))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("../data/pca_results/PCA_{}_{}.png".format(phase, dim))
    plt.close()
    print("Saved PCA plot!")

    # Save the PCA object
    with open("../data/pca_results/pca_reducer_{}_{}.pkl".format(phase, dim), "wb") as f:
        pickle.dump(pca_reducer, f)
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stdout)
    else:
        print("Finished successfully!")