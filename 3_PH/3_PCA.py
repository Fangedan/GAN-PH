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

plt.rcParams['font.family'] ='Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.linewidth'] = 1.0
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
        return np.array(self.data_dict[label_vf])

    def getLabel_vf(self):
        return np.array(self.label_vf)

    def getLabel_RF(self):
        return np.array(self.label_RF)

    def getData(self):
        return np.array(self.data)


def calc_pca(ad, n_components=2, pca_reducer=None):
    """
    Calculate PCA on persistence image data.
    """
    if pca_reducer is None:
        X = ad.getData()
        X_norm = X / X.max()
        pca_reducer = PCA(n_components=n_components)
        pca_reducer.fit(X_norm)
        X_pca = pca_reducer.transform(X_norm)
        print("  Cumulative explained variance ratio: {:.03f}%".format(
            np.sum(pca_reducer.explained_variance_ratio_)*100))
        print("  Data shape: {} ---> {}".format(X_norm.shape, X_pca.shape))
    else:
        X = ad.getData()
        X_norm = X / X.max()
        X_pca = pca_reducer.transform(X_norm)
        print("  Data shape: {} ---> {}".format(X_norm.shape, X_pca.shape))

    y_vf = ad.getLabel_vf().reshape(-1,1)
    y_RF = ad.getLabel_RF().reshape(-1,1)
    _columns = ["PC {}".format(i+1) for i in range(n_components)]
    _columns.extend(["label_vf", "label_RF"])
    df_pca = pd.DataFrame(
        np.concatenate([X_pca, y_vf, y_RF], axis=1),
        columns=_columns
    ).astype({"label_vf":str, "label_RF":str})

    return df_pca, pca_reducer


def run_pca_for_phase(phase, dim="Dim_0", n_components=2):
    """
    Run the full PCA pipeline for one phase and save results.
    phase: "Ni", "YSZ", or "Pore"
    dim:   "Dim_0", "Dim_1", "Dim_2", or "all"
    """
    print(f"\n{'='*50}")
    print(f"Running PCA: phase={phase}, dim={dim}")
    print(f"{'='*50}")

    # ── Load persistence image pickle ──────────────────────────────────────
    pi_path = f"../data/persistent_images/phase_{phase}/P_images_{dim}.pkl"
    if not os.path.exists(pi_path):
        print(f"  ERROR: File not found: {pi_path}")
        print(f"  Make sure you ran 2_PI.py for the {phase} phase first.")
        return

    with open(pi_path, "rb") as f:
        pd_vects = pickle.load(f)

    print(f"  Loaded {len(pd_vects)} persistence images from {pi_path}")

    # ── Run PCA ────────────────────────────────────────────────────────────
    ad = AnodesData(pd_vects)
    df_pca, pca_reducer = calc_pca(ad, n_components=n_components)

    # ── Plot ───────────────────────────────────────────────────────────────
    os.makedirs("../data/pca_results", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(
        x="PC 1", y="PC 2",
        hue="label_vf", style="label_RF",
        data=df_pca, ax=ax
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA of Persistence Images ({phase} {dim})")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plot_path = f"../data/pca_results/PCA_{phase}_{dim}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved plot: {plot_path}")

    # ── Save PCA reducer ───────────────────────────────────────────────────
    reducer_path = f"../data/pca_results/pca_reducer_{phase}_{dim}.pkl"
    with open(reducer_path, "wb") as f:
        pickle.dump(pca_reducer, f)
    print(f"  Saved reducer: {reducer_path}")


def main():
    # Run PCA for all three phases
    # Change the phases list or dim if you want to run specific ones
    phases = ["Ni", "YSZ", "Pore"]
    dim    = "Dim_0"

    for phase in phases:
        run_pca_for_phase(phase, dim=dim)

    print(f"\n{'='*50}")
    print("All done! PCA plots saved to ../data/pca_results/")
    print(f"{'='*50}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stdout)
    else:
        print("Finished successfully!")
