# generate persistent diagram for a given structure data

import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import homcloud.pyvistahelper as pvhelper
import cv2
import homcloud.interface as hc
from tqdm.notebook import tqdm
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import trange

plt.rcParams['font.family'] ='Times New Roman'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10.5 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['mathtext.fontset'] = 'cm'


class LoadStructure:
    def __init__(self, header, n_size=64, n_struc=None):
        """ 
        :param header: str(path to the structure data)
        :param n_size: int(voxel size of the structure, default=64)
        :param n_struc: int(number of structures, if None, it will be calculated automatically)
        """

        self.header = header
        # print("Loading structures from {}".format(self.header))
        self.n_size = n_size
        if n_struc is None:
            self.n_struc = sum([1 for item in os.listdir(self.header) if os.path.isdir(os.path.join(self.header, item))])
        else:
            self.n_struc = n_struc
        print("Number of loading structures: {}".format(self.n_struc))
        self.th_values = {"Ni":255, "YSZ":127, "Pore":0}
        self.s_data = self._s_data(self.n_size)
    
    def _extract_phase(self,phase,strucs) -> np.array:
        """
        Extract the phase from structures
        :param phase: str
        :param strucs: np.array (n_struc[500], n_size[64], n_size[64], n_size[64])
        """ 
        _len = len(strucs)
        _strucs = []

        for i in range(_len):
            imgs = strucs[i]
            imgs_ex = np.array([imgs == self.th_values[phase]]).squeeze()
            _strucs.append(imgs_ex)

        return _strucs

    def _load_strucs(self,n_size=64) -> np.array:
        """
        Load the structures
        :param n_size: int(voxel size of the structure, default=64)
        """

        strucs = []
        for i in tqdm(range(self.n_struc)):
            imgs = np.zeros([n_size,n_size,n_size]).astype(np.float32)

            for j in range(n_size):
                img = cv2.imread(os.path.join(self.header, r"structure_{:04}\slice_{:04}.bmp".format(i+1,j)), 
                                cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (n_size,n_size), interpolation=cv2.INTER_NEAREST)
                imgs[j] = img

            strucs.append(imgs)

        return np.array(strucs)

    def _s_data(self,n_size=64) -> dict:
        """
        Get extracted data
        :param n_size: int(voxel size of the structure, default=64)
        """
        s_data = {}
        phases = ["Ni","YSZ","Pore"]
        strucs = self._load_strucs(n_size=n_size)

        for phase in phases:
            s_data[phase] = self._extract_phase(phase,strucs)

        return s_data

    def get_data(self, phase):
        return np.array(self.s_data[phase])
    

def plot_3D(data, num, phase, out_folder):
    """  
    Function to plot 3D structure
    :param data: np.array(structure data)
    :param num: int(number of structure)
    :param phase: str(phase to be analyzed, Ni, YSZ, Pore)
    :param out_folder: str(path to save the structure)
    """
    colors = {"Ni":"lime", "YSZ":"gold","Pore":"slategrey"}
    pl = pv.Plotter(shape=(1,1))
    pl.subplot(0,0)
    pl.add_mesh(
        pvhelper.Bitmap3D(data[num]).threshold(0.5),
        show_scalar_bar=False, opacity=1.0, color=colors[phase] 
    )
    # pl.show()
    pl.save_graphic(os.path.join(out_folder,"{}_structure.svg".format(phase)))
    pl.screenshot(os.path.join(out_folder,"{}_structure.png".format(phase)))
    pl.close()

    return pl

def p_Diagram(ex_data,phase,output_path):
    """
    Persistence Diagram
    :param ex_data: np.array(data to be analyzed)
    :param output_path: str(path to save the diagram)
    :param phase: str(phase to be analyzed, Ni, YSZ, Pore)
    """
    os.makedirs(os.path.join(output_path,"phase_{}".format(phase)), exist_ok=True)

    for i in tqdm(range(len(ex_data))):
        hc.PDList.from_bitmap_levelset(
            hc.distance_transform(ex_data[i],signed=True),
            save_to=os.path.join(output_path,"phase_{}/PD_{}.pdgm".format(phase,i+1))
        )
    
    return

def vis_PD(pd, out_folder, fig_name, x_range=[-18.5,18.5], figsize=(3.5, 2.8)):

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=400)
    pd.histogram(
        x_range = (x_range[0],x_range[1]),
        x_bins = int(x_range[1] - x_range[0])
    ).plot(
        colorbar={"type": "log"},
        ax = ax
    )

    ax.grid()
    ax.grid(linestyle=":")
    fig.tight_layout()
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(os.path.join(out_folder, "PD_{}.png".format(fig_name)))
    plt.close()

    return

def saving_PD(header, out_folder, n_struc = 300):
    
    os.makedirs(out_folder, exist_ok=True)
    
    # ------ Load the structures ------ #
    ls = LoadStructure(header=header, n_size=64, n_struc=n_struc)

    # ------ save persistence diagram ------ #
    phases = ["Ni", "YSZ", "Pore"]

    for phase in phases:
        # ------ save the 3D structure ------ #
        pl = plot_3D(ls.get_data(phase), 0, phase, out_folder)

        # ------ save the persistence diagram ------ #
        p_Diagram(ls.get_data(phase),phase,out_folder)

    return

def main():
    header = "../synthetic_data"  # path to the structure data
    out_folder = "../data/persistent_diagram"  # path to save the persistent diagram
    saving_PD(header, out_folder, n_struc = 50)

if __name__ == "__main__":
    main()