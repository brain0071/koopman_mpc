import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import os

def plot(file_path_sindy, file_path_edmd):
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42  
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    
    
    data_sindy = pd.read_csv(file_path_sindy)
    data_edmd = pd.read_csv(file_path_edmd)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plot_settings = [
        ("t1_to_v1", "darkorange", r"$T_{1}$ and $V_{1}$"),
        ("t1_to_v2", "darkgreen", r"$T_{1}$ and $V_{2}$"),
        ("t2_to_v2", "steelblue", r"$T_{2}$ and $V_{2}$"),
        ("t2_to_v1", "purple", r"$T_{2}$ and $V_{1}$")
    ]

    for column, color, label in plot_settings:
        data_sindy.plot(x="time", y=column, ax=ax1, legend=False, color=color, linestyle='--', label=label + " (SINDy)")
        ax1.scatter(data_sindy["time"], data_sindy[column], color=color, edgecolors="black", s=10, zorder=3) 
    
    for column, color, label in plot_settings:
        data_edmd.plot(x="time", y=column, ax=ax1, legend=False, color=color, linestyle='-', label=label + " (Koopman)")
        ax1.scatter(data_edmd["time"], data_edmd[column], color=color, edgecolors="black", s=10, zorder=3) 


    ax1.set_xlabel(r'Training set (s)', fontsize=20, weight="bold")
    ax1.set_ylabel(r'RMSE (rad/s)', fontsize=20, weight="bold")
    plt.ylim(0, 10)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=18, weight="bold")
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=18, weight="bold")
    plt.xlim(4, 31)
    ax1.legend(loc='upper right', labelspacing=0.3, handlelength=1, bbox_to_anchor=(1, 1), prop={'size': 16, 'weight': 'bold'})
    plt.tight_layout()
    plt.grid()
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'RMSE.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()
    


if __name__ == "__main__":
    
    file_path_sindy = "~/Workspace/koopman_mpc/koopman_model/Data/Model/RMSE/SINDy_RMSE.csv"
    file_path_edmd = "~/Workspace/koopman_mpc/koopman_model/Data/Model/RMSE/Koopman_RMSE.csv"
    plot(file_path_sindy, file_path_edmd) 
   