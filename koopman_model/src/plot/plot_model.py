import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os


def plot_control_data(sixty_file_path, thirty_file_path):
    
    df_sixty = pd.read_csv(sixty_file_path)
    df_thirty = pd.read_csv(thirty_file_path)
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42  
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # v1
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    df_thirty.plot(x="time", y="p", ax=ax1, legend=False, color='k', linestyle='--', label="Measured Data")
    df_thirty.plot(x="time", y="linear_p", ax=ax1, legend=False, color="blue", label="Linear")
    df_thirty.plot(x="time", y="sindy_p", ax=ax1, legend=False, color="r", label="SINDy")
    df_thirty.plot(x="time", y="edmd_p", ax=ax1, legend=False, color="green", label="Koopman")
    ax1.set_xlabel(r'Time (s)', fontsize=30, weight="bold")  
    ax1.set_ylabel(r'$p$ (rad/s)', fontsize=30, weight="bold")
    plt.yticks([-20, -10, 0, 10, 20], fontsize=25, weight="bold")
    plt.ylim(-20, 20)
    plt.xticks([0, 10, 20, 30], fontsize=25, weight="bold")
    plt.xlim(0, 30)
    ax1.legend(loc='lower left', handlelength=0.8, ncol=2, columnspacing=0.3, handletextpad=0.2, bbox_to_anchor=(-0.02, 0), borderpad=0.2, prop={'size': 25, 'weight': 'bold'})    
    plt.tight_layout()
    plt.grid()
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'V1.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()    
    
    # v2
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    df_sixty.plot(x="time", y="p", ax=ax1, legend=False, color='k', linestyle='--', label="Measured Data")
    df_sixty.plot(x="time", y="linear_p", ax=ax1, legend=False, color="blue", label="Linear")
    df_sixty.plot(x="time", y="sindy_p", ax=ax1, legend=False, color="red", label="SINDy")
    df_sixty.plot(x="time", y="edmd_p", ax=ax1, legend=False, color="green", label="Koopman")
    
    ax1.set_xlabel(r'Time (s)', fontsize=30, weight="bold")  
    ax1.set_ylabel(r'$p$ (rad/s)', fontsize=30, weight="bold")
    plt.yticks([-30, -15, 0, 15, 30], fontsize=25, weight="bold")
    plt.ylim(-30, 30)
    plt.xticks([0, 10, 20, 30], fontsize=25, weight="bold")
    plt.xlim(0, 30)
    ax1.legend(loc='upper left', handlelength=0.8, ncol=2, columnspacing=0.3, handletextpad=0.2, bbox_to_anchor=(-0.02, 1.03), borderpad=0.2, labelspacing=0.3,  prop={'size': 25, 'weight': 'bold'})
    plt.tight_layout()
    plt.grid()    
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'V2.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()


    # v1 error
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    df_thirty.plot(x="time", y="linear_error", ax=ax1, legend=False, color="dodgerblue", label="Linear")
    df_thirty.plot(x="time", y="sindy_error", ax=ax1, legend=False, color="orange", label="SINDy")
    df_thirty.plot(x="time", y="edmd_error", ax=ax1, legend=False, color="darkgreen", label="Koopman")
    ax1.set_xlabel(r'Time (s)', fontsize=30, weight="bold")  
    ax1.set_ylabel(r'$p$ Error (rad/s)', fontsize=30, weight="bold")
    plt.yticks([0, 5, 10, 15], fontsize=25, weight="bold")
    plt.ylim(-1, 15)
    plt.xticks([0, 10, 20, 30], fontsize=25, weight="bold")
    plt.xlim(0, 30)
    ax1.legend(loc='upper left', handlelength=0.7, ncol=3, columnspacing=0.3, handletextpad=0.2, borderpad=0.2, bbox_to_anchor=(-0.02, 1.02), prop={'size': 25, 'weight': 'bold'})
    plt.tight_layout()
    plt.grid()
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'V1_Error.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()

    # v2 error
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    df_sixty.plot(x="time", y="linear_error", ax=ax1, legend=False, color="dodgerblue", label="Linear")
    df_sixty.plot(x="time", y="sindy_error", ax=ax1, legend=False, color="orange", label="SINDy") 
    df_sixty.plot(x="time", y="edmd_error", ax=ax1, legend=False, color="darkgreen", label="Koopman")    
    ax1.set_xlabel(r'Time (s)', fontsize=30, weight="bold")  
    ax1.set_ylabel(r'$p$ Error (rad/s)', fontsize=30, weight="bold")
    plt.yticks([0, 10, 20], fontsize=25, weight="bold")
    plt.ylim(-2, 20)
    plt.xticks([0, 10, 20, 30], fontsize=25, weight="bold")
    plt.xlim(0, 30)
    ax1.legend(loc='upper right', handlelength=0.8, ncol=3, columnspacing=0.3, handletextpad=0.2, bbox_to_anchor=(1.02, 1.02), borderpad=0.2, prop={'size': 25, 'weight': 'bold'})
    plt.tight_layout()
    plt.grid()
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'V2_Error.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()

   
if __name__ == "__main__":

    sixty_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/Sim_T2_v2.csv"
    thirty_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/Sim_V1_T1.csv"
    plot_control_data(sixty_file_path, thirty_file_path)