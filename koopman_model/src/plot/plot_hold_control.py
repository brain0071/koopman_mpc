import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D 
import os


def plot_control_data(linear_file_path, sindy_file_path, koopman_file_path):
    
    df_linear = pd.read_csv(linear_file_path)
    df_sindy = pd.read_csv(sindy_file_path)
    df_koopman = pd.read_csv(koopman_file_path)
    
    df_linear["u_4"] *= 100
    df_sindy["u_4"] *= 100
    df_koopman["u_4"] *= 100

    df_linear["u_5"] *= 100
    df_sindy["u_5"] *= 100
    df_koopman["u_5"] *= 100

    df_linear["u_6"] *= 100
    df_sindy["u_6"] *= 100
    df_koopman["u_6"] *= 100

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42    
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    mpl.rcParams.update({
        'font.size': 25,          
        'axes.labelsize': 25,     
        'xtick.labelsize': 20,    
        'ytick.labelsize': 20,    
        'legend.fontsize': 20     
    })
    
    ylabel_position_1 = -0.12 
    ylabel_position_2 = -0.10
    ylabel_position_3 = -0.15

    fig = plt.figure(figsize=(20, 10))  
    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.6, hspace=0.3, top=0.84)

    legend_handles = [
        Line2D([0], [0], color='b', lw=2),  
        Line2D([0], [0], color='r', lw=2),  
        Line2D([0], [0], color='g', lw=2), 
    ] 
    legend_labels = ["Linear + MPC", "SINDy + MPC", "Koopman + MPC"]  
    leg = fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),           
        bbox_transform=fig.transFigure,       
        ncol=3, frameon=True, framealpha=1.0,
        borderpad=0.4, handlelength=2.0, columnspacing=1.5,
        prop={'weight': 'bold', 'size': 24}   
    )

    leg.get_frame().set_edgecolor("k")                
    leg.get_frame().set_linewidth(1.2) 

    ax1 = fig.add_subplot(3, 3, 1)
    df_linear.plot(x="time", y="ref_phi", ax=ax1, legend=False, color='black', linewidth=2, linestyle="--")
    df_linear.plot(x="time", y="phi", ax=ax1, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="phi", ax=ax1, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="phi", ax=ax1, legend=False, color="g", linewidth=2)
    ax1.set_xlabel(r'Time (s)', weight="bold")  
    ax1.set_ylabel(r'$\phi$ (rad)', weight="bold")
    ax1.yaxis.set_label_coords(ylabel_position_1, 0.5)
    ax1.tick_params(width=2)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight("bold")
    plt.yticks([0, 0.3, 0.6])
    plt.ylim(0, 0.6)
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.grid(True)

    axins = inset_axes(ax1, width="80%", height="40%", loc="lower right", bbox_to_anchor=(0, 0.11, 1, 1), 
                       bbox_transform=ax1.transAxes)
    df_linear.plot(x="time", y="ref_phi", ax=axins, legend=False, color='black', linewidth=2, linestyle="--")
    df_linear.plot(x="time", y="phi", ax=axins, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="phi", ax=axins, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="phi", ax=axins, legend=False, color="g", linewidth=2)
    axins.set_xlim(10, 60)   
    axins.set_yticks([0.5, 0.55])
    axins.set_ylim(0.5, 0.55) 
    axins.tick_params(labelsize=16, width=2)   
    for label in (axins.get_xticklabels() + axins.get_yticklabels()):   
        label.set_fontweight("bold")
    axins.set_xlabel("")

    ax2 = fig.add_subplot(3, 3, 4)
    df_linear.plot(x="time", y="ref_theta", ax=ax2, legend=False, color='black', linewidth=2, linestyle="--")
    df_linear.plot(x="time", y="theta", ax=ax2, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="theta", ax=ax2, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="theta", ax=ax2, legend=False, color="g", linewidth=2)
    ax2.set_xlabel(r'Time (s)', weight="bold")  
    ax2.set_ylabel(r'$\theta$ (rad)', weight="bold")
    ax2.yaxis.set_label_coords(ylabel_position_1, 0.5)
    ax2.tick_params(width=2)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight("bold")
    plt.yticks([-0.1, 0.0, 0.1])
    plt.ylim(-0.1, 0.1)
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.grid(True)
    
    ax3 = fig.add_subplot(3, 3, 7)
    df_linear.plot(x="time", y="ref_psi", ax=ax3, legend=False, color='black', linewidth=2, linestyle="--")
    df_linear.plot(x="time", y="psi", ax=ax3, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="psi", ax=ax3, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="psi", ax=ax3, legend=False, color="g", linewidth=2)
    ax3.set_xlabel(r'Time (s)', weight="bold")  
    ax3.set_ylabel(r'$\psi$ (rad)', weight="bold")
    ax3.yaxis.set_label_coords(ylabel_position_1, 0.5)
    ax3.tick_params(width=2)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontweight("bold")
    plt.yticks([-0.3, 0, 0.3])
    plt.ylim(-0.3, 0.3)
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.grid(True)

    axins3 = inset_axes(ax3, width="80%", height="30%", loc="lower right", 
                        bbox_to_anchor=(0, 0.06, 1, 1), bbox_transform=ax3.transAxes)
    df_linear.plot(x="time", y="ref_psi", ax=axins3, legend=False, color='black', linewidth=2, linestyle="--")
    df_linear.plot(x="time", y="psi", ax=axins3, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="psi", ax=axins3, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="psi", ax=axins3, legend=False, color="g", linewidth=2)
    axins3.set_xlim(10, 60)   
    axins3.set_yticks([-0.01, 0.02])             
    axins3.set_ylim(-0.01, 0.02)           
    axins3.tick_params(labelsize=16, width=2)
    for label in (axins3.get_xticklabels() + axins3.get_yticklabels()):
        label.set_fontweight("bold")
    axins3.set_xlabel("")                  
    axins3.set_ylabel("")                  


    ax4 = fig.add_subplot(3, 3, 2)
    df_linear.plot(x="time", y="u_4", ax=ax4, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="u_4", ax=ax4, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="u_4", ax=ax4, legend=False, color="g", linewidth=2)
    ax4.set_xlabel(r'Time (s)', weight="bold")  
    ax4.set_ylabel(r'$u_{p}$ (%)', weight="bold")
    ax4.yaxis.set_label_coords(ylabel_position_2, 0.5)
    ax4.tick_params(width=2)
    for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([12, 15, 18])
    plt.ylim(12, 18)
    plt.grid(True)

    ax5 = fig.add_subplot(3, 3, 5)
    df_linear.plot(x="time", y="u_5", ax=ax5, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="u_5", ax=ax5, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="u_5", ax=ax5, legend=False, color="g", linewidth=2)
    ax5.set_xlabel(r'Time (s)', weight="bold")  
    ax5.set_ylabel(r'$u_{q}$ (%)', weight="bold")
    ax5.yaxis.set_label_coords(ylabel_position_2, 0.5)
    ax5.tick_params(width=2)
    for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([-5, 0, 5])
    plt.ylim(-5, 5)
    plt.grid(True)
    
    ax6 = fig.add_subplot(3, 3, 8)
    df_linear.plot(x="time", y="u_6", ax=ax6, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="u_6", ax=ax6, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="u_6", ax=ax6, legend=False, color="g", linewidth=2)
    ax6.set_xlabel(r'Time (s)', weight="bold")  
    ax6.set_ylabel(r'$u_{r}$ (%)', weight="bold")
    ax6.yaxis.set_label_coords(ylabel_position_2, 0.5)
    ax6.tick_params(width=2)
    for label in (ax6.get_xticklabels() + ax6.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([-10, 0, 10])
    plt.ylim(-10, 10)
    plt.grid(True)

    axins6 = inset_axes(ax6, width="80%", height="25%", loc="lower right",
                        bbox_to_anchor=(0, 0.07, 1, 1), bbox_transform=ax6.transAxes)
    df_linear.plot(x="time", y="u_6", ax=axins6, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="u_6", ax=axins6, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="u_6", ax=axins6, legend=False, color="g", linewidth=2)
    axins6.set_xlim(10, 60)      
    axins6.set_ylim(-0.5, 0.5)   
    axins6.set_yticks([-0.5, 0.5])  
    axins6.tick_params(labelsize=16, width=2)
    for label in (axins6.get_xticklabels() + axins6.get_yticklabels()):
        label.set_fontweight("bold")
    axins6.set_xlabel("")        
    axins6.set_ylabel("")        


    ax7 = fig.add_subplot(3, 3, 3)
    df_linear.plot(x="time", y="phi_error", ax=ax7, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="phi_error", ax=ax7, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="phi_error", ax=ax7, legend=False, color="g", linewidth=2)
    ax7.set_xlabel(r'Time (s)', weight="bold")
    ax7.set_ylabel(r'$\phi$ Error (rad)', weight="bold")
    ax7.yaxis.set_label_coords(ylabel_position_3, 0.5)
    ax7.tick_params(width=2)
    for label in (ax7.get_xticklabels() + ax7.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.25, 0.5])
    plt.ylim(0.00, 0.5)
    plt.grid(True)

    axins7 = inset_axes(ax7, width="80%", height="45%", loc="upper right",
                        bbox_to_anchor=(0, -0.05, 1, 1), bbox_transform=ax7.transAxes)
    df_linear.plot(x="time", y="phi_error", ax=axins7, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="phi_error", ax=axins7, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="phi_error", ax=axins7, legend=False, color="g", linewidth=2)
    axins7.set_xlim(10, 60)      
    axins7.set_ylim(0.00, 0.02)   
    axins7.set_yticks([0.00, 0.02]) 
    axins7.tick_params(labelsize=16, width=2)
    for label in (axins7.get_xticklabels() + axins7.get_yticklabels()):
        label.set_fontweight("bold")
    axins7.set_xlabel("")        
    axins7.set_ylabel("")         

    ax8 = fig.add_subplot(3, 3, 6)  
    df_linear.plot(x="time", y="theta_error", ax=ax8, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="theta_error", ax=ax8, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="theta_error", ax=ax8, legend=False, color="g", linewidth=2)
    ax8.set_xlabel(r'Time (s)', weight="bold")
    ax8.set_ylabel(r'$\theta$ Error (rad)', weight="bold")
    ax8.yaxis.set_label_coords(ylabel_position_3, 0.5)
    ax8.tick_params(width=2)
    for label in (ax8.get_xticklabels() + ax8.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.05, 0.1])
    plt.ylim(0.0, 0.1)
    plt.grid(True)

    ax9 = fig.add_subplot(3, 3, 9)    
    df_linear.plot(x="time", y="psi_error", ax=ax9, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="psi_error", ax=ax9, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="psi_error", ax=ax9, legend=False, color="g", linewidth=2) 
    ax9.set_xlabel(r'Time (s)', weight="bold")
    ax9.set_ylabel(r'$\psi$ Error (rad)', weight="bold")
    ax9.yaxis.set_label_coords(ylabel_position_3, 0.5)
    ax9.tick_params(width=2)
    for label in (ax9.get_xticklabels() + ax9.get_yticklabels()):
        label.set_fontweight("bold")
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.15, 0.3])
    plt.ylim(0.0, 0.3)
    plt.grid(True)

    axins9 = inset_axes(ax9, width="80%", height="45%", loc="upper right",
                        bbox_to_anchor=(0, -0.05, 1, 1), bbox_transform=ax9.transAxes)
    df_linear.plot(x="time", y="psi_error", ax=axins9, legend=False, color="b", linewidth=2)
    df_sindy.plot(x="time", y="psi_error", ax=axins9, legend=False, color="r", linewidth=2)
    df_koopman.plot(x="time", y="psi_error", ax=axins9, legend=False, color="g", linewidth=2)
    axins9.set_xlim(10, 60)      
    axins9.set_yticks([0.00, 0.02])             
    axins9.set_ylim(0.00, 0.02)    
    axins9.tick_params(labelsize=16, width=2)
    for label in (axins9.get_xticklabels() + axins9.get_yticklabels()):
        label.set_fontweight("bold")
    axins9.set_xlabel("")        
    axins9.set_ylabel("")       

    
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'Attitude_Hold.pdf')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_file, format='pdf')
    plt.show()
        

if __name__ == "__main__":

    linear_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Hold/LinearMPC.csv"
    sindy_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Hold/SINDyMPC.csv"
    koopman_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Hold/KoopmanMPC.csv"

    plot_control_data(linear_file_path, sindy_file_path, koopman_file_path)
    
    
 
