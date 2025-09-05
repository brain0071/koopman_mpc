import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle, ConnectionPatch
import os
from matplotlib.lines import Line2D 


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
    plt.yticks([-0.6, 0, 0.6])
    plt.ylim(-0.6, 0.6)
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.grid(True)

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
    plt.yticks([-0.1, 0, 0.1])
    plt.ylim(-0.1, 0.1)
    plt.xticks([0, 20, 40, 60])
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
    plt.yticks([-0.1, 0, 0.1])
    plt.ylim(-0.1, 0.1)
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.grid(True)
    
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
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.yticks([-20, 0, 20])
    plt.ylim(-20, 20)
    plt.grid(True)

    x_left_0, x_right_0 = 5, 14      
    y_lower_0, y_upper_0 = -16, -7      
    ax4ins = inset_axes(ax4, width="22%", height="47%", loc="upper right", 
                        bbox_to_anchor=(-0.71, 0.10, 1, 1), 
                        bbox_transform=ax4.transAxes,  
                        borderpad=1.0)
    df_linear.plot(x="time", y="u_4", ax=ax4ins, legend=False, color="b", linewidth=1.6)
    df_sindy.plot(x="time", y="u_4", ax=ax4ins, legend=False, color="r", linewidth=1.6)
    df_koopman.plot(x="time", y="u_4", ax=ax4ins, legend=False, color="g", linewidth=1.6)
    ax4ins.set_xlim(x_left_0, x_right_0)       
    ax4ins.set_ylim(y_lower_0, y_upper_0)       
    ax4ins.tick_params(labelsize=12, width=1.6)   
    for lab in (ax4ins.get_xticklabels() + ax4ins.get_yticklabels()):  
        lab.set_fontweight("bold")
    ax4ins.set_xlabel("")          
    ax4ins.set_ylabel("")
    ax4ins.set_xticks([])
    ax4ins.set_yticks([])         

    rect = Rectangle((x_left_0, y_lower_0), x_right_0-x_left_0, y_upper_0-y_lower_0, fill=False, ls="--", lw=1.6, ec="k")
    ax4.add_patch(rect)
    con1 = ConnectionPatch(xyA=(x_right_0, y_upper_0), coordsA=ax4.transData,
                           xyB=(1, 1),    coordsB=ax4ins.transAxes,
                           ls="--", lw=1.2, color="k")
    con2 = ConnectionPatch(xyA=(x_left_0, y_upper_0), coordsA=ax4.transData,
                           xyB=(0, 0),    coordsB=ax4ins.transAxes,
                           ls="--", lw=1.2, color="k")
    ax4.add_artist(con1)
    ax4.add_artist(con2)
    x_left_1, x_right_1 = 18, 27
    y_lower_1, y_upper_1 = 8, 17
    ax4ins2 = inset_axes(
        ax4,
        width="22%", height="47%",
        loc="lower right",   
        bbox_to_anchor=(-0.46, -0.11, 1, 1), 
        bbox_transform=ax4.transAxes,                  
        borderpad=1.0
    )

    df_linear.plot(x="time", y="u_4", ax=ax4ins2, legend=False, color="b", linewidth=1.6)
    df_sindy.plot(x="time", y="u_4", ax=ax4ins2, legend=False, color="r", linewidth=1.6)
    df_koopman.plot(x="time", y="u_4", ax=ax4ins2, legend=False, color="g", linewidth=1.6)
    ax4ins2.set_xlim(x_left_1, x_right_1)
    ax4ins2.set_ylim(y_lower_1, y_upper_1)
    ax4ins2.set_xticks([])
    ax4ins2.set_yticks([])
    ax4ins2.set_xlabel("")
    ax4ins2.set_ylabel("")
    rect2 = Rectangle(
        (x_left_1, y_lower_1),
        x_right_1 - x_left_1,
        y_upper_1 - y_lower_1,
        fill=False, ls="--", lw=1.6, ec="k"
    )
    ax4.add_patch(rect2)
    con3 = ConnectionPatch(xyA=(x_right_1, y_lower_1), coordsA=ax4.transData,
                       xyB=(1, 1), coordsB=ax4ins2.transAxes,
                       ls="--", lw=1.2, color="k")
    con4 = ConnectionPatch(xyA=(x_left_1, y_lower_1), coordsA=ax4.transData,
                       xyB=(0, 0), coordsB=ax4ins2.transAxes,
                       ls="--", lw=1.2, color="k")
    ax4.add_artist(con3)
    ax4.add_artist(con4)

    x_left_2, x_right_2 = 35, 44
    y_lower_2, y_upper_2 = -16, -7
    ax4ins3 = inset_axes(
        ax4,
        width="22%", height="47%",
        loc="upper right",
        bbox_to_anchor=(-0.21, 0.1, 1, 1),   
        bbox_transform=ax4.transAxes,
        borderpad=1.0
    )
    df_linear.plot(x="time", y="u_4", ax=ax4ins3, legend=False, color="b", linewidth=1.6)
    df_sindy.plot(x="time", y="u_4", ax=ax4ins3, legend=False, color="r", linewidth=1.6)
    df_koopman.plot(x="time", y="u_4", ax=ax4ins3, legend=False, color="g", linewidth=1.6)
    ax4ins3.set_xlim(x_left_2, x_right_2)
    ax4ins3.set_ylim(y_lower_2, y_upper_2)
    ax4ins3.set_xticks([])
    ax4ins3.set_yticks([])
    ax4ins3.set_xlabel("")
    ax4ins3.set_ylabel("")
    rect3 = Rectangle(
        (x_left_2, y_lower_2),
        x_right_2 - x_left_2,
        y_upper_2 - y_lower_2,
        fill=False, ls="--", lw=1.6, ec="k"
    )
    ax4.add_patch(rect3)
    con5 = ConnectionPatch(xyA=(x_right_2, y_upper_2), coordsA=ax4.transData,
                           xyB=(1, 1), coordsB=ax4ins3.transAxes,
                           ls="--", lw=1.2, color="k")
    con6 = ConnectionPatch(xyA=(x_left_2, y_upper_2), coordsA=ax4.transData,
                           xyB=(0, 0), coordsB=ax4ins3.transAxes,
                           ls="--", lw=1.2, color="k")
    ax4.add_artist(con5)
    ax4.add_artist(con6)
    

    x_left_3, x_right_3 = 48, 57
    y_lower_3, y_upper_3 = 8, 17
    ax4ins4 = inset_axes(
        ax4,
        width="22%", height="47%",
        loc="lower right",
        bbox_to_anchor=(0.04, -0.11, 1, 1),   
        bbox_transform=ax4.transAxes,
        borderpad=1.0
    )

    df_linear.plot(x="time", y="u_4", ax=ax4ins4, legend=False, color="b", linewidth=1.6)
    df_sindy.plot(x="time", y="u_4", ax=ax4ins4, legend=False, color="r", linewidth=1.6)
    df_koopman.plot(x="time", y="u_4", ax=ax4ins4, legend=False, color="g", linewidth=1.6)
    ax4ins4.set_xlim(x_left_3, x_right_3)
    ax4ins4.set_ylim(y_lower_3, y_upper_3)
    ax4ins4.set_xticks([])
    ax4ins4.set_yticks([])
    ax4ins4.set_xlabel("")
    ax4ins4.set_ylabel("")
    rect4 = Rectangle(
        (x_left_3, y_lower_3),
        x_right_3 - x_left_3,
        y_upper_3 - y_lower_3,
        fill=False, ls="--", lw=1.6, ec="k"
    )
    ax4.add_patch(rect4)
    con7 = ConnectionPatch(xyA=(x_right_3, y_lower_3), coordsA=ax4.transData,
                           xyB=(1, 1), coordsB=ax4ins4.transAxes,
                           ls="--", lw=1.2, color="k")
    con8 = ConnectionPatch(xyA=(x_left_3, y_lower_3), coordsA=ax4.transData,
                           xyB=(0, 0), coordsB=ax4ins4.transAxes,
                           ls="--", lw=1.2, color="k")
    ax4.add_artist(con7)
    ax4.add_artist(con8)


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
    plt.xticks([0, 20, 40, 60])
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
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.yticks([-5, 0, 5])
    plt.ylim(-5, 5)
    plt.grid(True)

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
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.15, 0.3])
    plt.ylim(0.0, 0.3)
    plt.grid(True)

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
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.03, 0.06])
    plt.ylim(0.0, 0.06)
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
    plt.xticks([0, 20, 40, 60])
    plt.xlim(0, 60)
    plt.yticks([0.0, 0.05, 0.10])
    plt.ylim(0.0, 0.10)
    plt.grid(True)

    
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'Attitude_Tracking.pdf')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_file, format='pdf')
    plt.show()
        

if __name__ == "__main__":
    
    linear_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Tracking/LinearMPC.csv"
    sindy_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Tracking/SINDyMPC.csv"
    koopman_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Control/Attitude_Tracking/KoopmanMPC.csv"

    plot_control_data(linear_file_path, sindy_file_path, koopman_file_path)
    
    
 
