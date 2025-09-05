import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import os

def plot(file_path):
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42 
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    
    data = pd.read_csv(file_path)
    r = data["phi"].values
    u = data["u_4"].values * 54.63

    r_sin = np.sin(r)
    r_sin_reshaped = r_sin.reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(r_sin_reshaped, u)
    a = model.coef_[0]  

    plt.figure(figsize=(14, 8))
    plt.scatter(r_sin, u, color='steelblue', alpha=1, label='Measured data', marker='o')  
    plt.plot(r_sin, a * r_sin, color='darkorange', label='Linear Regression', linewidth=2)
    plt.xlabel(r'sin $\phi$', fontsize=40, weight="bold")
    plt.ylabel(r'$u_{p} \tau_{max}(p)$ (N$\cdot$m)', fontsize=40, weight="bold") 
    plt.yticks([-15, -10, -5, 0, 5, 10, 15], fontsize=35, weight="bold")
    plt.ylim(-15, 15)
    plt.xticks([-1.00, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.00], fontsize=35, weight="bold")
    plt.xlim(-1.00, 1.00)
    plt.legend(loc='upper left', prop={'size': 35, 'weight': 'bold'})
    plt.tight_layout(rect=(0, 0, 1, 0.96)) 
    plt.grid()
    output_folder = '/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Fig'  
    output_file = os.path.join(output_folder, 'Linear_Regression.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    
    file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Model/Linear_Model/Linear_Data.csv"
    plot(file_path)
   
