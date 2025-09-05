import pykoopman as pk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import csv

warnings.filterwarnings("ignore")
import time

class EDMDc_Model:

    def __init__(self, train_file_path, test_file_path):

        self.r = None  # angular velocity
        self.u = None  

        self.r_test = None
        self.u_test = None

        df_train = pd.read_csv(train_file_path)
        df_test = pd.read_csv(test_file_path)
        self.load_data(df_train, df_test)
        self.learn()
        self.write_data()

    def load_data(self, df_train, df_test):

        self.r = np.column_stack((df_train["p"].values, df_train["q"].values, df_train["r"].values))
        self.u = np.column_stack((df_train["u_4"].values, df_train["u_5"].values, df_train["u_6"].values,
                                  np.cos(df_train["theta"].values) * np.cos(df_train["phi"].values),
                                  np.cos(df_train["theta"].values) * np.sin(df_train["phi"].values),
                                  np.sin(df_train["theta"].values)))

        self.r_test = np.column_stack((df_test["p"].values, df_test["q"].values, df_test["r"].values))

        self.u_test = np.column_stack((df_test["u_4"].values, df_test["u_5"].values, df_test["u_6"].values,
                                       np.cos(df_test["theta"].values) * np.cos(df_test["phi"].values),
                                       np.cos(df_test["theta"].values) * np.sin(df_test["phi"].values),
                                       np.sin(df_test["theta"].values)))
        
        self.t_test = df_test["time"].values

    def learn(self):
        
        EDMDc = pk.regression.EDMDc()

        # p
        # + q
        # + r
        # + p * p
        # + p * q
        # + p * r
        # + q * q
        # + q * r
        # + r * r
        # + p * p * p
        # + q * q * q
        # + r * r * r
        # + p * p * q
        # + p * p * r
        # + p * q * r
        # + p * r * r
        # + p * q * q
        # + q * q * r
        # + q * r * r
        
        observables = [lambda x: x ** 2, lambda x: x ** 3, lambda x, y: x * y, lambda x, y: x ** 2 * y, lambda x, y: x * y ** 2, lambda x, y, z: x * y * z]
        observable_names = [
            lambda x: f"{x}^2",
            lambda x: f"{x}^3",
            lambda x, y: f"{x} {y}",
            lambda x, y: f"{x}^2 {y}",
            lambda x, y: f"{x} {y}^2",
            lambda x, y, z: f"{x} {y} {z}",
        ]
        obs = pk.observables.CustomObservables(
            observables, observable_names=observable_names
        )
        model = pk.Koopman(observables=obs, regressor=EDMDc)
        last = time.time()
        print(last)
        model.fit(self.r, u=self.u, dt=0.05)
        new = time.time()
        print(new)
        print("time:",  new - last)
        print("A:")
        print(model.A.shape)
        print("B")
        print(model.B[:3])
        print(model.get_feature_names())
        print(model.C)
        # 
        self.xpred = model.simulate(self.r_test[0,:], self.u_test, n_steps=self.r_test.shape[0])
        
    def write_data(self):
        for i in range(self.t_test.shape[0]):
            new_row_data = {"edmd_p": self.xpred[i, 0], "edmd_q": self.xpred[i, 1], "edmd_r": self.xpred[i, 2]}
            with open(
                "/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/sim.csv",
                "a",
                newline="",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=new_row_data.keys())
                writer.writerow(new_row_data)
        

def main():

    train_file_path = ("~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/T_1.csv")
    test_file_path = ("~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/V_1.csv")
    EDMDc_Model(train_file_path, test_file_path)

if __name__ == "__main__":
    main()
