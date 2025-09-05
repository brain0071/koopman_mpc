import numpy as np
import pandas as pd
import pysindy as ps
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

# sys.path.append("/home/naodai/Workspace/koopman_mpc/koopman_model")
# from src.sindy_model.plot_data import plot_simulate


class SINDy:

    def __init__(self, train_file_path, test_file_path):

        self.r = None  # angular velocity
        self.u = None  # u4 ~ u6 and R P Y
        self.t = None  # time

        self.r_test = None
        self.u_test = None
        self.t_test = None

        self.start_time = time.time()
        n_features = 21
        n_target = 3
        self.initial_guess = np.zeros((n_target, n_features))
        
        # initial_guess
        self.initial_guess[0, 0] = 45.0784
        self.initial_guess[1, 1] = 17.9005
        self.initial_guess[1, 4] = -0.5105
        self.initial_guess[2, 2] = 19.8220
        self.initial_guess[2, 5] = -3.0241

        self.origin_coef_ = np.zeros((n_target, n_features))
        self.origin_coef_[0, 0] = 45.0784
        self.origin_coef_[0, 13] = -11.4204
        self.origin_coef_[1, 1] = 17.9005
        self.origin_coef_[1, 4] = -0.5105
        self.origin_coef_[2, 2] = 19.8220
        self.origin_coef_[2, 5] = -3.0241

        # ['u4', 'u5', 'u6']
        identity_inputs_per_lib = [(3,), (3, 4, 5,),]
        identity_lib_2 = ps.IdentityLibrary()
        identity_lib_1 = ps.PolynomialLibrary(degree=0, include_bias=True)
        identity_tensor_array = [[1, 1]]
        identity_library = ps.GeneralizedLibrary(
            [identity_lib_1, identity_lib_2],
            tensor_array=identity_tensor_array,
            exclude_libraries=[0, 1,],
            inputs_per_library=identity_inputs_per_lib,)

        # ['p']
        p_inputs_per_lib = [(0,), (0,)]
        p_1 = ps.IdentityLibrary()
        p_2 = ps.PolynomialLibrary(degree=0, include_bias=True)
        p_tensor_array = [[1, 1]]
        p = ps.GeneralizedLibrary([p_1, p_2], tensor_array=p_tensor_array, exclude_libraries=[0, 1],
                                   inputs_per_library=p_inputs_per_lib,)

        # ['q']
        q_inputs_per_lib = [(1,), (0, 1, 2)]
        q_1 = ps.IdentityLibrary()
        q_2 = ps.PolynomialLibrary(degree=0, include_bias=True)
        q_tensor_array = [[1, 1]]
        q = ps.GeneralizedLibrary([q_1, q_2], tensor_array=q_tensor_array, exclude_libraries=[0, 1,],
                                  inputs_per_library=q_inputs_per_lib,)

        # ['r']
        r_inputs_per_lib = [(2,), (0, 1, 2)]
        r_1 = ps.IdentityLibrary()
        r_2 = ps.PolynomialLibrary(degree=0, include_bias=True)
        r_tensor_array = [[1, 1]]
        r = ps.GeneralizedLibrary([r_1, r_2], tensor_array=r_tensor_array, exclude_libraries=[0, 1,],
            inputs_per_library=r_inputs_per_lib,)

        poly_library = (
            p
            # + q
            # + r
            + p * p
            + p * q
            + p * r
            # + q * q
            # + q * r
            # + r * r
            + p * p * p
            # + q * q * q
            # + r * r * r
            + p * p * q
            + p * p * r
            + p * q * r
            + p * r * r
            + p * q * q
            # + q * q * r
            # + q * r * r
            # + p * p * p * p * p
            # + p * p * p * p * q
            # + p * p * p * p * r
            # + p * p * p * q * q
            # + p * p * p * q * r
            # + p * p * p * r * r
            # + p * p * q * q * q
            # + p * p * q * q * r
            # + p * p * q * r * r
            # + p * p * r * r * r
            # + p * q * q * q * q
            # + p * q * q * q * r
            # + p * q * q * r * r
            # + p * q * r * r * r
            # + p * r * r * r * r
            # + q * q * q * q * q
            # + q * q * q * q * r
            # + q * q * q * r * r
            # + q * q * p * r * r
            # + q * p * r * r * r
            # + r * r * r * r * r
        )

        # 'sin(1 R)', 'cos(1 R)', 'sin(1 P)', 'cos(1 P)',
        # 'sin(1 R) sin(1 P)', 'sin(1 R) cos(1 P)',
        # 'cos(1 R) sin(1 P)', 'cos(1 R) cos(1 P)'
        fourier_inputs_per_lib = [(6,), (7,)]
        fourier_lib_1 = ps.FourierLibrary()
        fourier_lib_2 = ps.FourierLibrary()
        fourier_tensor_array = [[1, 1]]
        fourier_library = ps.GeneralizedLibrary([fourier_lib_1, fourier_lib_2], tensor_array=fourier_tensor_array,
                                                 inputs_per_library=fourier_inputs_per_lib,)


        self.generalized_library = identity_library + poly_library + fourier_library
        self.differentiator_list = [("Savitzky", "c", ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3))]
        self.optimizer_list = [("MIOSR", ps.MIOSR(group_sparsity=(5, 5, 5), target_sparsity=None, 
                                                  initial_guess=self.initial_guess,),),]

        df_train = pd.read_csv(train_file_path)
        df_test = pd.read_csv(test_file_path)
        self.load_data(df_train, df_test)
        self.learn()

    def load_data(self, df_train, df_test):

        self.r = np.column_stack((np.column_stack((df_train["p"].values, df_train["q"].values.T)), 
                                  df_train["r"].values.T,))
        self.u = np.column_stack((np.column_stack((df_train["u_4"].values, df_train["u_5"].values.T)),
                                  df_train["u_6"].values.T, df_train["phi"].values.T, 
                                  df_train["theta"].values.T, df_train["psi"].values.T,))
        
        self.t = df_train["time"].values

        self.r_test = np.column_stack((np.column_stack((df_test["p"].values, df_test["q"].values.T)),
                                       df_test["r"].values.T,))
        self.u_test = np.column_stack((np.column_stack((df_test["u_4"].values, df_test["u_5"].values.T)),
                                       df_test["u_6"].values.T, df_test["phi"].values.T, df_test["theta"].values.T, 
                                       df_test["psi"].values.T,))
        self.t_test = df_test["time"].values

    def learn(self):
        """
        Parameters: 1.optimizer, 2.feature_library, 3.differentiation_method, 4.feature_names,
        """

        feature_names = ["p", "q", "r", "u4", "u5", "u6", "R", "P", "Y"]

        for diff_name, color, diff_method in self.differentiator_list:

            x_dot_precomputed = diff_method(self.r, self.t)

            for opt_name, opt in self.optimizer_list:

                self.model = ps.SINDy(optimizer=opt, feature_library=self.generalized_library,
                                      feature_names=feature_names,)

                start_time = time.time()
                self.model.fit(self.r, t=self.t, x_dot=x_dot_precomputed, u=self.u)
                stop_time = time.time()
                print("time:", stop_time - start_time)

                "u4", "u5", "u6",
                "p", "pp", "pq", "pr", "ppp", "ppq", "ppr", "pqr", "prr", "pqq",
                "sin(R)", "cos(R)", "sin(P)", "cos(P)", "sin(R)sin(P)", "sin(R)cos(P)", "cos(R)sin(P)", "cos(R)cos(P)"

                print("SINDy model: ")
                self.model.print()
                print(self.model.get_feature_names())

                # simulation
                self.plot_simulate(opt, feature_names)

    def plot_simulate(self, opt, feature_names):
        mpl.rcParams["font.family"] = "Times New Roman"
        x_test_sim = self.model.simulate(self.r_test[0], self.t_test, self.u_test, integrator="odeint")
        # origin model simulation
        opt.coef_ = self.origin_coef_
        origin_model_score = self.model.score(self.r_test, t=self.t_test, u=self.u_test)
        print("origin_model_score", origin_model_score)
        self.model.print()
        print("*******************************************************")
        x_origin_sim = self.model.simulate(self.r_test[0], self.t_test, self.u_test, integrator="odeint")
        self.t_test = self.t_test[:-1]  
        self.r_test = self.r_test[:-1, :]  

        for i in range(self.t_test.shape[0]):
            new_row_data = {"time": self.t_test[i],
                            "p": self.r_test[i, 0],
                            "q": self.r_test[i, 1],
                            "r": self.r_test[i, 2],
                            "sindy_p": x_test_sim[i, 0],
                            "sindy_q": x_test_sim[i, 1],
                            "sindy_r": x_test_sim[i, 2],
                            "linear_p": x_origin_sim[i, 0],
                            "linear_q": x_origin_sim[i, 1],
                            "linear_r": x_origin_sim[i, 2],}

            with open("/home/naodai/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/test.csv", "a",
                newline="",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=new_row_data.keys())
                writer.writerow(new_row_data)

        fig, axs = plt.subplots(self.r_test.shape[1], 1, sharex=True, figsize=(7, 9))
        for i in range(self.r_test.shape[1]):
            print(i)
            axs[i].plot(self.t_test, self.r_test[:, i], color="g", label="Measured data")
            axs[i].plot(self.t_test, x_test_sim[:, i], color="r", label="SINDy model simulation")
            axs[i].plot(self.t_test, x_origin_sim[:, i], color="b", label="Linear model simulation")
            axs[i].legend()
            axs[i].set(xlabel="t", ylabel=rf"${feature_names[i]}$")
        plt.show()

def main():

    train_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/T_1.csv"
    test_file_path = "~/Workspace/koopman_mpc/koopman_model/Data/Model/Koopman/V_1.csv"
    SINDy(train_file_path, test_file_path)
    
if __name__ == "__main__":
    main()
