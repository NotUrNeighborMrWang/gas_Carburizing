'''
说明：
    - 代码测试
功能：
    - 强渗阶段C浓度分布模拟
数据域：
    - t:0,1
    - x:0,5
'''


# Libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from deepxde.backend import tf
import deepxde as dde
import numpy as np
import math
from scipy.interpolate import CubicSpline

from math import erfc, sqrt

import torch
print(torch.__version__)


# Constants ----------------------------------------------
LENGTH_OF_DOMAIN = 5
TIME = 1
INITIAL_CONC = 0.23
c_bar = 0.92


# Plotting Class for comparision
class plotly_compare():

    def __init__(self, arr_X, name_models):
        if len(arr_X) != len(name_models):
            raise NotImplementedError("X and name arr are of different lengths i.e. {} and {}".format(
                len(arr_X), len(name_models)))
        if len(arr_X) == 0 or len(arr_X[0]) < 2:
            raise NotImplementedError("Please check dimensions of X")
        self.arr_X = arr_X
        self.name_models = name_models
        self.nuller = np.zeros(len(arr_X[0]), dtype="float").reshape(-1, 1)
        if len(arr_X[0][0]) == 2:
            for i in range(len(arr_X)):
                arr_X[i] = np.insert(arr_X[i], [0], self.nuller, axis=1)
        if len(arr_X[0][0]) == 3:
            for i in range(len(arr_X)):
                arr_X[i] = np.insert(arr_X[i], [2], self.nuller, axis=1)
        self.arr_n = np.array([])
        for X in arr_X:
            self.arr_n = np.append(self.arr_n, len(X))
        self.color_row = np.array([])
        for i in range(len(self.arr_n)):
            self.color_row = np.hstack(
                (self.color_row, np.repeat(name_models[i], self.arr_n[i])))
        self.x = np.array([])
        self.y = np.array([])
        self.t = np.array([])
        self.u = np.array([])
        for i in range(len(arr_X)):
            self.x = np.hstack((self.x, np.array(arr_X[i][:, 0: 1]).flatten()))
            self.y = np.hstack((self.y, np.array(arr_X[i][:, 1: 2]).flatten()))
            self.u = np.hstack((self.u, np.array(arr_X[i][:, 2: 3]).flatten()))
            self.t = np.hstack((self.t, np.array(arr_X[i][:, 3: 4]).flatten()))
        self.unified_X = np.array(
            [self.x, self.y, self.u, self.t, self.color_row]).T

    def creater_plot(self, x=None, y=None, t=None, error_range=0.0001):
        df = pd.DataFrame()

        def fil(X):
            cond = True
            if x is not None:
                cond = (cond and abs(round(float(X[0]), 2)-x) <= error_range)
            if y is not None:
                cond = (cond and abs(round(float(X[1]), 2)-y) <= error_range)
            if t is not None:
                cond = (cond and abs(round(float(X[3]), 2)-t) <= error_range)
            return cond
        filtered_X = self.unified_X[np.array(list(map(fil, self.unified_X)))]
        df["x"] = filtered_X[:, 0: 1].flatten().astype("float64").round(4)
        df["u"] = filtered_X[:, 1: 2].flatten().astype("float64").round(8)
        df["color_row"] = filtered_X[:, 4: 5].flatten()
        df.sort_values(by="x")
        fig = px.line(df, x="x", y="u", color="color_row",
                      title="time frame:{}s".format(t), hover_data="x")
        # y_org = df[df["color_row"] == "pinn"]
        # x_org = y_org["x"]
        # y_org = y_org["u"]
        # y_cal = df[df["color_row"] == "analytical"]
        # x_cal = y_cal["x"]
        # y_cal = y_cal["u"]
        # plt.plot(x_org, y_org, "-*")
        # plt.plot(x_cal, y_cal)
        # plt.legend(["PINN", "Analytical"], loc="best")
        # plt.title(f"Time frame: {round(t, 2)}s")
        # plt.xlabel("distance")
        # plt.ylabel("concentration")
        # plt.ylim((0, 1.5))

        # save_path = "./images/" + str(round(t, 2)) + ".jpg"
        # if save_path:
        #     plt.savefig(save_path)
        #     plt.close()  # 关闭图形以确保下一次绘图不会受到先前图形的影响
        # else:
        #     plt.show()

        return fig, df
    

def l_boundary(X, on_boundary):
    return on_boundary and np.isclose(X[0], 0)


# PDE definition
def pde(X, u):
    du_dt = dde.grad.jacobian(u, X, i=0, j=1)
    ddu_dxx = dde.grad.hessian(u, X, i=0, j=0)
    return du_dt - D * (ddu_dxx)


def ref_sol(x, t):
    if np.isclose(t, 0):
        return 0
    u = c_bar * (erfc(x / (2 * np.sqrt(D * t))))
    return u


def gen_exactSol(X):
    y = np.zeros(len(X))
    for i in range(len(X)):
        y[i] = ref_sol(X[i][0], X[i][1])
    return y


def train_1st(opt):

    global D
    D = opt.Dc

    # Domain Definition ----------------------------------------------
    LineDomain = dde.geometry.Interval(0, LENGTH_OF_DOMAIN)
    TimeDomain = dde.geometry.TimeDomain(0, TIME)
    CombinedDomain = dde.geometry.GeometryXTime(LineDomain, TimeDomain)

    # Boundary Condition ----------------------------------------------
    l_bc = dde.icbc.DirichletBC(CombinedDomain, lambda X: c_bar, l_boundary)
    initial_bc = dde.icbc.IC(CombinedDomain, lambda X: INITIAL_CONC, lambda _, on_initial: on_initial)

    # PDE module ----------------------------------------------
    module = dde.data.TimePDE(
        CombinedDomain,
        pde,
        [l_bc, initial_bc],
        num_domain=5000,
        num_boundary=1200,
        num_initial=1200,
        num_test=5000,
    )

    # Defining and Compiling Model ----------------------------------------------
    net = dde.nn.FNN([2] + [100] * 2 + [1], "tanh", "Glorot normal")
    model = dde.Model(module, net)
    model.compile("adam", lr=0.0005)

    # Training Model ----------------------------------------------
    LossHistory, TrainState = model.train(iterations=opt.ph1_iterations_num)
    dde.saveplot(LossHistory, TrainState, issave=True, isplot=False,
                 loss_fname = opt.ph1_loss_fname_path,
                 train_fname= opt.ph1_train_fname_path,
                 test_fname= opt.ph1_test_fname_path)
    dde.utils.external.dat_to_csv(opt.ph1_test_dat_path, opt.ph1_test_csv_path,["x","t","y"])

    # Finding PINN Solution ----------------------------------------------
    ux = np.linspace(0, int(LENGTH_OF_DOMAIN), (int(LENGTH_OF_DOMAIN) * opt.ph1_step_ux) + 1, dtype="float")
    ut = np.linspace(0, TIME, (TIME * opt.ph1_step_ut) + 1, dtype="float")
    x, t = np.meshgrid(ux, ut)
    X = np.array([x.flatten(), t.flatten()]).T

    predicted_c = model.predict(X)
    model_X_pinn = np.insert(X, [1], predicted_c.reshape((-1, 1)), axis=1)

    # Finding Analytical Solution ----------------------------------------------
    y = gen_exactSol(X)
    model_X_analytical = np.insert(X, [1], y.reshape((-1, 1)), axis=1)

    # Plotting Comparision graphs ----------------------------------------------
    comparision = plotly_compare([model_X_pinn, model_X_analytical], ["pinn", "analytical"])
    for i in range(1, len(ut)):
        fig, df = comparision.creater_plot(t=ut[i], error_range=0.01)
        df.to_csv(str(opt.ph1_data_t_save_path) + str(round(ut[i], 2)) + ".csv", index=True)
