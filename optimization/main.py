import phase_1st.main_1st as phase1code
import phase_2nd.main_2nd as phase2code
import argparse
import numpy as np
import os
import pandas as pd
import platform
import time
import sys
from pathlib import Path
import logging
import inspect


def limit_elements(lst):
    """
    - 将列表中的元素都限制不超过两位小数
    """
    # 遍历列表中的每个元素
    for i in range(len(lst)):
        # 将元素转换为浮点数
        element = float(lst[i])
        # 限制小数点后两位
        element = round(element, 2)
        lst[i] = element
    return lst


def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument('--Dc', nargs='+', type=int, default=1.5, help="扩散系数")
    # 1st参数
    parser.add_argument('--ph1_iterations_num', nargs='+', type=int, default=10000, help="")
    parser.add_argument('--ph1_step_ux', nargs='+', type=int, default=50, help="数据保存的数量")
    parser.add_argument('--ph1_step_ut', nargs='+', type=int, default=50, help="数据保存的数量")
    parser.add_argument('--ph1_data_t_save_path', nargs='+', type=str, 
                        default="./phase_1st/data/data_", help="1st每个时刻的u-x数据保存地址")
    parser.add_argument('--ph1_test_dat_path', nargs='+', type=str, 
                        default="./phase_1st/log/test.dat" , help="1st的test文件保存地址")
    parser.add_argument('--ph1_test_csv_path', nargs='+', type=str, 
                        default="./phase_1st/log/test.csv", help="1st的test文件保存地址")
    parser.add_argument('--ph1_loss_fname_path', nargs='+', type=str, 
                        default="./phase_1st/log/loss.dat", help="训练/测试log")
    parser.add_argument('--ph1_train_fname_path', nargs='+', type=str, 
                        default="./phase_1st/log/train.dat", help="训练/测试log")
    parser.add_argument('--ph1_test_fname_path', nargs='+', type=str, 
                        default="./phase_1st/log/test.dat", help="训练/测试log")
    # 2nd参数
    parser.add_argument('--ph2_iterations_num', nargs='+', type=int, default=10000, help="")
    parser.add_argument('--ph2_step_ux', nargs='+', type=int, default=50, help="数据保存的数量")
    parser.add_argument('--ph2_step_ut', nargs='+', type=int, default=50, help="数据保存的数量")
    parser.add_argument('--ph2_data_t_save_path', nargs='+', type=str, 
                        default="./phase_2nd/data/data_", help="1st每个时刻的u-x数据保存地址")
    parser.add_argument('--ph2_test_dat_path', nargs='+', type=str, 
                        default="./phase_2nd/log/test_" , help="1st的test文件保存地址")
    parser.add_argument('--ph2_test_csv_path', nargs='+', type=str, 
                        default="./phase_2nd/log/test_", help="1st的test文件保存地址")
    parser.add_argument('--ph2_loss_fname_path', nargs='+', type=str, 
                        default="./phase_2nd/log/loss_", help="训练/测试log")
    parser.add_argument('--ph2_train_fname_path', nargs='+', type=str, 
                        default="./phase_2nd/log/train_", help="训练/测试log")
    parser.add_argument('--ph2_test_fname_path', nargs='+', type=str, 
                        default="./phase_2nd/log/test_", help="训练/测试log")


    opt = parser.parse_args()
    return opt

if __name__ == "__main__": 

    # 参数获取
    opt = parse_opt()

    flag_train_1st = input('是否进行强渗阶段的模型训练？(y or any key): ')
    flag_train_2nd = input('是否进行扩散阶段的模型训练？(y or any key): ')

    # Step1: 训练1st的模型 --------------------------------------------------------
    print("\n --------------------- Step1 ---------------------")
    time_train1_start = time.time()
    if flag_train_1st == "y":
        phase1code.train_1st(opt)   # 1st模型训练
    else:
        print("\n --------------------- Jump ---------------------")
    time_train1_end = time.time()   

    # Step2: 取1st的结果作为2nd的初始值，训练2nd的模型 --------------------------------------------------------
    print("\n --------------------- Step2 ---------------------")
    # 根据opt.ph1_step_ut判定强渗时间t1的取值范围: t1_list
    t1_list = []
    data_path = opt.ph1_test_csv_path
    csv_data = pd.read_csv(data_path, low_memory = False) #防止弹出警告
    csv_df = pd.DataFrame(csv_data)
    df_cash = csv_df[(csv_df["x"] == 1.0) & (csv_df["y"] > 0.3) & (csv_df["y"] < 0.4)]
    t1_list = df_cash["t"].values.tolist()
    print("t1_list:",t1_list)
    # 2nd模型训练
    time_train2_start = time.time()
    if flag_train_2nd == "y":
        for t1_1st in t1_list:
            # print("\n强渗时间: ", os.path.basename(t1_1st)[5:-4])
            print("ic对应的时间为: ",t1_1st)
            phase2code.train_2nd(opt,t1_1st)
    else:
        print("\n --------------------- Jump ---------------------")
    time_train2_end = time.time()
    
    # Step3: 从2nd的test.csv中找出1mm、0.4%的min(t) --------------------------------------------------------
    print("\n --------------------- Step3 ---------------------")
    ratio_list = []
    time_infer_start = time.time()
    for t1 in t1_list:
        csv_file = opt.ph2_test_csv_path + str(t1) + ".csv"
        csv_data = pd.read_csv(csv_file, low_memory = False) #防止弹出警告
        csv_df = pd.DataFrame(csv_data)
        new_df = csv_df[(csv_df["y"] >= 0.39) & (csv_df["y"] <= 0.4) & 
                        (csv_df["t"] > 0) & 
                        (csv_df["x"] >= 0.85) & (csv_df["x"] <= 0.95)]
        new_df = new_df.sort_values(by="t", ascending=True)

        # print("\n强渗时间为 {t1_time} 时符合条件的扩散结果: ".format(t1_time=t1))
        # print(new_df.head(1))
        if new_df.empty:
            pass
        else:
            t2 = new_df.head(1)["t"].values[0]
            ratio = round((float(t1)/float(t2)), 3)
            ratio_list.append(ratio)
            print(t1, t2)
            td, tb = t1, t2

    print("ratio_list:", ratio_list)
    print("--> 强渗时间:扩散时间的最大比值为：{}:1".format(max(ratio_list)))
    time_infer_end = time.time()

    print("\n --------------------- 时间统计 ---------------------")
    print(f"train1: {time_train1_end- time_train1_start} 秒")
    print(f"train2: {time_train2_end - time_train2_start} 秒")
    print(f"infer : {time_infer_end - time_infer_start} 秒")


