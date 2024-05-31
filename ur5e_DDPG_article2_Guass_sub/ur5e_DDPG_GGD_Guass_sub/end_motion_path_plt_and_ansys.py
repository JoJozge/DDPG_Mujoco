"""
利用DDPG算法进行轨迹规划任务，判断完成任务好坏的指标。需要完成所有训练之后才能够使用该部分。
可以利用该函数进行机械臂末端点在三维空间的运动轨迹绘图。
可以利用该函数分析抖动指标。
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib
import os

matplotlib.rc("font", family='YouYuan')  # 调用中文字体，family还有其它选项
plot_3D = True
plot_step = 1

N_pf = 0  #可能完成轨迹规划任务的次数N_pf
N_sf = 0  #真实完成轨迹规划任务的次数
P_f = 0   #轨迹规划任务可能完成时，不发生抖动的概率
P_m = 0   #轨迹规划任务中成功完成的回合占总训练回合的概率

for k in range(5000):   #2500为训练的回合数，应该随实际训练的回合数修改

    # 创建一个Figure对象和一个3D坐标轴对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df = pd.read_csv('./end_motion_path_csv/end_motion_table_{}.csv'.format(k+1), header=None)
    numpy_array = df[1:].values.astype(float)
    # print(numpy_array)
    x = numpy_array[:, 0]
    y = numpy_array[:, 1]
    z = numpy_array[:, 2]
    total_reward = numpy_array[:, 3][len(x)-2]      # 只需要total_reward在step为t-1时的值，
    print(total_reward)
    # print(x)

    already_reach_target = False    #是否已经到达目标
    first_reach_step = 0    #第一次到达的回合数
    shake_count = 0     #在已经达到目标位置之后又偏离出目标位置称为抖动
    shake = False       #是否判定为抖动

    print('len(x):{}'.format(len(x)))
    if total_reward >= 950:
        N_pf += 1
        for i in range(len(x)):
            if abs(x[i] - 0.5) <= 0.02 and abs(y[i] - 0.5) <= 0.02 and abs(z[i] - 0.5) <= 0.02:
                already_reach_target = True

            if not already_reach_target:
                first_reach_step = i+1

            if already_reach_target:
                if abs(x[i] - 0.5) > 0.02 or abs(y[i] - 0.5) > 0.02 or abs(z[i] - 0.5) > 0.02:
                    shake_count += 1

    if shake_count >= 50:
        shake = True

    if not shake and total_reward >= 950:
        N_sf += 1

    print('{}----shake_count:{}, shake:{}, first_reach_step:{} total_reward:{} N_pf:{} '.format(k+1, shake_count, shake, first_reach_step, total_reward, N_pf))

    if plot_3D and k % plot_step == 0:
        # 定义球体参数
        center_x_sphere = 0.5
        center_y_sphere = 0.5
        center_z_sphere = 0.5
        radius_sphere = 0.02
        theta_sphere = np.linspace(0, 2 * np.pi, 50)
        phi_sphere = np.linspace(0, np.pi, 50)

        # 创建参数范围
        Theta_sphere, Phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

        # 计算球体表面上的点的坐标
        X_sphere = center_x_sphere + radius_sphere * np.outer(np.cos(Theta_sphere), np.sin(Phi_sphere))
        Y_sphere = center_y_sphere + radius_sphere * np.outer(np.sin(Theta_sphere), np.sin(Phi_sphere))
        Z_sphere = center_z_sphere + radius_sphere * np.outer(np.ones(np.size(Theta_sphere)), np.cos(Phi_sphere))

        # 绘制圆柱体表面
        # ax.plot_surface(X1, Y1, Z1, rstride=5, cstride=5, color='lightblue', alpha=0.6)
        # ax.plot_surface(X2, Y2, Z2, rstride=5, cstride=5, color='lightblue', alpha=0.6)
        # ax.plot_surface(X3, Y3, Z3, rstride=5, cstride=5, color='lightblue', alpha=0.6)
        plt.ion()
        # 绘制球体表面
        ax.plot_surface(X_sphere, Y_sphere, Z_sphere, color='red', alpha=0.6)
        ax.plot(x,y,z)
        # 设置刻度为固定值
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_zticks(np.linspace(0, 0.6, 6))

        # 设置标题和坐标轴标签
        # ax.set_title("")
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        # 显示图形
        # plt.show()
        filename = 'end_motion_path_{}_{:0.0f}.png'.format(k+1, total_reward)
        plt.savefig('./end_motion_path_csv2png3D/{}'.format(filename))
        plt.pause(1)

        # plt.ioff()
        # plt.close(fig)
        # plt.close("all")

P_m = N_sf/2500
P_f = N_sf/N_pf
print('P_m:{}, P_f:{}'.format(P_m, P_f))