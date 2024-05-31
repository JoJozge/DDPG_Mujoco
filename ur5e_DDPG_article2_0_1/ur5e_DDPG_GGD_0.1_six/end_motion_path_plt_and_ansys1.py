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

class End_Motion_Path():

    def __init__(self, max_episode, emp_path, est_path):
        self.N_pf = 0   #可能完成轨迹规划任务的次数N_pf
        self.N_sf = 0   #真实完成轨迹规划任务的次数
        self.P_f = 0    #轨迹规划任务可能完成时，不发生抖动的概率
        self.P_m = 0    #轨迹规划任务中成功完成的回合占总训练回合的概率
        self.expisode_pf = 0    #首次可能完成的回合序号
        self.expisode_sf0 = 0    #首次真实完成的回合序号
        self.expisode_sf = 0
        self.expisode_sf_list = []
        self.max_episode = max_episode  #单次训练任务的总回合数
        self.emp_path = emp_path    #end_motion_table的存储路径
        self.est_path = est_path  # episode_sf_table的存储路径
        self.end_point = {'x': 0.5, 'y': 0.5, 'z': 0.5}

    def read_csv(self, k=None, type=0):
        """
        读取第k回合的end_motion_table（记录的末端点随step（时间步）移动的空间坐标）
        :param k: 第k回合
        :return: 空间坐标x的列表，空间坐标y的列表，空间坐标z的列表，该回合总奖励值
        """
        if type == 0:
            df = pd.read_csv(self.emp_path+'end_motion_table_{}.csv'.format(k), header=None)
            numpy_array = df[1:].values.astype(float)
            x_list = numpy_array[:, 0]
            y_list = numpy_array[:, 1]
            z_list = numpy_array[:, 2]
            total_reward = numpy_array[:, 3][len(x_list) - 2]  # 只需要total_reward在step为t-1时的值，
            return x_list, y_list, z_list, total_reward
        else:
            file_name = 'episode_sf_table.csv'
            df = pd.read_csv(self.est_path + file_name, header=None)
            qpos_array = df[1:].values.astype(int)
            episode_sf_list = qpos_array[:, 0]
            first_reach_step_list = qpos_array[:, 1]
            return episode_sf_list, first_reach_step_list

    def EMP_Ansys(self, x_list=None, y_list=None, z_list=None, total_reward=None, k=None):
        """
        对第k回合的end_motion_path数据进行分析，并按照论文中的判据条件进行输出。
        :param x_list:空间坐标x的列表
        :param y_list:空间坐标y的列表
        :param z_list:空间坐标z的列表
        :param total_reward:第k回合的总奖励值
        :param k:第k回合
        :return:无返回值
        """
        already_reach_target = False  # 是否已经到达目标
        first_reach_step = 0  # 第一次到达的回合数
        shake_count = 0  # 在已经达到目标位置之后又偏离出目标位置称为抖动
        shake = False  # 是否判定为抖动

        if total_reward >= 950:
            self.N_pf += 1
            if self.N_pf <= 1 and k!=None:
                self.expisode_pf = k
            for i in range(len(x_list)):
                if abs(x_list[i] - 0.5) <= 0.02 and abs(y_list[i] - 0.5) <= 0.02 and abs(z_list[i] - 0.5) <= 0.02:
                    already_reach_target = True

                if not already_reach_target:
                    first_reach_step = i + 1
                if already_reach_target:
                    if abs(x_list[i] - 0.5) > 0.02 or abs(y_list[i] - 0.5) > 0.02 or abs(z_list[i] - 0.5) > 0.02:
                        shake_count += 1

        if shake_count >= 1:
            shake = True

        if not shake and total_reward >= 950:
            self.N_sf += 1
            self.expisode_sf = k
            self.expisode_sf_list.append(np.concatenate(([self.expisode_sf], [first_reach_step]),axis=0))
            if self.N_sf <= 1 and k!=None:
                self.expisode_sf0 = k

        print('{}----shake_count:{}, shake:{}, first_reach_step:{} total_reward:{:.2f} N_pf:{} '.format(k, shake_count,
                                                                                                    shake,
                                                                                                    first_reach_step,
                                                                                                    total_reward, self.N_pf))

    def plt_output(self, x_list, y_list, z_list, plot_3D = False, plot_step = 1, barrier=None, k=None, quiet=False):
        """
        保存第k回合的end_motion_path的3D图
        :param x_list: 空间坐标x的列表
        :param y_list: 空间坐标y的列表
        :param z_list: 空间坐标z的列表
        :param plot_3D: 是否画图
        :param plot_step: 画图间隔
        :param barrier:
        :param k:第k回合
        :return:None
        """
        # 创建一个Figure对象和一个3D坐标轴对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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

            # if barrier != None:
                # ax.plot_surface(X1, Y1, Z1, rstride=5, cstride=5, color='lightblue', alpha=0.6)
                # ax.plot_surface(X2, Y2, Z2, rstride=5, cstride=5, color='lightblue', alpha=0.6)
                # ax.plot_surface(X3, Y3, Z3, rstride=5, cstride=5, color='lightblue', alpha=0.6)
                # plt.ion()
            # 绘制球体表面
            ax.plot_surface(X_sphere, Y_sphere, Z_sphere, color='red', alpha=0.6)
            ax.plot(x_list, y_list, z_list)
            # 设置刻度为固定值
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_zticks(np.linspace(0, 0.6, 6))

            # 设置标题和坐标轴标签
            # ax.set_title("End motion path in 3D space")
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            # 显示图形
            # plt.show()
            filename = 'end_motion_path_{}.png'.format(k)
            plt.savefig('./end_motion_path_csv2png3D/{}'.format(filename))

            if quiet != False:
                print('''A 3D picture is successful save in directory 'end_motion_path_csv2png3D',episode={}'''.format(k))
            # plt.pause(1)

            # plt.ioff()
            # plt.close(fig)
            # plt.close("all")

    def result_output(self):
        """
        轨迹规划效果判据的结果输出
        :return: None
        """
        if self.N_pf != 0:
            P_m = self.N_sf / self.max_episode
            P_f = self.N_sf / self.N_pf
            print('-----------------------result-----------------------')
            print('expisode_pf:{}'.format(self.expisode_pf))
            print('expisode_sf0:{}'.format(self.expisode_sf0))
            print('P_m:{}, P_f:{}'.format(P_m, P_f))
        else:
            P_m = self.N_sf / self.max_episode
            P_f = 0
            print('-----------------------result-----------------------')
            print('expisode_pf:{}'.format(self.expisode_pf))
            print('expisode_sf0:{}'.format(self.expisode_sf0))
            print('P_m:{}, P_f:{}'.format(P_m, P_f))

    def batch_size_endpoint_var(self, episode_sf_list, first_reach_step_list, size=1, random=True, batch_ansys=True,
                            k=None):
        """
        从成功的回合中，随机取size个数据集。并将这些数据组合在一起求方差
        :param episode_sf_list:轨迹规划成功的episode（回合数）的集合
        :param first_reach_step_list:成功的episode中第一次到达目标点的step（时间步）的集合
        :param size:参与求方差的episodd_sf个数
        :param random:是否随机取值
        :param k:是否选取第k个训练回合
        :return:第一次到达目标点之后机械臂末端点距离的方差
        """

        index_list = []
        if k != None:
            batch_ansys = False

        if batch_ansys:
            if random:
                index_list = np.random.randint(0, len(episode_sf_list), (size,))
            else:
                index_list = np.arange(0, size)
        else:
            for j in range(len(episode_sf_list)):
                if episode_sf_list[j] == k:
                    index_list.append(j)
                    break

        end_point_reach_list = []
        for i in range(len(index_list)):
            x_list, y_list, z_list, total_reward = self.read_csv(k=episode_sf_list[index_list[i]], type=0)
            # print(episode_sf_list[index_list[-1]])
            Dist_after_first_reach_list = []    #第一次到达目标点之后的机械臂末端点到目标点距离的数据集
            for j in range (len(x_list[first_reach_step_list[index_list[i]]:])):
                x = x_list[first_reach_step_list[index_list[i]]:][j]
                y = y_list[first_reach_step_list[index_list[i]]:][j]
                z = z_list[first_reach_step_list[index_list[i]]:][j]

                dist = np.array([x - self.end_point['x'], y - self.end_point['y'], z - self.end_point['z']])
                dist = np.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2)
                Dist_after_first_reach_list.append(dist)
            end_point_reach_list.append(Dist_after_first_reach_list)

        total_var = 0
        for i in range(len(end_point_reach_list)):
            var = np.var(end_point_reach_list[i])
            total_var += var
        average_var = total_var / len(end_point_reach_list)
        return average_var

class SD_csv_plt():

    def __init__(self, SD_path):
        self.SD_path = SD_path  # end_motion_table的存储路径
    def read_csv(self):
        """
        读取SD_table
        :param k: 第k回合
        :return: 回合数的列表，G_sd的列表
        """
        df = pd.read_csv(self.SD_path + 'SD_table.csv', header=None)
        numpy_array = df[1:].values.astype(float)
        episode_list = numpy_array[:, 0]
        G_sd_list = numpy_array[:, 1]  # 衰减高斯噪声标准差
        return episode_list, G_sd_list
    def SD_plt(self, episode_list, G_sd_list):
        image_name = 'G_sd'
        fig = plt.figure(1, figsize=(9, 6))
        plt.title(r'$\mathrm{G}_{sd}$' + ' vary with episodes', fontsize=16)
        plt.xlabel('episodes', fontsize=14)
        plt.ylabel(r'$\mathrm{G}_{sd}$', fontsize=14)
        plt.plot(episode_list, G_sd_list, linewidth=2, color='C0')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        filename = '{}.png'.format(image_name)
        plt.savefig('./SD_csv/{}'.format(filename))
        # print('Saved trajectory to {}.'.format(filename))
        # plt.show()


if __name__ == '__main__':
    emp_path = '../../ur5e_DDPG_article2_0_1_20000csv/end_motion_path_csv/'
    est_path = './'
    EMP = End_Motion_Path(max_episode=20000, emp_path=emp_path, est_path=est_path)
    #
    for i in range(EMP.max_episode):
        x_list, y_list, z_list, total_reward= EMP.read_csv(i+1)
        EMP.EMP_Ansys(x_list=x_list, y_list=y_list, z_list=z_list, total_reward=total_reward, k=i+1)
    EMP.result_output()
    #
    # expisode_sf_table = np.array(EMP.expisode_sf_list)
    # df = pd.DataFrame(expisode_sf_table, columns=['episode', 'first_reach_step'])
    # df.to_csv('./episode_sf_table.csv', index=False)

    # 获取衰减正态噪声随回合衰减的变化大小
    # SD_path = './SD_csv/'
    # SD_plt = SD_csv_plt(SD_path=SD_path)
    # episode_list, G_sd_list = SD_plt.read_csv()
    # SD_plt.SD_plt(episode_list[:1500], G_sd_list[:1500])

    # 末端点方差
    # episode_sf_list, first_reach_step_list = EMP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    # endpoint_var = EMP.batch_size_endpoint_var(episode_sf_list=episode_sf_list,first_reach_step_list=first_reach_step_list, size=len(episode_sf_list),random=True)
    # print(endpoint_var)
    # episode_sf_list, first_reach_step_list = EMP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    #
    # endpoint_var = EMP.batch_size_endpoint_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, size=len(episode_sf_list), random=True)
    # print(endpoint_var)
    # 画单独回合的3D图
    # x_list, y_list, z_list, total_reward = EMP.read_csv(5304)
    # EMP.EMP_Ansys(x_list=x_list, y_list=y_list, z_list=z_list, total_reward=total_reward, k=5304)
    # EMP.plt_output(x_list=x_list, y_list=y_list, z_list=z_list, plot_3D=True, k=5304)
