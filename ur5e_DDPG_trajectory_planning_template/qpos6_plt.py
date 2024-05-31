"""
处理.csv表格,并且生成相对于的图表。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
#
# matplotlib.rc("font", family='YouYuan')  # 调用中文字体，family还有其它选项
class Joint_Space_Path():
    def __init__(self, max_episode, jsp_path, est_path=None, smooth=False):
        self.jsp_path = jsp_path  # joint_path_table的存储路径
        self.max_episode = max_episode  # 单次训练任务的总回合数
        self.smooth = smooth    #是否进行平滑操作
        self.est_path = est_path # episode_sf_table的存储路径

    def moving_average(self, a, window_size):
        """
        滑动平均函数，对有噪声的曲线进行降噪处理
        :param a:需要平滑的数据集
        :param window_size: 平滑的窗口大小
        :return:平滑后的数据集
        """
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def smooth_plt(self, step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5):
        if self.smooth:
            # 平滑处理
            yplt_list_qpos0 = self.moving_average(yplt_list_qpos0, 101)
            yplt_list_qpos1 = self.moving_average(yplt_list_qpos1, 101)
            yplt_list_qpos2 = self.moving_average(yplt_list_qpos2, 101)
            yplt_list_qpos3 = self.moving_average(yplt_list_qpos3, 101)
            yplt_list_qpos4 = self.moving_average(yplt_list_qpos4, 101)
            yplt_list_qpos5 = self.moving_average(yplt_list_qpos5, 101)
            yplt_list_qpos0 = yplt_list_qpos0[:len(step_list)]
            yplt_list_qpos1 = yplt_list_qpos1[:len(step_list)]
            yplt_list_qpos2 = yplt_list_qpos2[:len(step_list)]
            yplt_list_qpos3 = yplt_list_qpos3[:len(step_list)]
            yplt_list_qpos4 = yplt_list_qpos4[:len(step_list)]
            yplt_list_qpos5 = yplt_list_qpos5[:len(step_list)]
        return step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5

    def read_csv(self, k=None, type=0):
        """
        type0: 读取DDPG训练出来的第k回合的joint space运动策略
        type1: 读取episode_sf_table.csv表格
        type2: 读取episode_sf_table_after_filter1.csv
        type3: 读取smooth_plt文件夹下的joint_path_table_smooth_{}.csv文件
        读取csv表格文件
        :param k:第k回合的文件，当type=1，不用对k赋值
        :param type: #type=0，读取joint_path_table文件。#type=1，读取episode_sf_table文件
        :return:numpy格式的列表
        """
        # 读取表格文件
        if type == 0:
            file_name = 'joint_path_table_{}.csv'.format(k)
            df = pd.read_csv(self.jsp_path+file_name, header=None)
            qpos_array = df[1:].values.astype(float)
            # matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
            step_list = qpos_array[:, 0]
            yplt_list_qpos0 = qpos_array[:, 1]
            yplt_list_qpos1 = qpos_array[:, 2]
            yplt_list_qpos2 = qpos_array[:, 3]
            yplt_list_qpos3 = qpos_array[:, 4]
            yplt_list_qpos4 = qpos_array[:, 5]
            yplt_list_qpos5 = qpos_array[:, 6]
            total_reward = qpos_array[:, 7][-2]
            step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5 = self.smooth_plt(
                step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4,
                yplt_list_qpos5)
            return step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward
        elif type == 1:
            file_name = 'episode_sf_table.csv'
            df = pd.read_csv(self.est_path+file_name, header=None)
            qpos_array = df[1:].values.astype(int)
            episode_sf_list = qpos_array[:, 0]
            first_reach_step_list = qpos_array[:, 1]
            return episode_sf_list, first_reach_step_list
        elif type == 2:
            file_name = 'episode_sf_table_after_filter1.csv'
            df = pd.read_csv('./' + file_name, header=None)
            qpos_array = df[1:].values.astype(int)
            episode_sf_list_after_filter = qpos_array[:, 0]
            first_reach_step_list_after_filter = qpos_array[:, 1]
            return episode_sf_list_after_filter, first_reach_step_list_after_filter
        else:
            file_name = 'joint_path_table_smooth_{}.csv'.format(k)
            df = pd.read_csv('./smooth_plt/' + file_name, header=None)
            qpos_array = df[1:].values.astype(float)
            # matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
            step_list = qpos_array[:, 0]
            yplt_list_qpos0 = qpos_array[:, 1]
            yplt_list_qpos1 = qpos_array[:, 2]
            yplt_list_qpos2 = qpos_array[:, 3]
            yplt_list_qpos3 = qpos_array[:, 4]
            yplt_list_qpos4 = qpos_array[:, 5]
            yplt_list_qpos5 = qpos_array[:, 6]
            total_reward = qpos_array[:, 7][-2]
            step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5 = self.smooth_plt(
                step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4,
                yplt_list_qpos5)
            return step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward

    def plt_6qpos(self, step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward, k):

        if self.smooth:
            image_name = 'joint_path_smooth_{}'.format(k)
        else:
            image_name  = 'joint_path_{}'.format(k)

        delta_time = 0.006
        _, ax = plt.subplots(3, 2, figsize=(8, 8))
        ax[0][0].plot(step_list, yplt_list_qpos0,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[0][0].set_title('shoulder link')
        ax[0][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[0][0].set_ylabel(r'''$\theta_{1}$/rad''')
        ax[0][0].legend()
        ax[1][0].plot(step_list, yplt_list_qpos1,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[1][0].set_title('upper arm link')
        ax[1][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[1][0].set_ylabel(r'''$\theta_{2}$/rad''')
        ax[1][0].legend()
        ax[2][0].plot(step_list, yplt_list_qpos2,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[2][0].set_title('forearm link')
        ax[2][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[2][0].set_ylabel(r'''$\theta_{3}$/rad''')
        ax[2][0].legend()
        ax[0][1].plot(step_list, yplt_list_qpos3,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[0][1].set_title('wrist1 link')
        ax[0][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[0][1].set_ylabel(r'''$\theta_{4}$/rad''')
        ax[0][1].legend()
        ax[1][1].plot(step_list, yplt_list_qpos4,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[1][1].set_title('wrist2 link')
        ax[1][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[1][1].set_ylabel(r'''$\theta_{5}$/rad''')
        ax[1][1].legend()
        ax[2][1].plot(step_list, yplt_list_qpos5,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[2][1].set_title('wrist3 link')
        ax[2][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[2][1].set_ylabel(r'''$\theta_{6}$/rad''')
        ax[2][1].legend()
        plt.tight_layout()

        # 保存图片
        filename = '{}.png'.format(image_name)
        plt.savefig('./joint_path_csv2png/{}'.format(filename))
        # plt.close(fig)
        plt.close("all")
        print('Saved trajectory to {}.'.format(filename))

    def plt_qposx(self, step_list, yplt_list_qpos, x, total_reward, k):
        """

        :param step_list: 第K回合的step变化
        :param yplt_list_qposx: qposx的角度变化
        :param x: 第x个关节角
        :param total_reward: 总奖励值
        :param k: 第k个回合
        :return:
        """
        title_name = ''
        image_name = ''
        if x == 1 :
            title_name = 'shoulder link'
            image_name = 'shoulder_link'
        elif x == 2:
            title_name = 'upper arm link'
            image_name = 'upper_arm_link'
        elif x == 3:
            title_name = 'forearm link'
            image_name = 'forearm_link'
        elif x == 4:
            title_name = 'wrist1 link'
            image_name = 'wrist1_link'
        elif x == 5:
            title_name = 'wrist2 link'
            image_name = 'wrist2_link'
        elif x == 6:
            title_name = 'wrist3 link'
            image_name = 'wrist3_link'
        else:
            print('the input of x is wrong')

        image_name = '{}_{}'.format(image_name, k)
        fig = plt.figure(1, figsize=(9, 6))

        plt.title(title_name, fontsize=16)
        plt.xlabel('step', fontsize=14)
        plt.ylabel(r'''$\theta_{}$/rad'''.format(x), fontsize=14)
        plt.plot(step_list, yplt_list_qpos, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward), linewidth=2, color='C0')
        plt.legend()
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        filename = '{}.png'.format(image_name)
        plt.savefig('./joint_path_csv2png/{}'.format(filename))
        plt.close("all")
        print('Saved trajectory to {}.'.format(filename))
        # plt.show()+

    def first_reach_step_output(self, episode_sf_list, first_reach_step_list, k):
        """
        获取指定回合的first_reach_step。
        :param episode_sf_list:
        :param first_reach_step_list:
        :param k:
        :return:
        """
        j = 0
        for i in range(len(episode_sf_list)):
            if episode_sf_list[i] == k:
                j = i
                break
        return first_reach_step_list[j]

    def plt_qposx_D(self, x, k, font):
        """
        绘制第k回合的第x个关节的数据集D采样方式
        :param step_list: 第K回合的step变化
        :param yplt_list_qposx: qposx的角度变化
        :param x: 第x个关节角
        :param total_reward: 总奖励值
        :param k: 第k个回合
        :return:
        """
        yplt_list_qpos = []
        step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = self.read_csv(k=k, type=0)  # joint读取表格
        episode_sf_list, first_reach_step_list = self.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
        first_reach_step = self.first_reach_step_output(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, k=k)
        title_name = ''
        image_name = ''
        if x == 1:
            title_name = 'shoulder link'
            image_name = 'shoulder_link'
            yplt_list_qpos = yplt_list_qpos0
        elif x == 2:
            title_name = 'upper arm link'
            image_name = 'upper_arm_link'
            yplt_list_qpos = yplt_list_qpos1
        elif x == 3:
            title_name = 'forearm link'
            image_name = 'forearm_link'
            yplt_list_qpos = yplt_list_qpos2
        elif x == 4:
            title_name = 'wrist1 link'
            image_name = 'wrist1_link'
            yplt_list_qpos = yplt_list_qpos3
        elif x == 5:
            title_name = 'wrist2 link'
            image_name = 'wrist2_link'
            yplt_list_qpos = yplt_list_qpos4
        elif x == 6:
            title_name = 'wrist3 link'
            image_name = 'wrist3_link'
            yplt_list_qpos = yplt_list_qpos5
        else:
            print('the input of x is wrong')

        fig= plt.figure(1, figsize=(9, 6))

        plt.title(title_name, fontsize=16, fontdict=font)
        plt.xlabel('step', fontsize=16, fontdict=font)
        plt.ylabel(r'$\theta_{}$/rad'.format(x), fontsize=16, fontdict=font)
        # 画红虚线
        plt.axvline(first_reach_step, linestyle='dashed', color='r')
        plt.axvline(first_reach_step+100, linestyle='dashed', color='r')
        plt.plot(step_list, yplt_list_qpos,label='$\mathit{N}_{epi}$' + ' = {}'.format(k) + ' , ' + '$\mathit{R}_{epi}$' + ' = {:0.0f}'.format(total_reward), linewidth=2, color='C0')
        plt.legend()
        plt.plot([first_reach_step], [yplt_list_qpos[first_reach_step]], 'o', color='r')
        plt.plot([first_reach_step+100], [yplt_list_qpos[first_reach_step+100]], 'o', color='r')
        plt.annotate('step=$\mathit{N}_{\mathit{step}\mathrm{0}}$', xy=(first_reach_step, -0.04),  # t角度和角度值
                     xytext=(50, -0.04),  # fraction, fraction
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=14
                     )
        plt.annotate('', xy=(first_reach_step, -0.02),  # t角度和角度值
                     xytext=(first_reach_step+100, -0.02),  # fraction, fraction
                     arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',facecolor='black'),
                     fontsize=14
                    )
        # 绘制文字”D“
        D_text_font = {'style':'oblique','weight':'bold'}
        plt.text(first_reach_step + 50, -0.019, 'D', fontdict=D_text_font, fontsize=14)
        # 保存图片
        image_name = 'Ddata_{}_{}'.format(image_name, k)
        filename = '{}.png'.format(image_name)
        save_path = './joint_path_csv2png/{}'.format(filename)
        plt.savefig(save_path)
        plt.close("all")
        print('Saved {}.png to {}.'.format(image_name, save_path))
        # plt.show()

    def qpos_var_after_FirstReach(self, yplt_list_qpos, first_reach_step):
        """测试用代码，可忽略"""
        yplt_list_qpos = yplt_list_qpos[first_reach_step:]
        var = np.var(yplt_list_qpos)
        return var

    def batch_size_qpos_var(self, episode_sf_list, first_reach_step_list, size=1, x=1, random=True, index_begin=0, batch_ansys=True, k=None, after_soomth=False):
        """
        从成功的回合中，随机取size个D数据集。并将这些数据组合在一起求方差
        :param episode_sf_list:轨迹规划成功的episode（回合数）的集合
        :param first_reach_step_list:成功的episode中第一次到达目标点的step（时间步）的集合
        :param size:参与求方差的episodd_sf个数
        :param x:第x个qpos的方差
        :param random:是否随机取值
        :param index_begin:在不是随机取值时，最开始取值的序号。不指定的话就是从0开始。
        :param k:是否选取第k个训练回合
        :return:方差
        """
        index_list = []
        if k != None:
            batch_ansys = False

        if batch_ansys:
            if random:
                index_list = np.random.randint(0, len(episode_sf_list), (size,)) #有时候会重复选择相同的回合
            else:
                if index_begin+size >= len(episode_sf_list):
                    index_list = np.arange(index_begin, len(episode_sf_list))
                else:
                    index_list = np.arange(index_begin, index_begin+size)

        else:
            for j in range(len(episode_sf_list)):
                if episode_sf_list[j] == k:
                    index_list.append(j)
                    break
        # print(len(index_list))
        qpos_after_first_reach_list = []
        for i in range(len(index_list)):
            if after_soomth:
                step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = self.read_csv(k=episode_sf_list[index_list[i]], type=3)
            else:
                step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = self.read_csv(k=episode_sf_list[index_list[i]], type=0)
            if x == 1:
                qpos_after_first_reach_list.append(yplt_list_qpos0[first_reach_step_list[index_list[i]]:])
            elif x == 2:
                qpos_after_first_reach_list.append(yplt_list_qpos1[first_reach_step_list[index_list[i]]:])
            elif x == 3:
                qpos_after_first_reach_list.append(yplt_list_qpos2[first_reach_step_list[index_list[i]]:])
            elif x == 4:
                qpos_after_first_reach_list.append(yplt_list_qpos3[first_reach_step_list[index_list[i]]:])
            elif x == 5:
                qpos_after_first_reach_list.append(yplt_list_qpos4[first_reach_step_list[index_list[i]]:])
            elif x == 6:
                qpos_after_first_reach_list.append(yplt_list_qpos5[first_reach_step_list[index_list[i]]:])
        # print(qpos_after_first_reach_list)
        total_var = 0
        # print(len(qpos_after_first_reach_list))
        for i in range(len(qpos_after_first_reach_list)):
            var = np.var(qpos_after_first_reach_list[i])
            total_var += var
        average_var = total_var/len(qpos_after_first_reach_list)
        return average_var
        # return var

    def qposDvar_plt(self, episode_sf_list=None, first_reach_step_list=None, size=1, x=1, after_filter=False):
        """
        对机械臂某个关节角在所有成功完成的回合，绘制关节角的数据集D在各个成功回合的方差分布。
        :param episode_sf_list:成功的回合数
        :param first_reach_step_list:成功的回合中第一次到达目标点的step
        :param size:默认值不改动
        :param x:第几个关节角
        :param after_filter:是否过滤
        :return:
        """
        i_plt_list = []
        qposDvar_plt_list = []
        for i in range(len(episode_sf_list)):
            qposDvar = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list,
                                               first_reach_step_list=first_reach_step_list, index_begin=i, size=size, x=x,
                                               random=False)
            i_plt_list.append(i)
            qposDvar_plt_list.append(qposDvar)

        qposDvar_plt_list_smooth = self.moving_average(qposDvar_plt_list, 51)
        if after_filter:
            image_name = 'qpos' + '{}'.format(x) + 'Dvar'+'_after_filter'
        else:
            image_name = 'qpos'+'{}'.format(x)+'Dvar'
        fig = plt.figure(1, figsize=(9, 6))
        plt.title('qposDvar' + ' change ', fontsize=16)
        plt.xlabel('times', fontsize=14)
        plt.ylabel('qposDvar', fontsize=14)
        plt.plot(i_plt_list, qposDvar_plt_list, linewidth=2, color='C0')
        plt.plot(i_plt_list, qposDvar_plt_list_smooth, linewidth=2, color='r')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        filename = '{}.png'.format(image_name)
        plt.savefig('./SD_csv/{}'.format(filename))
        # print('Saved trajectory to {}.'.format(filename))
        # plt.show()

    def average_qposDvar_table(self, episode_sf_list=None, first_reach_step_list=None, after_filter=False):
        """
        对episode_sf_table中所有的成功回合求方差的平均值，然后生成表格average_qposDvar_table_after或者average_qposDvar_table_before
        :param episode_sf_list:
        :param first_reach_step_list:
        :param after_filter:
        :return:
        """
        qposDvar_plt_list = [] # 每个角度根据eipsode_sf_list求得的角度方差
        for i in range(6):
            qposDvar = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, index_begin=0, size=len(episode_sf_list), x=i+1, random=False) #顺序选择
            # qposDvar = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, size=len(episode_sf_list), x=i + 1, random=True) #随机选择
            qposDvar_plt_list.append(qposDvar)
        print(qposDvar_plt_list)

        if after_filter:
            filename = 'average_qposDvar_table_after'
        else:
            filename = 'average_qposDvar_table_before'
        qposDvar_plt_list = np.array(qposDvar_plt_list)
        qposDvar_plt_list = qposDvar_plt_list.reshape(1, 6)
        df = pd.DataFrame(qposDvar_plt_list, columns=['shoulder link', 'upper arm link', 'forearm link', 'wrist1 link', 'wrist2 link', 'wrist3 link'])
        df.to_csv('./'+'{}'.format(filename)+'.csv', index=False)

    def qposDvar_filter(self, episode_sf_list=None, first_reach_step_list=None, filter_k=None):
        """
         读取qposDvar_table_before存储的数据，并且进行处理，来对方差过大的角度进行过滤，只保留方差小的成功回合的数据。
        :return:
        """
        #读取qposDvar_table_before表格用来作为比较的基准
        file_name = 'average_qposDvar_table_before.csv'
        df = pd.read_csv('./' + file_name, header=None)
        qpos_array = df[1:].values.astype(float)
        average_of_qposDvar = qpos_array.reshape(6,) #降维度
        #准备开始对episode_sf_list数据过滤
        episode_sf_list_after_filter = [] #过滤后的成功回合
        for i in range(len(episode_sf_list)):
            qposvar_list = []
            for j in range(6):
                x = j + 1
                qposvar = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, index_begin=i, size=1, x=x, random=False)
                qposvar_list.append(qposvar)

            # if qposvar_list[0] <= filter_k[0] * average_of_qposDvar[0]:
            #     if qposvar_list[1] <= filter_k[1] * average_of_qposDvar[1]:
            #         if qposvar_list[2] <= filter_k[2] * average_of_qposDvar[2]:
            #             if qposvar_list[3] <= filter_k[3] * average_of_qposDvar[3]:
            #                 if qposvar_list[4] <= filter_k[4] * average_of_qposDvar[4]:
            #                     if qposvar_list[5] <= filter_k[5] * average_of_qposDvar[5]:
            #                         episode_sf_list_after_filter.append(np.concatenate(([episode_sf_list[i]], [first_reach_step_list[i]]), axis=0)) #像之前存储episode_sf_table的格式存储过滤后的

            if qposvar_list[0] <= filter_k[0]:
                if qposvar_list[1] <= filter_k[1]:
                    if qposvar_list[2] <= filter_k[2]:
                        if qposvar_list[3] <= filter_k[3]:
                            if qposvar_list[4] <= filter_k[4]:
                                if qposvar_list[5] <= filter_k[5]:
                                    episode_sf_list_after_filter.append(
                                        np.concatenate(([episode_sf_list[i]], [first_reach_step_list[i]]),
                                                       axis=0))  # 像之前存储episode_sf_table的格式存储过滤后的
        episode_sf_list_after_filter = np.array(episode_sf_list_after_filter)
        df = pd.DataFrame(episode_sf_list_after_filter, columns=['episode', 'first_reach_step'])
        df.to_csv('./episode_sf_table_after_filter.csv', index=False)

if __name__ == '__main__':
    jsp_path = '../../ur5e_DDPG_article2_0_01_20000csv/joint_path_csv/'
    est_path = './'
    JSP = Joint_Space_Path(max_episode=20000, jsp_path=jsp_path, est_path=est_path,smooth=False)  # smooth选项决定True平滑还是不平滑False
    # 读取表格
    # step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = JSP.read_csv(k=10672, type=0)
    # # 画出六个角度的随step变化的图并保存
    # JSP.plt_6qpos(step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4,yplt_list_qpos5, total_reward, k=10672)
    # # 画图一个角度随step变化的图并保存
    # JSP.plt_qposx(step_list=step_list, yplt_list_qpos=yplt_list_qpos1, x=2, k=370, total_reward=total_reward)
    # 单个joint的角度稳定时的方差
    # var = JSP.qpos_var_after_FirstReach(yplt_list_qpos=yplt_list_qpos1, first_reach_step=73)
    # print(var)

    # episode_sf_list, first_reach_step_list = JSP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    # 特定回合采样输出方差
    # qpos1var = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, x=2, k=1383) # 用这种方式输入可以求某个特定的episode的D集合方差
    # 随机采样输出方差
    # qpos1var = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, size=1000, x=6, random=True)  #
    # 顺序采样输出方差
    # qpos1var = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, index_begin=0, size=7995, x=1, random=False)
    # print(qpos1var)

    # JSP.qposDvar_plt(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, size=1, x=3)

    # JSP.qposDvar_table_before(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list)
    # JSP.qposDvar_filter(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list)
    # episode_sf_list, first_reach_step_list = JSP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    # JSP.qposDvar_plt(episode_sf_list=episode_sf_list,first_reach_step_list=first_reach_step_list, x=3, after_filter=False)  # 可以产生某个角度过滤后的方差分布图
    # episode_sf_list_after_filter, first_reach_step_list_after_filter = JSP.read_csv(type=2)  # 读取过滤后的表格episode_sf_table_after_filter
    # JSP.qposDvar_plt(episode_sf_list=episode_sf_list_after_filter, first_reach_step_list=first_reach_step_list_after_filter, x=3, after_filter=True)  # 可以产生某个角度过滤后的方差分布图

    # 对原来的episode_sf_list进行过滤
    episode_sf_list, first_reach_step_list = JSP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    # JSP.average_qposDvar_table(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, after_filter=False)    #生成过滤前的所有成功回合的关节角数据集D的方差的均值，作为过滤标准
    filter_k2 = [0.00001, 0.00001, 0.00001, 0.0001, 0.0001, 0.0001]
    filter_k1 = [0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]
    JSP.qposDvar_filter(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, filter_k=filter_k2) #过滤处理，得到过滤后的episode_sf_list，过滤掉了方差很大的一些数据
    episode_sf_list_after_filter, first_reach_step_list_after_filter = JSP.read_csv(type=2) #读取过滤后的表格episode_sf_table_after_filter
    # JSP.qposDvar_plt(episode_sf_list=episode_sf_list_after_filter,first_reach_step_list=first_reach_step_list_after_filter, x=1, after_filter=True) #可以产生某个角度过滤后的方差分布图
    # JSP.average_qposDvar_table(episode_sf_list=episode_sf_list_after_filter, first_reach_step_list=first_reach_step_list_after_filter, after_filter=True) #求过滤后的的所有成功回合的关节角数据集D的方差的均值，作为稳定性判断依据

    # episode_sf_list, first_reach_step_list = JSP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
    # JSP.qposDvar_plt(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, x=6, after_filter=False)  # 可以产生某个角度过滤后的方差分布图
    # 求K值
    # file_name = 'average_qposDvar_table_after.csv'
    # df = pd.read_csv('./' + file_name, header=None)
    # qpos_array = df[1:].values.astype(float)
    # average_of_qposDvar_after = qpos_array.reshape(6, )  # 降维度
    # K = 0.25*average_of_qposDvar_after[0]+0.25*average_of_qposDvar_after[1]+0.25*average_of_qposDvar_after[2]+0.12*average_of_qposDvar_after[3]+0.12*average_of_qposDvar_after[4]+0.05*average_of_qposDvar_after[5]
    # print(average_of_qposDvar_after)
    # print(K)

