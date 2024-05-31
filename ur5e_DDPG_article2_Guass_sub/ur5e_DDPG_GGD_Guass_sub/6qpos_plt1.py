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
    def __init__(self, max_episode, jsp_path, smooth=False):
        self.jsp_path = jsp_path  # jnd_path_table的存储路径
        self.max_episode = max_episode  # 单次训练任务的总回合数
        self.smooth = smooth

    def moving_average(self, a, window_size): # 滑动平均函数，对有噪声的曲线进行降噪处理
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def read_csv(self, k):
        # 读取表格文件
        self.file_name = 'joint_path_table_{}.csv'.format(k)
        df = pd.read_csv(self.jsp_path+self.file_name, header=None)
        qpos_array = df[1:].values.astype(float)
        # matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
        xplt_list = qpos_array[:, 0]
        yplt_list_qpos0 = qpos_array[:, 1]
        yplt_list_qpos1 = qpos_array[:, 2]
        yplt_list_qpos2 = qpos_array[:, 3]
        yplt_list_qpos3 = qpos_array[:, 4]
        yplt_list_qpos4 = qpos_array[:, 5]
        yplt_list_qpos5 = qpos_array[:, 6]
        total_reward = qpos_array[:, 7][-2]
        return xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward

    def smooth_plt(self, xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5):
        if self.smooth:
            # 平滑处理
            yplt_list_qpos0 = self.moving_average(yplt_list_qpos0, 101)
            yplt_list_qpos1 = self.moving_average(yplt_list_qpos1, 101)
            yplt_list_qpos2 = self.moving_average(yplt_list_qpos2, 101)
            yplt_list_qpos3 = self.moving_average(yplt_list_qpos3, 101)
            yplt_list_qpos4 = self.moving_average(yplt_list_qpos4, 101)
            yplt_list_qpos5 = self.moving_average(yplt_list_qpos5, 101)
            yplt_list_qpos0 = yplt_list_qpos0[:len(xplt_list)]
            yplt_list_qpos1 = yplt_list_qpos1[:len(xplt_list)]
            yplt_list_qpos2 = yplt_list_qpos2[:len(xplt_list)]
            yplt_list_qpos3 = yplt_list_qpos3[:len(xplt_list)]
            yplt_list_qpos4 = yplt_list_qpos4[:len(xplt_list)]
            yplt_list_qpos5 = yplt_list_qpos5[:len(xplt_list)]
        return xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5

    def plt_6qpos(self, xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward, k):

        if self.smooth:
            image_name = 'joint_path_smooth_{}'.format(k)
        else:
            image_name  = 'joint_path_{}'.format(k)

        delta_time = 0.006
        _, ax = plt.subplots(3, 2, figsize=(8, 8))
        ax[0][0].plot(xplt_list, yplt_list_qpos0,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[0][0].set_title('shoulder link')
        ax[0][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[0][0].set_ylabel(r'''$\theta_{1}$/rad''')
        ax[0][0].legend()
        ax[1][0].plot(xplt_list, yplt_list_qpos1,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[1][0].set_title('upper arm link')
        ax[1][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[1][0].set_ylabel(r'''$\theta_{2}$/rad''')
        ax[1][0].legend()
        ax[2][0].plot(xplt_list, yplt_list_qpos2,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[2][0].set_title('forearm link')
        ax[2][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[2][0].set_ylabel(r'''$\theta_{3}$/rad''')
        ax[2][0].legend()
        ax[0][1].plot(xplt_list, yplt_list_qpos3,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[0][1].set_title('wrist1 link')
        ax[0][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[0][1].set_ylabel(r'''$\theta_{4}$/rad''')
        ax[0][1].legend()
        ax[1][1].plot(xplt_list, yplt_list_qpos4,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[1][1].set_title('wrist2 link')
        ax[1][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[1][1].set_ylabel(r'''$\theta_{5}$/rad''')
        ax[1][1].legend()
        ax[2][1].plot(xplt_list, yplt_list_qpos5,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
        ax[2][1].set_title('wrist3 link')
        ax[2][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
        ax[2][1].set_ylabel(r'''$\theta_{6}$/rad''')
        ax[2][1].legend()
        plt.tight_layout()
        # 保存图片
        filename = '{}.png'.format(image_name)
        plt.savefig('./joint_path_csv2png/{}'.format(filename))
        print('Saved trajectory to {}.'.format(filename))

if __name__ == '__main__':
    jsp_path = '../../ur5e_DDPG_article2_Guass_sub_20000csv/joint_path_csv/'
    JSP = Joint_Space_Path(max_episode=20000, jsp_path=jsp_path, smooth=False)  #smooth选项决定True平滑还是不平滑False
    xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = JSP.read_csv(k=4486)
    xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5 = JSP.smooth_plt(xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5)
    JSP.plt_6qpos(xplt_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward, k=4486)