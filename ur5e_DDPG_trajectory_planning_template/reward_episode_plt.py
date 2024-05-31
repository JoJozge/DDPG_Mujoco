"""
处理episode_reward.csv表格,并且生成相对于的图表。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 字体初始化
plt.rc('font', family='Times New Roman', size=18) # 全局设置Times New Roman字体
# 将公式的字体全部设置为常规字体，对本例而言就是Times New Roman字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

# 滑动平均函数，对有噪声的曲线进行降噪处理
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

k = 20000
# 读取表格文件
df = pd.read_csv('../../ur5e_DDPG_article2_0_01_20000csv/reward_episode_csv/reward_episode_table_{}.csv'.format(k), header=None)
qpos_array = df[1:].values.astype(float)

image_name = 'reward'

episode_list = qpos_array[:, 0]
reward_list = qpos_array[:, 1]
i = episode_list[len(episode_list)-1]
total_reward = reward_list[len(episode_list)-1]

fig = plt.figure(1, figsize=(9, 6))
plt.title('rewards vary with episodes', font1)
plt.xlabel('episodes', font1)
plt.ylabel('rewards', font1)
# plt.xlabel('回合数', fontsize=20)
# plt.ylabel('奖励值', fontsize=20)
plt.plot(episode_list, reward_list,linewidth=2, color='C0')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
filename = 'step_{}_{}_{:0.0f}.png'.format(image_name, i, total_reward)
plt.savefig('./reward_episode_csv2png/{}'.format(filename))
# print('Saved trajectory to {}.'.format(filename))
plt.show()
