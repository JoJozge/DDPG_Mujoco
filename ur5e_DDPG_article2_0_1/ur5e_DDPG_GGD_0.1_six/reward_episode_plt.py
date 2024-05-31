"""
处理episode_reward.csv表格,并且生成相对于的图表。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 滑动平均函数，对有噪声的曲线进行降噪处理
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

k = 4
# 读取表格文件
df = pd.read_csv('./reward_episode_csv/reward_episode_table_{}.csv'.format(k), header=None)
qpos_array = df[1:].values.astype(float)

matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
image_name = 'reward'

episode_list = qpos_array[:, 0]
reward_list = qpos_array[:, 1]
i = episode_list[len(episode_list)-1]
print(i)
total_reward = reward_list[len(episode_list)-1]
print(total_reward)

fig = plt.figure(1, figsize=(9, 6))
plt.title('rewards vary with episodes',fontsize=16)
plt.xlabel('episodes',fontsize=14)
plt.ylabel('rewards',fontsize=14)
# plt.xlabel('回合数', fontsize=20)
# plt.ylabel('奖励值', fontsize=20)
plt.plot(episode_list, reward_list,linewidth=2, color='C0')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
filename = 'step_{}_{}_{:0.0f}.png'.format(image_name, i, total_reward)
plt.savefig('./reward_episode_csv2png/{}'.format(filename))
# print('Saved trajectory to {}.'.format(filename))
plt.show()
