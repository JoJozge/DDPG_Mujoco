import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.gridspec as gridspec

# 字体初始化
plt.rc('font', family='Times New Roman', size=10) # 全局设置Times New Roman字体
# 将公式的字体全部设置为常规字体，对本例而言就是Times New Roman字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
# 数据读取并且转化为绘图格式
file1_path = '../../ur5e_DDPG_article2_0_01_20000csv/reward_episode_csv/'
file2_path = '../../ur5e_DDPG_article2_0_001_20000csv/reward_episode_csv/'
file3_path = '../../ur5e_DDPG_article2_0_1_20000csv/reward_episode_csv/'
file4_path = '../../ur5e_DDPG_article2_0_0001_20000csv/reward_episode_csv/'
file5_path = '../../ur5e_DDPG_article2_Guass_sub1_20000csv/reward_episode_csv/'

file_name = 'reward_episode_table_{}.csv'.format(20000)

df1 = pd.read_csv(file1_path + file_name, header=None)
df2 = pd.read_csv(file2_path + file_name, header=None)
df3 = pd.read_csv(file3_path + file_name, header=None)
df4 = pd.read_csv(file4_path + file_name, header=None)
df5 = pd.read_csv(file5_path + file_name, header=None)

plt1 = df1[1:].values.astype(float)
plt2 = df2[1:].values.astype(float)
plt3 = df3[1:].values.astype(float)
plt4 = df4[1:].values.astype(float)
plt5 = df5[1:].values.astype(float)
x_plt_list = plt1[:, 0]
y_plt_list1 = plt1[:, 1]
y_plt_list2 = plt2[:, 1]
y_plt_list3 = plt3[:, 1]
print(np.sort(y_plt_list3,kind='quick',order=None))
y_plt_list4 = plt4[:, 1]
y_plt_list5 = plt5[:, 1]
print(np.sort(y_plt_list4,kind='quick',order=None))

# 画图
fig = plt.figure(figsize=(11,6)) # 通过调整该行来调整子图大小
ax1 = plt.subplot(232)
ax2 = plt.subplot(233)
ax3 = plt.subplot(235)
ax4 = plt.subplot(236)
ax5 = plt.subplot(231)
# 可以单独修改某一个图的坐标刻度格式，因为已经全局设置了Times New Roman字体所以这里不用使用了
# x1_label = ax[0][0].get_xticklabels()
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax[0][0].get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

ax1.plot(x_plt_list, y_plt_list1)
ax1.set_title('(a) $\mathit{\sigma}_1=0.01$', font1)
ax1.set_xlabel('episodes', font1)
ax1.set_ylabel('rewards', font1)

ax2.plot(x_plt_list, y_plt_list2)
ax2.set_title('(b) $\mathit{\sigma}_2=0.001$', font1)
ax2.set_xlabel('episodes', font1)
ax2.set_ylabel('rewards', font1)

ax3.plot(x_plt_list, y_plt_list3)
ax3.set_title('(c) $\mathit{\sigma}_3=0.1$', font1) #\mathrm{\sigma}表示正体字，\mathit{\sigma}表示斜体字
ax3.set_xlabel('episodes', font1)
ax3.set_ylabel('rewards', font1)

ax4.plot(x_plt_list, y_plt_list4)
ax4.set_title('(d) $\mathit{\sigma}_4=0.0001$', font1)
ax4.set_xlabel('episodes', font1)
ax4.set_ylabel('rewards', font1)

ax5.plot(x_plt_list, y_plt_list5)
ax5.set_title('$\mathit{\sigma}_{sd}$', font1)
ax5.set_xlabel('episodes', font1)
ax5.set_ylabel('rewards', font1)

plt.tight_layout()
image_name = 'thegama_compare'
# 保存图片
filename = '{}.png'.format(image_name)
plt.savefig('./{}'.format(filename))
# print('Saved trajectory to {}.'.format(filename))
