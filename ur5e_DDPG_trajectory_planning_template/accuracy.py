"""
通过该算法可以统计出某一次训练中所有有效策略的精度。并且以散点图的形式呈现出来，然后保存为文件accuracy of episodes，并且存放在
当前路径下。
"""
import end_motion_path_plt_and_ansys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# ------------------------------------------绘图字体初始化---------------------------------------------
# 字体初始化
plt.rc('font', family='Times New Roman', size=12) # 全局设置Times New Roman字体
# 将公式的字体全部设置为常规字体，对本例而言就是Times New Roman字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
# ------------------------------------------绘图字体初始化---------------------------------------------

# ------------------------------------------绘图主体部分---------------------------------------------
#初始化路径、对象
emp_path = '../../ur5e_DDPG_article2_0_01_20000csv/end_motion_path_csv/'
est_path = './'
EMP = end_motion_path_plt_and_ansys.End_Motion_Path(max_episode=20000, emp_path=emp_path, est_path=est_path)
# 初始化参数
after_smooth = False
episode_sf_list, first_reach_step_list = EMP.read_csv(type=1)   #读取episode_sf_table.csv文件
accuracy_list = []
for i in range(len(episode_sf_list)):
    accuracy_of_end = EMP.batch_size_endpoint_mean(episode_sf_list=episode_sf_list,
                                               first_reach_step_list=first_reach_step_list,
                                               size=1, random=False, index_begin=i,
                                               batch_ansys=True, k=None, after_smooth=False) #这种参数配置可以按照episode_sf_table.csv的顺序从前到后输出每一回合的精度
    accuracy_list.append(accuracy_of_end)
print(accuracy_list)

#画散点图,准备数据
x_plt = episode_sf_list
y_plt = accuracy_list
episode_max = np.argmax(y_plt)
episode_min = np.argmin(y_plt)

#设定名称
title_name = '$\mathit{\sigma}_1  \mathit{train}$'
fig = plt.figure(1, figsize=(9, 6))
plt.title(title_name, fontsize=18)
plt.xlabel('episodes', fontsize=16)
# Y_label-----------------------------原本的ylabel
D_text_font = {'style': 'oblique', 'weight': 'bold'}    #将下标加粗
plt.text(-2300, 0.0130, '$\mathit{E}$'+'   /$\mathit{m}$', fontsize=16, rotation=90)
plt.text(-2100, 0.0135, 'D{}'.format("'"), fontdict=D_text_font, fontsize=10, rotation=90)   #用这三行来设计Y_label因为普通的已经不能够满足字体要求
# Y_label-----------------------------
plt.scatter(x_plt, y_plt, s=0.4)
# 画最小值最大值点
plt.plot([x_plt[episode_max]], [y_plt[episode_max]], 'o', color='r', linewidth=0.1)
plt.plot([x_plt[episode_min]], [y_plt[episode_min]], 'o', color='r', linewidth=0.1)
plt.text(x_plt[episode_max]-4500, y_plt[episode_max], 'max({},{:0.3f})'.format(x_plt[episode_max],y_plt[episode_max]), fontsize=14)
plt.text(x_plt[episode_min]-4500, y_plt[episode_min], 'min({},{:0.3f})'.format(x_plt[episode_min],y_plt[episode_min]), fontsize=14)
# 保存图片
image_name = 'accuracy_of_episodes'
filename = '{}.png'.format(image_name)
plt.savefig('./{}'.format(filename))
plt.close("all")
print('Saved accuracy_of_episodes.png to {}.'.format(filename))
# ------------------------------------------绘图主体部分---------------------------------------------