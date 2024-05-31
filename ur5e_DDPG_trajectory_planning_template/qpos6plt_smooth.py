"""
绘制平滑处理之后的六个关节旋转图
"""
import end_motion_path_plt_and_ansys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import qpos6_plt
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
# 设置需要绘图的回合
k = 1383

jsp_path = '../../ur5e_DDPG_article2_0_01_20000csv/joint_path_csv/'
est_path = './'
JSP = qpos6_plt.Joint_Space_Path(max_episode=20000, jsp_path=jsp_path, est_path=est_path,smooth=False)  # smooth选项决定True平滑还是不平滑False

# 读取表格
step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = JSP.read_csv(k=k, type=3)  # 读取smooth后的六个关节角转动表


# 画图
fig, ax = plt.subplots(2, 3, figsize=(11, 6))

# 可以单独修改某一个图的坐标刻度格式，因为已经全局设置了Times New Roman字体所以这里不用使用了
# x1_label = ax[0][0].get_xticklabels()
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax[0][0].get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]


ax[0][0].plot(step_list, yplt_list_qpos0, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[0][0].set_title('shoulder link')
ax[0][0].set_xlabel('step')
ax[0][0].set_ylabel(r'''$\theta_{1}/rad$''')
ax[0][0].legend()

ax[0][1].plot(step_list, yplt_list_qpos1, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[0][1].set_title('upper arm link')
ax[0][1].set_xlabel('step')
ax[0][1].set_ylabel(r'''$\theta_{2}/rad$''')
ax[0][1].legend()

ax[0][2].plot(step_list, yplt_list_qpos2, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[0][2].set_title('forearm link')
ax[0][2].set_xlabel('step')
ax[0][2].set_ylabel(r'''$\theta_{3}/rad$''')
ax[0][2].legend()

ax[1][0].plot(step_list, yplt_list_qpos3, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[1][0].set_title('wrist1 link')
ax[1][0].set_xlabel('step')
ax[1][0].set_ylabel(r'''$\theta_{4}/rad$''')
ax[1][0].legend()

ax[1][1].plot(step_list, yplt_list_qpos4, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[1][1].set_title('wrist2 link')
ax[1][1].set_xlabel('step')
ax[1][1].set_ylabel(r'''$\theta_{5}/rad$''')
ax[1][1].legend()

ax[1][2].plot(step_list, yplt_list_qpos5, label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[1][2].set_title('wrist3 link')
ax[1][2].set_xlabel('step')
ax[1][2].set_ylabel(r'''$\theta_{6}/rad$''')
ax[1][2].legend()

plt.tight_layout()
image_name = 'qpos6plt_smooth'
# 保存图片
filename = '{}.png'.format(image_name)
plt.savefig('./{}'.format(filename))
# print('Saved trajectory to {}.'.format(filename))
