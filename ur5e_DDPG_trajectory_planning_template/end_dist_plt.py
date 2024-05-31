"""
绘制特定回合末端点到目标点距离随时间变化的图,以及该图的局部放大图
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
emp_path = '../../ur5e_DDPG_article2_0_01_20000csv/end_motion_path_csv/'
est_path = './'
EMP = end_motion_path_plt_and_ansys.End_Motion_Path(max_episode=20000, emp_path=emp_path, est_path=est_path)
k = 18996
episode_sf_list, first_reach_step_list = EMP.read_csv(type=1)
# 平滑处理之前绘图
origin_first_reach = EMP.dist_plt(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, k=k, after_smooth=False, type=1) #原图
EMP.dist_plt(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, k=k, after_smooth=False, type=2) #对原图进行局部放大的图
# 获取平滑处理之后的episode_sf_smooth_table主要是first_reach_step
x_list, y_list, z_list, total_reward= EMP.read_csv(k=k, type=2) # 读取平滑处理之后的末端点的空间坐标，这项程序需要依赖joint_path_smooth_visualize.py例程运行
first_reach_step = EMP.EMP_Ansys(x_list=x_list, y_list=y_list, z_list=z_list, total_reward=total_reward, k=k)   # 对平滑之后的运动策略重新进行分析
smooth_expisode_sf_table = np.array(EMP.expisode_sf_list)  # 直接读取平滑之后的
df = pd.DataFrame(smooth_expisode_sf_table, columns=['episode', 'first_reach_step'])
df.to_csv('./episode_sf_smooth_table.csv', index=False)
#平滑处理之后的图片画图方式
EMP.dist_plt(episode_sf_list=[k], first_reach_step_list=[first_reach_step], k=k, origin_first_reach=origin_first_reach, after_smooth=True, type=1)
EMP.dist_plt(episode_sf_list=[k], first_reach_step_list=[first_reach_step], origin_first_reach=origin_first_reach, k=k, after_smooth=True, type=2)
# ------------------------------------------绘图主体部分---------------------------------------------