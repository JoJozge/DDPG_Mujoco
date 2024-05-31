"""
获取指定回合的数据集D的方差
"""
import end_motion_path_plt_and_ansys
import qpos6_plt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


# 数据路径
jsp_path = '../../ur5e_DDPG_article2_0_01_20000csv/joint_path_csv/'
est_path = './'
JSP = qpos6_plt.Joint_Space_Path(max_episode=20000, jsp_path=jsp_path, est_path=est_path,smooth=False)  # smooth选项决定True平滑还是不平滑False

emp_path = '../../ur5e_DDPG_article2_0_01_20000csv/end_motion_path_csv/'
est_path = './'
EMP = end_motion_path_plt_and_ansys.End_Motion_Path(max_episode=20000, emp_path=emp_path, est_path=est_path)

k=1383
episode_sf_list, first_reach_step_list = JSP.read_csv(type=1)  # 读取episode_sf_table文件，该文件储存了轨迹规划成功的回合数，和每回合中第一次到达目标点的step值
# 特定回合采样输出方差
for i in range(6):
    qpos1var = JSP.batch_size_qpos_var(episode_sf_list=episode_sf_list, first_reach_step_list=first_reach_step_list, x=i+1, k=1383) # 用这种方式输入可以求某个特定的episode的D集合方差
    print(qpos1var)
print('-------------------------------------------')
# 读取表格
step_list, yplt_list_qpos0, yplt_list_qpos1, yplt_list_qpos2, yplt_list_qpos3, yplt_list_qpos4, yplt_list_qpos5, total_reward = JSP.read_csv(k=k, type=3)  # 读取smooth后的六个关节角转动表

x_list, y_list, z_list, total_reward1 = EMP.read_csv(k=k, type=2)
first_reach_step = EMP.EMP_Ansys(x_list=x_list, y_list=y_list, z_list=z_list, total_reward=total_reward, k=k)
for i in range(6):
    qpos1var = JSP.batch_size_qpos_var(episode_sf_list=[k], first_reach_step_list=[first_reach_step], x=i+1, k=1383, after_soomth=True)
    print(qpos1var)

