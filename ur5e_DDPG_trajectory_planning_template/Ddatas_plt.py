"""
绘制第k回合的第x个关节的D数据集图，必须是可以输出有效运动策略的回合才行
"""

import qpos6_plt
import matplotlib.pyplot as plt
import matplotlib

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

# ------------------------------------------绘图主函数部分---------------------------------------------
# 数据路径
jsp_path = '../../ur5e_DDPG_article2_0_01_20000csv/joint_path_csv/'
est_path = './'
JSP = qpos6_plt.Joint_Space_Path(max_episode=20000, jsp_path=jsp_path, est_path=est_path,smooth=False)  # smooth选项决定True平滑还是不平滑False
k = 1771
JSP.plt_qposx_D(x=3, k=k, font=font1)
# ------------------------------------------绘图主函数部分---------------------------------------------