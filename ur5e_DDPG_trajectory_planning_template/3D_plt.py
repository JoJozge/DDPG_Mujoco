"""
作者：Jozge
通过此部分代码可以将某一回合的末端点移动轨迹绘成三维空间中的图像，
并且将图像保存到文件夹”end_motion_path_csv2png3D“中
"""
import end_motion_path_plt_and_ansys
import matplotlib.pyplot as plt
import matplotlib

# ------------------------------------------绘图字体初始化---------------------------------------------
plt.rc('font', family='Times New Roman', size=12) # 全局设置Times New Roman字体，包括坐标系刻度的格式
# 将公式的字体全部设置为常规字体，对本例而言就是Times New Roman字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# 设置一种字体格式
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
# ------------------------------------------绘图字体初始化---------------------------------------------

# ------------------------------------------绘图主体部分---------------------------------------------
# 初始化路径和对象
emp_path = '../../ur5e_DDPG_article2_0_01_20000csv/end_motion_path_csv/'
est_path = './'
EMP = end_motion_path_plt_and_ansys.End_Motion_Path(max_episode=20000, emp_path=emp_path, est_path=est_path)

x_list, y_list, z_list, total_reward= EMP.read_csv(k=1383)
EMP.plt_output(x_list=x_list, y_list=y_list, z_list=z_list, plot_3D=True, k=1383, quiet=False)
# ------------------------------------------绘图主体部分---------------------------------------------