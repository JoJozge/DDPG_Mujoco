"""
主仿真程序
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
from ur5e_env import envCube
import torch
from DDPG import DDPG
import pandas as pd
import os

def mkdir(folderpath, foldername):
    folder = os.path.exists(folderpath+foldername)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folderpath+foldername)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('The folder named {} has been created'.format(foldername))
    else:
        print("The folder already exists!")

#-------------------------------------基本准备工作------------------------------#
#出现报错OMP的解决办法
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#保证画图的负号显示
plt.rcParams['axes.unicode_minus']=False
# 创建文件夹保存csv表格文件，以便后续分析数据使用，保存表格数据的文件夹应该与项目文件夹在同级路径之下
folderpath = "M:/ur5e_DDPG_trajectory_planning_20000csv/"   # 保存表格文件的路径
foldername1 = 'end_motion_path_csv'
foldername2 = 'joint_path_csv'
foldername3 = 'reward_episode_csv'
mkdir(folderpath, foldername1)  # 调用函数
mkdir(folderpath, foldername2)
mkdir(folderpath, foldername3)
#-------------------------------------基本准备工作------------------------------#

xml_path = './universal_robots_ur5e/scene.xml'  #xml file (assumes this is in the same folder as this file)
# simend = 2000  #simulation time simulate展示时长, mujoco源码中的，本例中未使用
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    # pass
    set_torque_servo(0, 1)  # shoulder_pan_joint力矩伺服器，针对模型ur5etest1_2.xml
    set_torque_servo(1, 1)  # shoulder_lift_joint力矩伺服器
    set_torque_servo(2, 1)  # elbow_joint力矩伺服器
    set_torque_servo(3, 1)  # wrist_1_joint力矩伺服器
    set_torque_servo(4, 1)  # wrist_2_joint力矩伺服器
    set_torque_servo(5, 1)  # wrist_3_joint力矩伺服器

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    # pass
    data.ctrl[0] = -3500 * (data.qpos[0] - state[0]) - 100 * (data.qvel[0] - 0)
    data.ctrl[1] = -3500 * (data.qpos[1] - state[1]) - 100 * (data.qvel[1] - 0)
    data.ctrl[2] = -3500 * (data.qpos[2] - state[2]) - 100 * (data.qvel[2] - 0)
    data.ctrl[3] = -3000 * (data.qpos[3] - state[3]) - 100 * (data.qvel[3] - 0)
    data.ctrl[4] = -3000 * (data.qpos[4] - state[4]) - 100 * (data.qvel[4] - 0)
    data.ctrl[5] = -3000 * (data.qpos[5] - state[5]) - 100 * (data.qvel[5] - 0)

def set_torque_servo(actuator_no, flag):
    """
    :param actuator_no: 执行器编号
    :param flag: 0关闭1开启
    :return:
    """
    if (flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 接触力显示化
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False     #接触点显示选项
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False    #接触力显示选项
opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False    #是否透明选项
model.vis.scale.contactwidth = 0.1              # 接触面显示宽度
model.vis.scale.contactheight = 0.01             # 接触面显示高度
model.vis.scale.forcewidth = 0.05               # 力显示宽度
model.vis.map.force = 0.5                       # 力显示长度
# opt.frame = mj.mjtFrame.mjFRAME_GEOM            # 显示geom的frame

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration/相机初始位置设定
cam.azimuth = 118.00000000000006 ; cam.elevation = -52.800000000000004 ; cam.distance =  2.840250715599666
cam.lookat =np.array([ -0.02409581650619193 , 0.010921802046488856 , 0.24125779903848252 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

#initialize
i = 0          # 第i个episode
time = 0
dt = 0.001      # 该值可以改变仿真时间步的间隔大小。注意仿真时间与实际时间不同。

# 种子设置，不同的值会对实验结果产生不同影响。将以后每次实验的随机数都初始化一致，
np.random.seed(0)
torch.manual_seed(0)

# 画图初始化
timelist = []
qvel0list = []
qvel1list = []
qvel2list = []
qvel3list = []
qvel4list = []
qvel5list = []

endpointlist = []

plt_3d_ok = False
# 强化学习环境初始化
max_episode = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.cuda.is_available())

env = envCube()
state_dim = env.state_dim
action_dim = env.action_dim
max_action = [float(env.action_bound0[1]), float(env.action_bound1[1]), float(env.action_bound2[1]), float(env.action_bound3[1]),float(env.action_bound4[1]),float(env.action_bound5[1])]
max_action = torch.FloatTensor(max_action).to(device)

directory = './exp' + "ur5e" +'./'

agent = DDPG(state_dim, action_dim, max_action)
ep_r = 0

load = False
if load: agent.load()

total_step = 0
t = 0

# table data init
save_table = True
reward_table_list = []
# plot
plot = True
reward_list = []
episode_list = []
plot_timestep = 2000
delta_time = 0.034

while not glfw.window_should_close(window) and i<(max_episode):

    # plt list init
    timelist = []
    steplist = []
    qpos0list = []
    qpos1list = []
    qpos2list = []
    qpos3list = []
    qpos4list = []
    qpos5list = []

    # qpos_dict = {}
    # for j in range(6):
    #     key_name = 'qpos{}list'.format(j)
    #     qpos_dict[key_name] = []
    # print(type(qpos_dict['qpos1list']))

    # table list init
    qpos_table_list = []
    endpointlist = []

    # new episode init
    state = env.reset() # (9,),state[0]-state[5]为六个角度的旋转值，state[6],state[7],state[8]为目标点的三个坐标
    for j in range(6):
        data.qpos[j] = state[j]
    mj.mj_forward(model, data)
    mj.mj_step(model, data)
    total_reward = 0
    step = 0
    done = False
    end_point = {'x': data.site_xpos[0][0], 'y': data.site_xpos[0][1], 'z': data.site_xpos[0][2]}
    goal = {'x': data.site_xpos[1][0], 'y': data.site_xpos[1][1], 'z': data.site_xpos[1][2]}
    dist4 = np.array([goal['x'] - end_point['x'], goal['y'] - end_point['y'], goal['z'] - end_point['z']])
    state = np.concatenate((state, dist4, [0.]), axis=0)    # (13,)

    i += 1
    target_init = [state[6], state[7], state[8]]

    while not done:

        # 从环境中获取末端点的空间坐标和目标点的空间坐标
        end_point = {'x': data.site_xpos[0][0], 'y': data.site_xpos[0][1], 'z': data.site_xpos[0][2]}
        goal = {'x': data.site_xpos[1][0], 'y': data.site_xpos[1][1], 'z': data.site_xpos[1][2]}
        # 强化学习步骤
        action = agent.select_action(state) # (6,)
        action = (action + np.random.normal(0, 0.01, size=6)).clip(env.action_bound0[0], env.action_bound0[1])  # 给每一时间步的决策添加噪声，该语句是增强智能体探索能力的关键语句，有几率使得智能体跳出局部最优解
        next_state, reward, done = env.step(action, target_init, goal, end_point)           # 在数值上更新强化学习中的环境参数s_，r等，

        # 末端点与目标点的位置
        # print(f'末端点的位置为({data.site_xpos[0][0]},{data.site_xpos[0][1]},{data.site_xpos[0][2]})')
        # print(f'目标点的位置为({data.site_xpos[1][0]},{data.site_xpos[1][1]},{data.site_xpos[1][2]})')

        if step <= 998:
            total_reward += reward  # 因为这个实验设置时间步为1000，所以当第999（开始的step=0，故step<=998）个时间步加上该步的奖励值之后不应该在继续加。因为第1000个时间步是终止状态。

        # 表格数据采样，每个仿真回合采样一次
        endpointlist.append([data.site_xpos[0][0], data.site_xpos[0][1], data.site_xpos[0][2], total_reward])   #末端点位置采样
        qpos_table_data = np.concatenate(([step], data.qpos[0:6], [total_reward]), axis=0)  # 将steps和data.time转化成[采样次数，13]的数据格式,采样次数=总仿真时间/单次采样时间间隔
        qpos_table_list.append(qpos_table_data) #六个关节的角度变化采样

        # replay_buffer存储
        agent.replay_buffer.push((state, next_state, action, reward, np.float32(done)))

        # 更新当前时间步的环境中的状态到下一个时间步
        state = next_state
        time_prev = time
        # 更新仿真环境，必须更新，不然仿真环境的数值不会发生变化。
        while (time - time_prev < 1.0 / 60.0):
            mj.mj_forward(model, data)
            time += dt
            mj.mj_step(model, data)

        if done:
            break

        step += 1

        # 画图
        timelist.append(data.time)
        steplist.append(step)
        qpos0list.append(data.qpos[0])
        qpos1list.append(data.qpos[1])
        qpos2list.append(data.qpos[2])
        qpos3list.append(data.qpos[3])
        qpos4list.append(data.qpos[4])
        qpos5list.append(data.qpos[5])
        # print('qpos0list:{}'.format(qpos0list))
        # for j in range(6):
        #     for k in range(6):
        #         key_name = 'qpos{}list'.format(k)
        #         if k == j:
        #             qpos_dict[key_name].append(data.qpos[j])
        # print('qpos0dict:{}'.format(qpos_dict['qpos0list']))
        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        #print camera configuration (help to initialize the view)
        if (print_camera_config==1):
            print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
            print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    total_step += step + 1
    print("Total T:{} Episode:\t{} Total Reward:\t{:0.2f}".format(total_step, i, total_reward))
    episode_list.append(i)
    reward_list.append(total_reward)
    agent.update()
    reward_table_list.append(np.concatenate(([i], [total_reward]),axis=0))

    if i % 1 == 0 and total_reward >= -5000 and save_table:
        plt_3d = np.array(endpointlist)
        df = pd.DataFrame(plt_3d, columns=['x', 'y', 'z', 'reward'])
        df.to_csv(folderpath+foldername1+'/end_motion_table_{}.csv'.format(i), index=False)
        # print("表格已保存为 'table_output_{}.csv'".format(i))
        # plt_3d_ok = True

        qpos_table = np.array(qpos_table_list)
        df = pd.DataFrame(qpos_table, columns=['time', 'shoulder link', 'upper arm link', 'forearm link', 'wrist1 link',
                                               'wrist2 link', 'wrist3 link', 'total_reward'])
        df.to_csv(folderpath+foldername2+'/joint_path_table_{}.csv'.format(i), index=False)

        reward_table = np.array(reward_table_list)
        df = pd.DataFrame(reward_table, columns=['episode', 'total_reward'])
        df.to_csv(folderpath + foldername3 + '/reward_episode_table.csv', index=False)

    image_name = 'reward'
    # matplotlib.rc("font", family='YouYuan')  # 调用中文字体，family还有其它选项
    if plot:
        if i % plot_timestep == 0:
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
            plt.savefig('./reward_episode_png/{}'.format(filename))
            # print('Saved trajectory to {}.'.format(filename))
            plt.close(fig)
            plt.close("all")

            image_name = '第{}个episode的关节角度变化'.format(i)
            xplt_list = np.array(steplist)
            yplt_list_qpos0 = np.array(qpos0list)
            yplt_list_qpos1 = np.array(qpos1list)
            yplt_list_qpos2 = np.array(qpos2list)
            yplt_list_qpos3 = np.array(qpos3list)
            yplt_list_qpos4 = np.array(qpos4list)
            yplt_list_qpos5 = np.array(qpos5list)

            # 可以绘制一回合内的六个关节角度的变化
        # if i % 1000 == 0:
            _, ax = plt.subplots(3, 2, figsize=(8, 8))
            ax[0][0].plot(xplt_list, yplt_list_qpos0, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[0][0].set_title('shoulder link')
            ax[0][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[0][0].set_ylabel(r'''$\theta_{1}$/rad''')
            ax[0][0].legend()
            ax[1][0].plot(xplt_list, yplt_list_qpos1, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[1][0].set_title('upper arm link')
            ax[1][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[1][0].set_ylabel(r'''$\theta_{2}$/rad''')
            ax[1][0].legend()
            ax[2][0].plot(xplt_list, yplt_list_qpos2, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[2][0].set_title('forearm link')
            ax[2][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[2][0].set_ylabel(r'''$\theta_{3}$/rad''')
            ax[2][0].legend()
            ax[0][1].plot(xplt_list, yplt_list_qpos3, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[0][1].set_title('wrist1 link')
            ax[0][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[0][1].set_ylabel(r'''$\theta_{4}$/rad''')
            ax[0][1].legend()
            ax[1][1].plot(xplt_list, yplt_list_qpos4, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[1][1].set_title('wrist2 link')
            ax[1][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[1][1].set_ylabel(r'''$\theta_{5}$/rad''')
            ax[1][1].legend()
            ax[2][1].plot(xplt_list, yplt_list_qpos5, label=r'$\mathrm{N}_{epi}$'+'={}'.format(i)+','+r'$\mathrm{R}_{epi}$'+'={:0.0f}'.format(total_reward))
            ax[2][1].set_title('wrist3 link')
            ax[2][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
            ax[2][1].set_ylabel(r'''$\theta_{6}$/rad''')
            ax[2][1].legend()
            plt.tight_layout()
            # 保存图片
            filename = '{}.png'.format(image_name)
            plt.savefig('./joint_path_png/{}'.format(filename))
            # print('Saved trajectory to {}.'.format(filename))
            plt.close(fig)
            plt.close("all")

    if i % 100 == 0:
        agent.save(total_reward, i)

glfw.terminate()
