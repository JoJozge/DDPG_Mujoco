"""
本例程相当于实验体对训练体输出的运动策略进行复现，同时也可以选择是否进行平滑处理操作，最后输出对应的表格。
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time

# ------------------------------------------滑动平滑处理函数---------------------------------------------
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
# ------------------------------------------滑动平滑处理函数---------------------------------------------

k = 16773 # 设置复现的回合
smooth = True   # 是否平滑处理运动策略

# ------------------------------------------读取文件---------------------------------------------
df = pd.read_csv('../../ur5e_DDPG_article2_0_01_20000csv/joint_path_csv/joint_path_table_{}.csv'.format(k), header=None) # 文件地址根据不同训练进行修改
qpos_array = df[1:].values.astype(float)

matplotlib.rc("font", family='YouYuan')  # 调用中文字体，family还有其它选项
image_name = 'joint_path_{}'.format(k)

xaimplt_list = qpos_array[:, 0]
yaim_list_qpos0 = qpos_array[:, 1]
yaim_list_qpos1 = qpos_array[:, 2]
yaim_list_qpos2 = qpos_array[:, 3]
yaim_list_qpos3 = qpos_array[:, 4]
yaim_list_qpos4 = qpos_array[:, 5]
yaim_list_qpos5 = qpos_array[:, 6]
total_reward = qpos_array[:, 7][-2]
print(total_reward)

if smooth:
    # 平滑处理
    yaim_list_qpos0 = moving_average(yaim_list_qpos0, 51)
    yaim_list_qpos1 = moving_average(yaim_list_qpos1, 51)
    yaim_list_qpos2 = moving_average(yaim_list_qpos2, 51)
    yaim_list_qpos3 = moving_average(yaim_list_qpos3, 51)
    yaim_list_qpos4 = moving_average(yaim_list_qpos4, 51)
    yaim_list_qpos5 = moving_average(yaim_list_qpos5, 51)

    yaim_list_qpos0 = yaim_list_qpos0[:len(xaimplt_list)]
    yaim_list_qpos1 = yaim_list_qpos1[:len(xaimplt_list)]
    yaim_list_qpos2 = yaim_list_qpos2[:len(xaimplt_list)]
    yaim_list_qpos3 = yaim_list_qpos3[:len(xaimplt_list)]
    yaim_list_qpos4 = yaim_list_qpos4[:len(xaimplt_list)]
    yaim_list_qpos5 = yaim_list_qpos5[:len(xaimplt_list)]

max_episode = len(xaimplt_list)

xml_path = './universal_robots_ur5e/scene.xml'  #xml file (assumes this is in the same folder as this file)
simend = 2000  #simulation time simulate展示时长
print_camera_config = 1 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# initial aim_position
# position_aim0 = 0
position_aim0_start = 0
position_aim0_end = 6.28
position_aim0 = np.linspace(position_aim0_start,position_aim0_end,max_episode)
position_aim1 = 0
position_aim2 = 0
position_aim3 = 1.57
position_aim4 = 6.28
position_aim5 = 0

def quat2euler(quat_mojoco):
    """
    把mujoco的四元数转换成欧拉角
    :param quat_mojoco:
    :return:euler[pitch,row,yaw]
    """
    quat_scipy = np.array([quat_mojoco[3],quat_mojoco[0],quat_mojoco[1],quat_mojoco[2]])
    r = R.from_quat(quat_scipy)
    euler = r.as_euler('xyz',degrees=True)
    return euler

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
    data.ctrl[0] = -3500 * (data.qpos[0] - yaim_list_qpos0[i]) - 100 * (data.qvel[0] - 0)
    data.ctrl[1] = -3500 * (data.qpos[1] - yaim_list_qpos1[i]) - 100 * (data.qvel[1] - 0)
    data.ctrl[2] = -3500 * (data.qpos[2] - yaim_list_qpos2[i]) - 100 * (data.qvel[2] - 0)
    data.ctrl[3] = -3000 * (data.qpos[3] - yaim_list_qpos3[i]) - 100 * (data.qvel[3] - 0)
    data.ctrl[4] = -3000 * (data.qpos[4] - yaim_list_qpos4[i]) - 100 * (data.qvel[4] - 0)
    data.ctrl[5] = -3000 * (data.qpos[5] - yaim_list_qpos5[i]) - 100 * (data.qvel[5] - 0)

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

def set_position_servo(actuator_no, kp):
    model.actuator_gainprm[actuator_no, 0] = kp
    model.actuator_biasprm[actuator_no, 1] = -kp

def set_velocity_servo(actuator_no, kv):
    model.actuator_gainprm[actuator_no, 0] = kv
    model.actuator_biasprm[actuator_no, 2] = -kv

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
cam.azimuth = -34.79999999999997 ; cam.elevation = -32.59999999999995 ; cam.distance =  1.2110691051576454
cam.lookat =np.array([ 0.1690917205283202 , 0.31239034173545277 , 0.2826418946807113 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

#initialize
i = 0
time_cnt = 0
dt = 0.001
data_time0 = 0
delta_time = 0
# 画图初始化
timelist = []
qvel0list = []
qvel1list = []
qvel2list = []
qvel3list = []
qvel4list = []
qvel5list = []
qpos0list = []
qpos1list = []
qpos2list = []
qpos3list = []
qpos4list = []
qpos5list = []
endpointlist = []
qpos_table_list = []

step = 0

while not glfw.window_should_close(window) and i<(max_episode):
    time_prev = time_cnt
    if i == 0:
        data_time0 = data.time
        # print(data_time0)
    if i == 1:
        delta_time = data.time - data_time0
        # print(delta_time)
    while (time_cnt - time_prev < 90.0/60.0):
        mj.mj_forward(model, data)
        time_cnt += dt
        mj.mj_step(model, data)
    # 末端点与目标点的位置
    # print(f'末端点的位置为({data.site_xpos[0][0]},{data.site_xpos[0][1]},{data.site_xpos[0][2]})')
    # print(f'目标点的位置为({data.site_xpos[1][0]},{data.site_xpos[1][1]},{data.site_xpos[1][2]})')
    # 画图
    timelist.append(data.time)
    qpos0list.append(data.qpos[0])
    qpos1list.append(data.qpos[1])
    qpos2list.append(data.qpos[2])
    qpos3list.append(data.qpos[3])
    qpos4list.append(data.qpos[4])
    qpos5list.append(data.qpos[5])
    endpointlist.append([data.site_xpos[0][0], data.site_xpos[0][1], data.site_xpos[0][2], qpos_array[:, 7][i]])  # 末端点位置采样
    qpos_table_data = np.concatenate(([step], data.qpos[0:6], [total_reward]),axis=0)  # 将steps和data.time转化成[采样次数，13]的数据格式,采样次数=总仿真时间/单次采样时间间隔
    qpos_table_list.append(qpos_table_data)  # 六个关节的角度变化采样
    step += 1
    # print(f'{data.time}:{data.qvel[7]}')
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
    i += 1

glfw.terminate()

# 存表格
if smooth:
    filename = 'end_motion_table_smooth_{}.csv'.format(k)
    endpointlist_table = np.array(endpointlist)
    df = pd.DataFrame(endpointlist_table, columns=['x', 'y', 'z', 'reward'])
    df.to_csv('./smooth_plt/{}'.format(filename), index=False)
    print('Saved end_motion_table_smooth_{}.csv to ./smooth_plt'.format(k))

if smooth:
    qpos_table = np.array(qpos_table_list)
    df = pd.DataFrame(qpos_table, columns=['time', 'shoulder link', 'upper arm link', 'forearm link', 'wrist1 link',
                                           'wrist2 link', 'wrist3 link', 'total_reward'])
    df.to_csv('./smooth_plt/joint_path_table_smooth_{}.csv'.format(k), index=False)
    print('Saved joint_path_table_smooth_{}.csv to ./smooth_plt'.format(k))
