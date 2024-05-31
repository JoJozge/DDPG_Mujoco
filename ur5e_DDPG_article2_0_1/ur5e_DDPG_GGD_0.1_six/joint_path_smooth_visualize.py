"""

"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

k = 2
smooth = True

df = pd.read_csv('./joint_path_csv/ur5e_qpos_table_{}.csv'.format(k), header=None)
qpos_array = df[1:].values.astype(float)

matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
image_name = 'joint_path_{}'.format(k)

xaimplt_list = qpos_array[:, 0]
yaim_list_qpos0 = qpos_array[:, 1]
yaim_list_qpos1 = qpos_array[:, 2]
yaim_list_qpos2 = qpos_array[:, 3]
yaim_list_qpos3 = qpos_array[:, 4]
yaim_list_qpos4 = qpos_array[:, 5]
yaim_list_qpos5 = qpos_array[:, 6]
total_reward = qpos_array[:, 7][998]
print(total_reward)

if smooth:
    # 平滑处理
    yaim_list_qpos0 = moving_average(yaim_list_qpos0, 101)
    yaim_list_qpos1 = moving_average(yaim_list_qpos1, 101)
    yaim_list_qpos2 = moving_average(yaim_list_qpos2, 101)
    yaim_list_qpos3 = moving_average(yaim_list_qpos3, 101)
    yaim_list_qpos4 = moving_average(yaim_list_qpos4, 101)
    yaim_list_qpos5 = moving_average(yaim_list_qpos5, 101)

    yaim_list_qpos0 = yaim_list_qpos0[:len(xaimplt_list)]
    yaim_list_qpos1 = yaim_list_qpos1[:len(xaimplt_list)]
    yaim_list_qpos2 = yaim_list_qpos2[:len(xaimplt_list)]
    yaim_list_qpos3 = yaim_list_qpos3[:len(xaimplt_list)]
    yaim_list_qpos4 = yaim_list_qpos4[:len(xaimplt_list)]
    yaim_list_qpos5 = yaim_list_qpos5[:len(xaimplt_list)]

max_episode = 1000

xml_path = './universal_robots_ur5e/scene.xml'  #xml file (assumes this is in the same folder as this file)
simend = 2000  #simulation time simulate展示时长
print_camera_config = 0 #set to 1 to print camera config
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
cam.azimuth = 118.00000000000006 ; cam.elevation = -52.800000000000004 ; cam.distance =  2.840250715599666
cam.lookat =np.array([ -0.02409581650619193 , 0.010921802046488856 , 0.24125779903848252 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

#initialize
i = 0
time = 0
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

while not glfw.window_should_close(window) and i<(max_episode):
    time_prev = time

    if i == 0:
        data_time0 = data.time
        # print(data_time0)
    if i == 1:
        delta_time = data.time - data_time0
        # print(delta_time)

    while (time - time_prev < 1.0/60.0):
        mj.mj_forward(model, data)
        time += dt
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

# 画图与保存
matplotlib.rc("font",family='YouYuan')  # 调用中文字体，family还有其它选项
image_name = 'smooth_plt'
xplt_list = np.array(timelist)
yplt_list_qpos0 = np.array(qpos0list)
yplt_list_qpos1 = np.array(qpos1list)
yplt_list_qpos2 = np.array(qpos2list)
yplt_list_qpos3 = np.array(qpos3list)
yplt_list_qpos4 = np.array(qpos4list)
yplt_list_qpos5 = np.array(qpos5list)

_, ax = plt.subplots(3, 2, figsize=(8, 8))
ax[0][0].plot(xplt_list, yplt_list_qpos0,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[0][0].set_title('shoulder link')
ax[0][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[0][0].set_ylabel(r'''$\theta_{1}$/rad''')
ax[0][0].legend()
ax[1][0].plot(xplt_list, yplt_list_qpos1,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[1][0].set_title('upper arm link')
ax[1][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[1][0].set_ylabel(r'''$\theta_{2}$/rad''')
ax[1][0].legend()
ax[2][0].plot(xplt_list, yplt_list_qpos2,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[2][0].set_title('forearm link')
ax[2][0].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[2][0].set_ylabel(r'''$\theta_{3}$/rad''')
ax[2][0].legend()
ax[0][1].plot(xplt_list, yplt_list_qpos3,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[0][1].set_title('wrist1 link')
ax[0][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[0][1].set_ylabel(r'''$\theta_{4}$/rad''')
ax[0][1].legend()
ax[1][1].plot(xplt_list, yplt_list_qpos4,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[1][1].set_title('wrist2 link')
ax[1][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[1][1].set_ylabel(r'''$\theta_{5}$/rad''')
ax[1][1].legend()
ax[2][1].plot(xplt_list, yplt_list_qpos5,label=r'$\mathrm{N}_{epi}$' + '={}'.format(k) + ',' + r'$\mathrm{R}_{epi}$' + '={:0.0f}'.format(total_reward))
ax[2][1].set_title('wrist3 link')
ax[2][1].set_xlabel('step/{:0.3f}s'.format(delta_time))
ax[2][1].set_ylabel(r'''$\theta_{6}$/rad''')
ax[2][1].legend()

plt.tight_layout()
# 保存图片
filename = '{}.png'.format(image_name)
plt.savefig('./smooth_plt/{}'.format(filename))
print('Saved trajectory to {}.'.format(filename))
plt.show()