import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

class envCube:
    dt = 0.05        #转动的参考时间 六个转轴最大转速为正负180°/s,0.05s可以转动正负9度，也就是正负0.157弧度

#action_bound0-action_bound4对应theta1-theta5的action的每帧动作范围
    action_bound0 = [-0.157, 0.157]   #转动角每一步的转动范围
    action_bound1 = [-0.157, 0.157]
    action_bound2 = [-0.157, 0.157]
    action_bound3 = [-0.157, 0.157]
    action_bound4 = [-0.157, 0.157]
    action_bound5 = [-0.157, 0.157]

#arminfo_bound0-arminfo_bound4对应theta1-theta5的角度可取值范围
    arminfo_bound0 = [-2 * np.pi, 2 * np.pi]
    arminfo_bound1 = [-np.pi, 0]
    arminfo_bound2 = [-np.pi/2, np.pi/2]
    arminfo_bound3 = [-2 * np.pi, 2 * np.pi]
    arminfo_bound4 = [-2 * np.pi, 2 * np.pi]
    arminfo_bound5 = [-2 * np.pi, 2 * np.pi]

    state_dim = 12               #DDPG算法的状态空间，5个theta，4个关节点到目标的x,y,z，3个目标随机生成地点，1个是否触碰到目标，5+3*4+3+1
    action_dim = 5
    GOAL = {'x': 1., 'y': 0., 'z': 0.5}
    POINT0 = {'x': 1., 'y': 0., 'z': 1}
    POINT1 = {'x': 2., 'y': 0., 'z': 1}
    POINT2 = {'x': 3., 'y': 0., 'z': 1}
    END_POINT = {'x': .9, 'y': .1, 'z': 0.4}

    def __init__(self):
        """
        初始化三个旋转角度的值均为0
        """
        self.arm_info = np.zeros(5, dtype=[('r', np.float32)])
        self.target_info = np.zeros(3, dtype=[('d', np.float32)])
        self.on_goal = False
        self.on_goal_count = 0

    def reset(self):
        self.episode_step = 0
        self.target_info['d'] = 0.8 * np.random.rand(3)
        self.target_info['d'] = np.random.choice([-1,1]) * self.target_info['d']
        self.arm_info['r'][0] = 2 * np.pi * np.random.rand(1)
        self.arm_info['r'][1] = np.pi * np.random.rand(1)
        self.arm_info['r'][2] = np.pi * np.random.rand(1)
        self.arm_info['r'][3] = np.pi * np.random.rand(1)
        self.arm_info['r'][4] = np.pi * np.random.rand(1)
        
        reset_state = np.concatenate((self.arm_info['r'], self.target_info['d']))
        # self.arm_info['r'][1] = 2 * np.pi * np.random.rand(1)
        # self.arm_info['r'][2] = 2 * np.pi * np.random.rand(1)

        return reset_state

    def step(self, action, target_position_init, goal=False,  end_point=False, point0=False, point1 = False, point2 = False):

        """
        输入动作，目标位置，末端点位置
        输出下一个状态，该动作奖励值，是否完成动作
        :param action:[theta1, theta2, theta3, theta4, theta5]
        :param goal: {'x':目标x坐标, 'y':目标y坐标, 'z':目标z坐标}
        :param obj: {'x':末端点x坐标, 'y':末端点y坐标, 'z':末端点z坐标}
        :return:[状态[theta1, theta2, theta3], 碰到目标的奖励reward, 是否已经完成done]
        """
        self.episode_step += 1

        done = False
        r = 0.
        # action[0] = np.clip(action, *self.action_bound0)[0]    #使输入的action在action_bound之中
        # action[1] = np.clip(action, *self.action_bound1)[1]
        # action[2] = np.clip(action, *self.action_bound2)[2]
        # action[3] = np.clip(action, *self.action_bound3)[3]
        # action[4] = np.clip(action, *self.action_bound4)[4]

        #上一个状态值（角度）加上动作值乘以时间，得出下一个角度的状态值
        self.arm_info['r'] += action
        self.arm_info['r'][0] = np.clip(self.arm_info['r'], *self.arminfo_bound0)[0]
        self.arm_info['r'][1] = np.clip(self.arm_info['r'], *self.arminfo_bound1)[1]
        self.arm_info['r'][2] = np.clip(self.arm_info['r'], *self.arminfo_bound2)[2]
        self.arm_info['r'][3] = np.clip(self.arm_info['r'], *self.arminfo_bound3)[3]
        self.arm_info['r'][4] = np.clip(self.arm_info['r'], *self.arminfo_bound4)[4]

        # self.arm_info['r'][0] %= np.pi * 2
        # self.arm_info['r'][1] %= np.pi
        # self.arm_info['r'][2] %= np.pi
        # self.arm_info['r'][3] %= np.pi
        # self.arm_info['r'][4] %= np.pi

        s = np.concatenate((self.arm_info['r'], target_position_init))

        if not goal:
            self.goal = self.GOAL
        else:
            self.goal = goal

        if not end_point:
            self.end_point = self.END_POINT
        else:
            self.end_point = end_point

        if not point0:
            self.point0 = self.POINT0
        else:
            self.point0 = point0

        if not point1:
            self.point1 = self.POINT1
        else:
            self.point1 = point1

        if not point2:
            self.point2 = self.POINT2
        else:
            self.point2 = point2

        #feature enginering
        # dist1 = np.array([self.goal['x'] - self.point0['x'], self.goal['y'] - self.point0['y'], self.goal['z'] - self.point0['z']])
        # dist2 = np.array([self.goal['x'] - self.point1['x'], self.goal['y'] - self.point1['y'], self.goal['z'] - self.point1['z']])
        # dist3 = np.array([self.goal['x'] - self.point2['x'], self.goal['y'] - self.point2['y'], self.goal['z'] - self.point2['z']])
        dist4 = np.array([self.goal['x'] - self.end_point['x'], self.goal['y'] - self.end_point['y'], self.goal['z'] - self.end_point['z']])

        dist5 = [(self.goal['x'] - self.end_point['x']), (self.goal['y'] - self.end_point['y']), (self.goal['z'] - self.end_point['z'])]
        r = -np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)

        # if abs(self.goal['x'] - self.end_point['x']) <= 0.3:
        #     if abs(self.goal['y'] - self.end_point['y']) <= 0.3:
        #         if abs(self.goal['z'] - self.end_point['z']) <= 0.3:
        #             r += 1

        if abs(self.goal['x'] - self.end_point['x']) <= 0.05:
            if abs(self.goal['y'] - self.end_point['y']) <= 0.05:
                if abs(self.goal['z'] - self.end_point['z']) <= 0.05:
                    r += 20
                    self.on_goal = True
                    self.on_goal_count += 1
                    if self.on_goal_count>100:                                #将on_goal设置过短会导致一次成功的触碰之后，reward为0异常
                        self.on_goal_count = 0
                        done = True

        if self.episode_step > 2000:
            done = True

        # s = np.concatenate((s, dist1, dist2, dist3, dist4, [1. if self.on_goal else 0.]))
        s = np.concatenate((s, dist4, [1. if self.on_goal else 0.]))
        self.on_goal = False
        # print(s)
        return s, r, done

    def sample_action(self):
        """
        进行一个随机操作，随机
        :return:
        """
        return np.random.rand(5) - 0.5

if __name__ == '__main__':

    cube = envCube()
    print(cube.arm_info['r'])
    # cube.reset()
    action = cube.sample_action()
    print(action)
    s_, r, done = cube.step(action)
    print(s_)
    print(r)
    print(done)





