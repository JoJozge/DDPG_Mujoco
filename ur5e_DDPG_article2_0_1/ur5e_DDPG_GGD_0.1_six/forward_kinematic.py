import numpy as np
import math

class Forward():
    def theta_matrix(self, theta):
        theta_matrix = np.array([[math.cos(theta), -math.sin(theta), 0, 0],[math.sin(theta), math.cos(theta), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        return theta_matrix
    def d_matrix(self, d):
        d_matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, d],[0, 0, 0, 1]])
        return d_matrix
    def alpha_matrix(self, alpha):
        alpha_matrix = np.array([[1, 0, 0, 0],[0, math.cos(alpha), -math.sin(alpha), 0],[0, math.sin(alpha), math.cos(alpha), 0],[0, 0, 0, 1]])
        return alpha_matrix
    def a_matrix(self, a):
        a_matrix = np.array([[1, 0, 0, a],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        return a_matrix
    def T_matrix(self, theta, d, alpha, a):
        matrix1 = self.theta_matrix(theta)
        matrix2 = self.d_matrix(d)
        matrix3 = self.alpha_matrix(alpha)
        matrix4 = self.a_matrix(a)
        return matrix1.dot(matrix2).dot(matrix3).dot(matrix4)

    def theta1_matrix(self, theta1, d):
        theta1_matrix = np.array([[math.cos(theta1), 0, math.sin(theta1), 0],[math.sin(theta1), 0, -math.cos(theta1), 0],[0, 1, 0, d],[0, 0, 0, 1]])
        return theta1_matrix
    def zrotation_matrix(self, theta):
        zrotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0, 0],[math.sin(theta), math.cos(theta), 0, 0],[0, 0, 1, 0],[0, 0, 0, 0]])
        return zrotation_matrix
if __name__=='__main__':

    forward1 = Forward()
    M1 = forward1.T_matrix(theta=1.57, d=0.163, alpha=np.pi/2, a=0)
    M2 = forward1.T_matrix(theta=-1.57, d=0, alpha=0, a=-0.425)
    M3 = forward1.T_matrix(theta=1.57, d=0, alpha=0, a=-0.392)
    M4 = forward1.T_matrix(theta=1.57, d=0.127, alpha=np.pi/2, a=0)
    M5 = forward1.T_matrix(theta=1.57, d=0.1, alpha=-np.pi/2, a=0)
    M6 = forward1.T_matrix(theta=1.57, d=0.1, alpha=0, a=0)
    Z = forward1.zrotation_matrix(-1.57)
    # a = forward1.theta_matrix(0.50)
    # print(a)
    # b = d_matrix(1)
    # print(b)
    # c = alpha_matrix(np.pi/2)
    # print(c)
    # d = a_matrix(0)
    # print(d)
    # e = a.dot(b).dot(c).dot(d)
    M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6)
    M = Z.dot(M)
    # M = M1
    print(M)

