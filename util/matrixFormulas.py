import numpy as np

def rotationX(x):
    th = np.radians(x)
    return np.array([[1, 0, 0], \
            [0, np.cos(th), -np.sin(th)],\
            [0, np.sin(th), np.cos(th)]])

def rotationY(y):
    th = np.radians(y)
    return np.array([[np.cos(th), 0, np.sin(th)],\
            [0, 1, 0],
            [-np.sin(th), 0, np.cos(th)]])

def rotationZ(z):
    th = np.radians(z)
    return np.array([[np.cos(th), -np.sin(th), 0],\
            [np.sin(th), np.cos(th), 0],\
            [0, 0, 1]])

def rotationMatrixFrom2Vectors(a:tuple, b:tuple) -> np.array:
    R = np.identity(4)
    v = np.cross(a,b)
    sin = np.sqrt(sum(v*v))
    cos = np.dot(a,b)
    v_x = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])
    R[:3,:3] = R[:3,:3] + v_x + v_x @ v_x * (1/(cos+1))
    return R

def eulerZYXToRotationMatrix(zyx): # in ZYX Euler angles
    z, y, x = zyx
    R = np.identity(4)
    R[:3,:3] = rotationZ(z) @ rotationY(y) @ rotationX(x)
    return R

#def log(R):
#    # rotation matrix to rotation vector
#    th = np.arccos(np.clip((R[0][0]+ R[1][1] + R[2][2] - 1) / 2, -1, 1))
#
#    sin = np.sin(th)
#    if sin == 0:
#        return np.zeros(3)
#
#    v1 = (R[2][1] - R[1][2]) / (2*sin)
#    v2 = (R[0][2] - R[2][0]) / (2*sin)
#    v3 = (R[1][0] - R[0][1]) / (2*sin)
#
#    return np.array([v1, v2, v3]) * th
#
#
#def exp(v):
#    # rotation vector to rotation matrix
#    M = np.identity(4)
#
#    th = np.sqrt(sum(v*v))
#    cos = np.cos(th)
#    sin = np.sin(th)
#
#    if th < 0.0001:
#        return M
#
#    x,y,z = v/th
#
#    M[:3,:3] = np.array([[cos + x*x*(1-cos), x*y*(1-cos) - z*sin, x*z*(1-cos) + y*sin],\
#            [y*x*(1-cos) + z*sin, cos + y*y*(1-cos), y*z*(1-cos) - x*sin],\
#            [z*x*(1-cos) - y*sin, z*y*(1-cos) + x*sin, cos + z*z*(1-cos)]])
#    return M

def rotationMatrixToEulerZYX(M):
    # m: a rotation matrix
    sy = np.sqrt(M[0,0] * M[0,0] +  M[1,0] * M[1,0])
    singular = sy < 1e-6
    if not singular:
       x = np.arctan2(M[2,1] , M[2,2])
       y = np.arctan2(-M[2,0], sy)
       z = np.arctan2(M[1,0], M[0,0])
    else:
       x = np.arctan2(-M[1,2], M[1,1])
       y = np.arctan2(-M[2,0], sy)
       z = 0
    return np.degrees([z, y, x])
