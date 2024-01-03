from util.matrixFormulas import *
from util.rotations import exp, log

import numpy as np

from scipy.spatial.transform import Rotation

from data.Node import Node

class Posture():
    def __init__(self, frame=None, leftFootContact=None, rightFootContact=None, scale=1):
        if frame is None:
            self.position, self.orientations = None, None
            return

        self.position = np.array(frame[:3] * scale, dtype=np.float32)
        self.orientations = self.eulerToRotationMatrices('ZYX', frame[3:])
        self.lfootContact, self.rfootContact = leftFootContact, rightFootContact

    def eulerToRotationMatrices(self, axes, eulers):

        def rotationMatrix(axis, degree):
            R = np.identity(4)
            if axis == 'X':
                R[:3,:3] = rotationX(degree)
            elif axis == 'Y':
                R[:3,:3] = rotationY(degree)
            elif axis == 'Z':
                R[:3,:3] = rotationZ(degree)
            return R

        rotationMatrices = []
        
        for i in range(0, len(eulers), 3):
            R = np.identity(4)
            for axis in axes:
                R = R @ rotationMatrix(axis, eulers[i])
                i += 1
            rotationMatrices.append(R)

        return np.array(rotationMatrices, dtype=np.float32)


    def rootAlignment(self, targetPose):
        rootAlignVector = self.getPosition() - targetPose.getPosition()
        rootAlignVector[1] = 0

        rotM = self.getOrientations()[0] @ targetPose.getOrientations()[0].T
        rootAlignMatrix = exp([0,1,0] * log(rotM)) # Y-up
        return rootAlignVector, rootAlignMatrix

    def stitchedPosition(self, startPoint, rootAlignVector, rootAlignMatrix):
        p0 = startPoint # was target of the last root alignment
        D = rootAlignVector

        p = self.getPosition()

        d = p - p0
        d = np.append(d, 1)
        R = rootAlignMatrix

        return (p0 + D) + (R @ d)[:3]


    def interpolate(self, difference, ratio):
        yPosDiff, rotDiff = difference
        self.interpolateOnlyOrientations(rotDiff, ratio)

        ratio = 1 - ratio # 1...0 decreasing
        self.position += ratio * np.array([0,yPosDiff,0])

    def interpolateOnlyOrientations(self, difference, ratio):
        assert(ratio <= 1 and ratio >= 0)
        assert(len(difference) == len(self.getOrientations()))

        ratio = 1 - ratio # 1...0 decreasing

        ori = self.getOrientations() # not a copy!
        for i, d in enumerate(difference):
            ori[i] = ori[i] @ exp(d * ratio)

    def getPosition(self):
        return self.position
    def getOrientations(self):
        return self.orientations

    def getLocalFrame(self):
        M = self.orientations[0].copy()

        # LaFAN1 : X-up to Y-up frame
#        M[:3,:3] = rotationZ(-90) @ rotationY(-90) @ M[:3,:3]

        # Y-up projection
        zAxis = M[:3, 2][:3] * [1,0,1]
        zAxis /= np.sqrt(sum(zAxis*zAxis))
        yAxis = np.array([0,1,0])
        xAxis = np.cross(yAxis, zAxis)

        M = np.identity(4)
        M[:3] = np.stack((xAxis, yAxis, zAxis, self.getPosition())).T

        return M


    def getFootContacts(self):
        return (self.lfootContact, self.rfootContact)

#    def getPositionOf(self, targetName:str, root:Node) -> np.array:
#        return dict(self.forwardKinematics(root))[targetName]
#
#    def getAllNodePositions(self, root):
#        points = np.zeros((3,0))
#        for point in dict(self.forwardKinematics(root)).values():
#            points = np.hstack((points, np.array([point]).T))
#        return points # 3 x 31

    def inertialBlending(self, rotations, dt, T):

        def multQuats(q1, q2):
            v1, w1 = q1[:3], q1[3]
            v2, w2 = q2[:3], q2[3]
            w = w1*w2 - np.dot(v1,v2)
            v = w1*v2 + w2*v1 + np.cross(v1, v2)
            return np.concatenate((v,[w]))

        def inv(q):
            sizeDoubled = np.sum(q*q, axis=1)
            q_star = q * np.array([-1,-1,-1,1])
            q_inv = q_star / sizeDoubled[:, None] 
            return q_inv

        rangeInPowers = np.array([pow(T, i) for i in range(6)], np.float32)

        _, t1, t2, t3, t4, t5 = rangeInPowers

        quat = {
            '0':Rotation.from_matrix(rotations['0'][:,:3,:3]).as_quat(),
            '-1':Rotation.from_matrix(rotations['-1'][:,:3,:3]).as_quat(),
            '-2':Rotation.from_matrix(rotations['-2'][:,:3,:3]).as_quat(),
            't':Rotation.from_matrix(rotations['t'][:,:3,:3]).as_quat()
        }
        quat_inv = {
            '0':inv(quat['0'])
        }


        q0 = np.array([multQuats(q1, q2) for q1, q2 in zip(quat['-1'], quat_inv['0'])], dtype=np.float32)
        axisAngle = Rotation.from_quat(q0).as_rotvec()
        x0 = np.linalg.norm(axisAngle, axis=1)
        x0_axis = axisAngle/x0[:,None]
        x0_axis[np.isnan(x0_axis)] = 0 # zero divisions

        q_1 = np.array([multQuats(q1, q2) for q1, q2 in zip(quat['-2'], quat_inv['0'])], dtype=np.float32)

        q_1_projection = np.sum(q_1[:,:3] * x0_axis, axis=1) # dot projection
        x_1 = 2*np.arctan(q_1_projection / q_1[:,3])

        v0 = (x0 - x_1)/dt
        v0 = np.clip(v0, -np.inf, 0) # < 0
        a0 = (-8*v0*t1 - 20*x0) / (t2)
        a0 = np.clip(a0, 0, np.inf) # > 0

        xt = 0
        t = 1

        A = -((a0*t2 + 6*v0*t1 + 12*x0) / (2*t5))
        B = (3*a0*t2 + 16*v0*t1 + 30*x0) / (2*t4)
        C = -((3*a0*t2 + 12*v0*t1 + 20*x0) / (2*t3))

        elements = [x0, v0, a0/2, C, B, A]
        for elem in elements:
            xt += elem * t
            t *= dt
        
        scaledDifference = Rotation.from_rotvec(xt[:,None] * x0_axis).as_quat()
        result = np.array([multQuats(q1, q2) for q1, q2 in zip(scaledDifference, quat['t'])], dtype=np.float32)
        return Rotation.from_quat(result).as_matrix()



    def forwardKinematics(self, root):

#        stack = [(root, np.identity(4))]

        localFrame = self.getLocalFrame()
        localFrame[:3,3] = 0
        oriInLocalFrame = np.linalg.inv(localFrame) @ self.orientations[0]
        stack = [(root, oriInLocalFrame)]

        index = 0
        while stack:
            node, M = stack.pop()

            if 'End' in node.name:
                continue

            T = np.identity(4)
            T[:,-1] = np.append(node.offset, 1)

            R = self.orientations[index] if not node.isRoot else np.identity(4)
            index += 1
            
            M = M @ T @ R
            yield node.name, (M @ np.append(np.zeros(3), 1))[:3]

            for child in reversed(node.children):
                stack.append((child, M))


    def inverseKinematics(self, targetName, targetPoint, root):
        def l2norm(v):
            return np.sqrt(sum(v*v))

        def indexTrajToTargetNode(root, targetName):
            stack = [(root, [])]
            index = 0

            while stack:
                node, trajectory = stack.pop()

                trajectory += [index]
                index += 1
                if node.name == targetName:
                    return trajectory

                for child in reversed(node.children):
                    if 'End' not in child.name:
                        stack.append((child, trajectory.copy()))
            return None

        def normalize(vector):
            return vector / l2norm(vector)

        def limb(root, targetName, targetPoint):
            nodePositions = dict(self.forwardKinematics(root))
            names = list(nodePositions.keys())

            traj = indexTrajToTargetNode(root, targetName)
            ancestorName, parentName = names[traj[-3]], names[traj[-2]]

            a_gr = np.identity(4)
            for index in traj[:-2]:
                a_gr = a_gr @ self.orientations[index]
            b_gr = a_gr @ self.orientations[traj[-2]]

            a_lr = self.orientations[traj[-3]].copy()
            b_lr = self.orientations[traj[-2]].copy()

            ## Step 1
            t = targetPoint

            a,b,c = nodePositions[ancestorName], nodePositions[parentName], nodePositions[targetName] 
            homogeneous = np.ones((4,3)); homogeneous[:3,:] = np.vstack((a,b,c)).T
            inGlobal = (self.getLocalFrame() @ homogeneous)[:3].T
            a,b,c = inGlobal[0], inGlobal[1], inGlobal[2]

            t[1] = c[1] # Y up

            l_ab = l2norm(b-a)
            l_cb = l2norm(b-c)
            eps = 0.01 
            l_at = np.clip(l2norm(t-a), eps, l_ab + l_cb - eps)

            ac_ab_0 = np.arccos(np.clip(np.dot(normalize(c-a), normalize(b-a)), -1, 1))
            ba_bc_0 = np.arccos(np.clip(np.dot(normalize(a-b), normalize(c-b)), -1 ,1))

            ac_ab_1 = np.arccos(np.clip((l_cb*l_cb - l_ab*l_ab - l_at*l_at) / (-2*l_ab*l_at), -1, 1))
            ba_bc_1 = np.arccos(np.clip((l_at*l_at - l_ab*l_ab - l_cb*l_cb) / (-2*l_ab*l_cb), -1, 1))

            axis0 = normalize(np.cross(c-a, b-a))

            temp = (np.linalg.inv(a_gr) @ np.append(axis0, 1))[:3]
            r0 = exp(temp * (ac_ab_1 - ac_ab_0))

            temp = (np.linalg.inv(b_gr) @ np.append(axis0, 1))[:3]
            r1 = exp(temp * (ba_bc_1 - ba_bc_0))

            a_lr = a_lr @ r0
            b_lr = b_lr @ r1

            ## Step 2
            ac_at_0 = np.arccos(np.clip(np.dot(normalize(c-a), normalize(t-a)), -1, 1))
            axis1 = normalize(np.cross(c-a, t-a))
            axis1 = (np.linalg.inv(a_gr) @ np.append(axis1, 1))[:3]
            r2 = exp(axis1 * (ac_at_0))

            a_lr = a_lr @ r2
            
            result = Posture()
            result.setPosture(self.position, self.orientations.copy())
            result.orientations[traj[-3]] = a_lr
            result.orientations[traj[-2]] = b_lr
            return result


        def jacobian():
            pass

        return limb(root, targetName, targetPoint)


    def setPosture(self, pos, ori):
        self.position = np.array(pos, dtype=np.float32)
        self.orientations = np.array(ori, dtype=np.float32)

