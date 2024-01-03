from util.matrixFormulas import *
from util.rotations import exp, log

from bisect import bisect_left

import numpy as np

FRAMETIME = 1/30

def qt_factory(inputType, FUTURE_TERM):
    # priority exists
    if inputType == "tracking":
        return LineQueryTrajectory(FUTURE_TERM)
    elif inputType == "direction":
        return DirectionQueryTrajectory(FUTURE_TERM)

    # return FutureMotionQueryTrajectory(FUTURE_TERM)
    return FutureContactQueryTrajectory(None)

class QueryTrajectory():
    def __init__(self, FUTURE_TERM):
        self.FUTURE_TERM = FUTURE_TERM

    # abstract method
    def calculate(self, obs, actor):
        raise NotImplementedError

    def format(self, points):
        queryTrajectory = []
        for p in points:
            queryTrajectory.extend([p[0], p[2], p[3], p[5]]) # Y-up
        assert(len(queryTrajectory)==12)
        return np.array(queryTrajectory)

    def verboseFormat(self, formatted):
        assert(len(formatted)==12)

        frames = []
        for i in range(0, 12, 4):
            *position, sin, cos = formatted[i:i+4] # Z forward pi/2-theta

            # Y-up
            M = np.identity(4)
            M[:3,0] = np.array([cos, 0, -sin])
            M[:3,2] = np.array([sin, 0, cos])
            M[:3,3] = np.array([position[0],0,position[1]])

            frames.append(M)
        return frames


class FutureContactQueryTrajectory(QueryTrajectory):
    def __init(self):
        super().__init__(None)
        leftContactPose = None
        rightContactPose = None

    def calculate(self, motion, _actor=None):
        localFrame = motion.getCurrentPosture().getLocalFrame() # 4x4

        # left
        pose, deltaTime = motion.getNextLeftContactPosture()
        positions = dict(pose.forwardKinematics(motion.getSkeletonRoot(), footOnly=True))
        frame = np.linalg.inv(localFrame) @ pose.getLocalFrame()
        footPosition = (frame @ np.append(positions["LeftFoot"], 1))[[0,2]]
        # footDirection = (frame @ np.append(positions["LeftToeBase"] - positions["LeftFoot"], 1))[[0,2]]
        footDirection = positions["LeftToeBase"] - positions["LeftFoot"]
        footDirection = (frame[:3,:3] @ footDirection)[[0,2]]
        footDirection /= np.linalg.norm(footDirection)
        
        ft = np.concatenate(([deltaTime], footPosition, footDirection))

        # right
        pose, deltaTime = motion.getNextRightContactPosture()
        positions = dict(pose.forwardKinematics(motion.getSkeletonRoot(), footOnly=True))
        frame = np.linalg.inv(localFrame) @ pose.getLocalFrame()
        footPosition = (frame @ np.append(positions["RightFoot"], 1))[[0,2]]
        footDirection = positions["RightToeBase"] - positions["RightFoot"]
        footDirection = (frame[:3,:3] @ footDirection)[[0,2]]
        footDirection /= np.linalg.norm(footDirection)
        
        ft = np.concatenate((ft, [deltaTime], footPosition, footDirection))

        return ft

    def format(self, points):
        assert(len(points) == 10)
        return points

    def verboseFormat(self, formatted):
        _, xL, zL, sinL, cosL, _, xR, zR, sinR, cosR = formatted

        leftFrame = np.array([\
            [ cosL, 0,sinL,xL],\
            [    0, 1,   0, 0],\
            [-sinL, 0,cosL,zL],\
            [0,0,0,1]])
        rightFrame = np.array([\
            [ cosR, 0,sinR,xR],\
            [    0, 1,   0, 0],\
            [-sinR, 0,cosR,zR],\
            [0,0,0,1]])

        return [leftFrame, rightFrame]



class LineQueryTrajectory(QueryTrajectory):
    def __init__(self, FUTURE_TERM):
        super().__init__(FUTURE_TERM)

    def calculate(self, obs, network):

        lineLog, guideline, window, pastDirectionSegment, inputTrajectory, indices= obs
        assert(len(guideline) >= 2)

        predicted = True if network else False
        if predicted:
            start = list(zip(*(lineLog.T), *(pastDirectionSegment.T)))
            inputTrajectory = np.delete(inputTrajectory, 1, axis=1)
            directions = network.predict((start, inputTrajectory), window)

#            indices = int(window/2) + np.array([0, 9, 19, 29])
#            print('indices', indices)
#            print()
#            directions = directions[indices]

        guideline = list(guideline)

        while len(guideline) < 4:
            FT = self.FUTURE_TERM
            p1, p2 = guideline[-2:]
            v = p2 - p1
            guideline.append(p2 + v)

        target = []

        tangents = inputTrajectory[1:] - inputTrajectory[:-1]
        tangents = np.vstack((tangents, tangents[-1]))
        lengths = np.linalg.norm(tangents, axis=1)

        while len(indices) < 4:
            indices = np.append(indices, indices[-1])

        for i in [1,2,3]:
            if predicted:
                direction = np.insert(directions[i], 1, 0) # index, value
            else:
#                index=i
#                tangent = guideline[index] - guideline[index - 1]
#                tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
#                direction = tangent

                x,_,z=tangents[indices[i]] / (lengths[indices[i]] + 1e-9)
                direction = [x,0,z]

            target.append(np.concatenate((guideline[i], direction)))
        return target


class DirectionQueryTrajectory(QueryTrajectory):
    def __init__(self, FUTURE_TERM):
        super().__init__(FUTURE_TERM)

    def calculate(self, keyFlags, _actor):

        FUTURE_TERM = self.FUTURE_TERM
        target = []

        for dt in [FUTURE_TERM, FUTURE_TERM*2, FUTURE_TERM*3]:
            dt *= FRAMETIME 
            eulerTrajectory = self.__eulerExpl__(dt, keyFlags)
            target.append(eulerTrajectory)
        
        return target

    def __eulerExpl__(self, dt, keyflags):

        if keyflags['left']:
            return np.array([0.6*dt,0,0,1,0,0])
        elif keyflags['right']:
            return np.array([-0.6*dt,0,0,-1,0,0])
        elif keyflags['up']:
            return np.array([0,0,1.5*dt,0,0,1])
        elif keyflags['down']:
            return np.array([0,0,-1.5*dt,0,0,1])

        print('Invalid operation: data/Trajectory.KeyBoardQueryTrajectory')
        return None



class FutureMotionQueryTrajectory(QueryTrajectory):
    def __init__(self, FUTURE_TERM):
        super().__init__(FUTURE_TERM)

    def calculate(self, motion, _actor=None):
        FUTURE_TERM = self.FUTURE_TERM
        pose = motion.getCurrentPosture()

        target = []

        # The resulting values are relative, so just use db original values.
        localFrame = pose.getLocalFrame()
        invLocalFrame = np.linalg.inv(localFrame)

        for dt in [FUTURE_TERM, FUTURE_TERM*2, FUTURE_TERM*3]:

            futurePose = motion.getFuturePosture(dt)
            if futurePose:

                futureLocalFrame = futurePose.getLocalFrame()

                diff = invLocalFrame @ futureLocalFrame
                pos, axis = diff[:3, 3], diff[:3, 2] # Z forward

                target.append(np.concatenate((pos, axis)))
            else:
                target.append(self.__eulerExpl__(motion, dt))
        
        return target
    
    def __eulerExpl__(self, motion, dt):
        posture = motion.getCurrentPosture()
        prevPosture = motion.getPrevPosture()

        diff = np.linalg.inv(prevPosture.getLocalFrame()) @ posture.getLocalFrame()
        vel = diff[:3, 3]
        axis = exp(log(diff) * dt)[:3, 2] # Z forward
        return np.concatenate((vel * dt, axis))
