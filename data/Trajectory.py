from util.matrixFormulas import *
from util.rotations import exp, log

from bisect import bisect_left

import numpy as np

FRAMETIME = 1/30
SCALE_FACTOR = 1/2

FUTURE_FEATURE_SECONDS = 1 # A second = FRAMETIME*3 = 3 future points

def qt_factory(inputType, FUTURE_TERM):
    # priority exists
    if inputType == "tracking":
        return LineQueryTrajectory(FUTURE_TERM)
    elif inputType == "direction":
        return DirectionQueryTrajectory(FUTURE_TERM)

    return FutureMotionQueryTrajectory(FUTURE_TERM)

class QueryTrajectory():
    def __init__(self, FUTURE_TERM):
        self.FUTURE_TERM = FUTURE_TERM

    def getFutureFeatureCount(self):
        nFrames = 1/FRAMETIME
        futurePointPerSecond = nFrames / self.FUTURE_TERM
        return int(futurePointPerSecond * FUTURE_FEATURE_SECONDS) # 3, 6, 9, ..

    # abstract method
    def calculate(self, obs, actor):
        raise NotImplementedError

    def format(self, points):
        queryTrajectory = []
        for p in points:
            queryTrajectory.extend([p[0], p[2], p[3], p[5]]) # Y-up
        # assert(len(queryTrajectory)==12)
        return np.array(queryTrajectory)

    def verboseFormat(self, formatted):
        # assert(len(formatted)==12)

        nPoints = self.getFutureFeatureCount()
        assert(len(formatted) == 4*nPoints)

        frames = []
        for i in range(0, 4*nPoints, 4):
            *position, sin, cos = formatted[i:i+4] # Z forward pi/2-theta

            # Y-up
            M = np.identity(4)
            M[:3,0] = np.array([cos, 0, -sin])
            M[:3,2] = np.array([sin, 0, cos])
            M[:3,3] = np.array([position[0],0,position[1]])

            frames.append(M)
        return frames


def futurePositionsScaledInTangentialFrame(localFuturePositions):
    if SCALE_FACTOR == 1:
        return localFuturePositions # do nothing

    a,b,c = localFuturePositions

    # tangential frame
    l = np.linalg.norm(c)
    if l < 0.001:
        l = 1

    axis_UP = np.array([0,1,0]) # Y UP
    axis_T = c/l # z
    axis_N = np.cross(axis_UP, axis_T) # x

    frame = np.stack((axis_N, axis_UP, axis_T)).T

    a = frame @ a
    b = frame @ b
    c = frame @ c

    # scale down
    a[2] *= SCALE_FACTOR
    b[2] *= SCALE_FACTOR
    c[2] *= SCALE_FACTOR

    # back to local frame
    a = frame.T @ a
    b = frame.T @ b
    c = frame.T @ c

    return a,b,c


class LineQueryTrajectory(QueryTrajectory):
    def __init__(self, FUTURE_TERM):
        super().__init__(FUTURE_TERM)

    def calculate(self, obs, network):

        lineLog, guideline, window, pastDirectionSegment, inputTrajectory, inputDirection, indices = obs
        assert(len(guideline) >= 2)

        predicted = True if network else False
        if predicted:
            start = list(zip(*(lineLog.T), *(pastDirectionSegment.T)))
            inputTrajectory = np.delete(inputTrajectory, 1, axis=1)
            directions = network.predict((start, inputTrajectory), window)

        guideline = list(guideline)

        count = self.getFutureFeatureCount()

        while len(guideline) < (1 + count):
            FT = self.FUTURE_TERM
            p1, p2 = guideline[-2:]
            v = p2 - p1
            guideline.append(p2 + v)

        target = []
        # targetDirections = []

        tangents = inputTrajectory[1:] - inputTrajectory[:-1]
        tangents = np.vstack((tangents, tangents[-1]))
        lengths = np.linalg.norm(tangents, axis=1)

        while len(indices) < (1 + count):
            indices = np.append(indices, indices[-1])

        for i in range(1, (1 + count)):
            if predicted:
                direction = np.insert(directions[i], 1, 0) # index, value
            else:
#                index=i
#                tangent = guideline[index] - guideline[index - 1]
#                tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
#                direction = tangent

                t = tangents[indices[i]] * [1,0,1]
                direction = t / np.linalg.norm(t)

                if inputDirection is not None: # has joystick input
                    forward = [0,0,1]
                    theta = np.arccos(inputDirection[-1])
                    if inputDirection[0] > 0:
                        theta = -theta
                    direction = rotationY(np.degrees(theta)) @ direction

            target.append(np.concatenate((guideline[i], direction)))
            # targetDirections.append(direction)

        # target = []
        # for i, p in enumerate(futurePositionsScaledInTangentialFrame(guideline[1:])):
            # target.append(np.concatenate((p, targetDirections[i])))

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

        count = self.getFutureFeatureCount()
        for dt in range(1, count + 1):
            dt *= FUTURE_TERM

            futurePose = motion.getFuturePosture(dt)
            if futurePose:
                futureLocalFrame = futurePose.getLocalFrame()

                diff = invLocalFrame @ futureLocalFrame
                pos, axis = diff[:3, 3], diff[:3, 2] # Z forward

                target.append(np.concatenate((pos, axis)))
            else:
                target.append(self.__eulerExpl__(motion, dt))
        

        # scale feature
        # p1,p2,p3 = futurePositionsScaledInTangentialFrame([p[:3] for p in target])
        # target[0][:3] = p1
        # target[1][:3] = p2
        # target[2][:3] = p3

        return target
    
    def __eulerExpl__(self, motion, dt):
        posture = motion.getCurrentPosture()
        prevPosture = motion.getPrevPosture()

        diff = np.linalg.inv(prevPosture.getLocalFrame()) @ posture.getLocalFrame()
        vel = diff[:3, 3]
        axis = exp(log(diff) * dt)[:3, 2] # Z forward
        return np.concatenate((vel * dt, axis))


