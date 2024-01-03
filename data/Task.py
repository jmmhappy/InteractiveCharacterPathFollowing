import numpy as np
from abc import *

from bisect import bisect_left
from data.Trajectory import qt_factory, QueryTrajectory

from data.Trajectory import FutureContactQueryTrajectory

maxVelocity = 3 # updated by slide bar, render.CanvasPanel
FUTURE_TERM = 10 # assume 10Hz*3

class TaskFactory():
    def __init__(self):
        pass
    def create(self, pastRootTrajectory, pastDirections, userInput):
        if userInput is None:
            return None

        if userInput['name'] == 'position':
            return PositionTask(userInput['position'], userInput['obstacles'])
        elif userInput["name"] == "joystick":
            return JoystickTask(userInput["joystick"])
        elif userInput['name'] == 'direction':
            return DirectionTask(userInput['direction'])
        elif userInput['name'] == 'trajectory':
            return TrackingTask(pastRootTrajectory, userInput['trajectory'],\
                pastDirections, userInput['amplifyCurve'])
        return DefaultTask() 


class Task(metaclass=ABCMeta):
    def __init__(self):
        self.name = None 
        self.obstacleMap = None
        self.queryTrajectory = None
        self._meta = {}

    @abstractmethod
    def getObservation(self, currentState):
        return

    def isReinforcementLearning(self):
        assert(self.name is not None)
        return self.queryTrajectory == None

    def action(self, observation, actor):
        assert(self.queryTrajectory)
        qt = self.queryTrajectory
        ft = qt.calculate(observation, actor)
        ft = qt.format(ft)
        return ft

    def getLogInFrames(self, localFrame, action):
        return [localFrame @ m for m in FutureContactQueryTrajectory.verboseFormat(None, action)]
        # return [localFrame @ m for m in QueryTrajectory.verboseFormat(None, action)]

class DefaultTask(Task):
    def __init__(self):
        super().__init__()
        self.name = "default"
        self.queryTrajectory = qt_factory(None, FUTURE_TERM)

    def getObservation(self, _, __):
        return None


class PositionTask(Task):
    def __init__(self, pos, obstacles):
        super().__init__()
        assert(len(pos) == 2)
        self.name = 'position'
        self._meta['position'] = pos

        self.obstacleMap = obstacles
        if obstacles:
            self.previousTerrain = obstacles.getLocalMap(np.identity(4)) # initial value
        else:
            self.previousTerrain = None

    def getObservation(self, feature, localFrame): # runtime
        x,z = self._meta['position']
        goal = np.linalg.inv(localFrame) @ np.array([x,0,z,1])
        character = np.append(feature[:15], goal[[0,2]])

        if self.obstacleMap:
            currentTerrain = self.obstacleMap.getLocalMap(localFrame)
            terrain = np.concatenate((self.previousTerrain, currentTerrain))

            self.previousTerrain = currentTerrain # update
            return {'character':character, 'terrain':terrain}

        return character

    def action(self, observation, actor):
        if actor is None:
            print("No policy network to query position.")
            return None
        
        return actor.action(observation)
        

class JoystickTask(Task):
    def __init__(self, jsInput):
        super().__init__()
        self.name = 'joystick'
        self._meta['targetVelocity'] = jsInput
    
    def getObservation(self, feature, localFrame):
        # used to update query at MotionStream.matchByQuery()
        v_x, v_z = self._meta["targetVelocity"]
        goal = np.linalg.inv(localFrame[:3,:3]) @ np.array([v_x,0,v_z])
        character = np.append(feature[:15], goal[[0,2]])
        return character 

    def action(self, observation, actor):
        if actor is None:
            print("No policy network to query target velocity.")
            return None
        return actor.action(observation)

class DirectionTask(Task):
    def __init__(self, keyflags):
        super().__init__()
        assert(type(keyflags) == dict)
        self.name = 'direction'
        self.queryTrajectory = qt_factory(self.name, FUTURE_TERM)
        self._meta = keyflags

    def getObservation(self, _, __):
        return self._meta



class TrackingTask(Task):
    def __init__(self, past, future, directions, amplifying):
        super().__init__()
        self.name = 'tracking'
        self.queryTrajectory = qt_factory(self.name, FUTURE_TERM)

        if len(future) <= 2:
            queryPoints = future + (past[-1]*np.array([1,0,1]) - future[0]) 
            cornerPoint = None
            indices = [0]
        else:
            indices, extra = self._getTargetIndices(future)
            queryPoints, cornerPoint = self._amplifyCurve(amplifying, future, indices, extra['tangents'])

        assert(len(queryPoints) <= 4)

        self.indices = indices 

        window = 120

        self._meta = {
            'past':past,
            'future':future,
            'window':window,
            'direction_log':directions[-int(window/2):],

            'cornerPoint':cornerPoint,
            'queryPoints':queryPoints,
            'limited':indices[-1] != 0,
        }


    def getObservation(self, _, localFrame):

        past, queryPoints = self._logAndGuidelineInLocalFrame(localFrame)
        direction_log = self._directionInLocalFrame(localFrame)
        inputTrajectory = self._futurePointsInLocalFrame(localFrame)

        pastTrajectory = np.delete(past, 1, axis=1)
        return pastTrajectory, queryPoints, self._meta['window'], direction_log, inputTrajectory, self.indices
   
    def _futurePointsInLocalFrame(self, localFrame):
        inv = np.linalg.inv(localFrame[:3,:3])
        points = self._meta['future']
        points = inv @ points.T
        return points.T


    def _directionInLocalFrame(self, localFrame):
        inv = np.linalg.inv(localFrame)

        directions = np.array(self._meta['direction_log'])
        directions = inv @ directions.T

        return directions[[0,2]].T

    def _logAndGuidelineInLocalFrame(self, localFrame):
        past = self._meta['past']
 #       future = self._meta['future']
        window = self._meta['window']

        def _projection(t_3d): # 3d-trajectory
            points = np.insert(t_3d, 3, 1, axis=1)
            points = (inv @ points.T)[:3]
            return points.T

        _past = past[-int(window/2):]
        _future = self._meta['queryPoints']

#        if window >= 2:
#            _past = past[-int(window/2):]
#            _future = future[:int(window/2)]
#        else:
#            _past, _future = past, future

#        trajectory = np.vstack((_projection(_past), _projection(_future)))
        inv = np.linalg.inv(localFrame)
        return _projection(_past), _projection(_future)

    def _getTargetIndices(self, points):

        indices = np.array([i*10 for i in range(4)])

        segment = points[:31]
        deltas = segment[1:] - segment[:-1]
        segLengths = np.linalg.norm(deltas, axis=1)

        tangents = deltas / segLengths[:, None]
        tangents[np.isnan(tangents)] = 0

        # len(points) == len(tangents) == len(segLengths)
        tangents = np.vstack((tangents, tangents[-1])) # last two has same tangents
        segLengths = np.insert(segLengths, 0, 0) # segment length of index 0 is 0
        segLengths = np.cumsum(segLengths)

        # <=3 point query, velocity of m/s
        maxMovement = maxVelocity * (1/3) * (len(indices) - 1)
        queryLength = min(maxMovement, segLengths[-1])

        # limit velocity
        maxIndex = bisect_left(segLengths, queryLength)
        queryScale = min(1, maxIndex / indices[-1])
        indices = np.array(indices * queryScale, int)

        return indices, {"tangents":tangents, "queryLength":queryLength}


    def _amplifyCurve(self, flag, points, indices, tangents):
        assert(len(points) >= 2)

        # CASE: len(points) <= 10
        if len(indices) == 1:
            return points[[0,-1]], None 

        # what to return
        result = [points[0]]

        # CASE: User do not want amplifying OR Curve is too smooth to amplify
        theta = np.arccos(np.clip(np.dot(tangents[0], tangents[indices[-1]]), -1, 1))
        if (not flag) or (theta < np.pi/2):
            for i in indices[1:]:
                result.append(points[i])
            return result, None 
        
        # CASE: Amplify strong curve, with enough points
        amplified = False
        corner = None
        result = [points[0]]
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i+1]
            if not amplified:
                theta = np.arccos(np.clip(np.dot(tangents[start], tangents[end]), -1, 1))
                if theta > np.pi/2: # sharp corner exists
                    cornerInDegrees = []
                    for j in range(start, end):
                        theta = np.arccos(np.clip(np.dot(tangents[j], tangents[j+1]), -1, 1))
                        cornerInDegrees.append(theta)
                    corner = start + np.argmax(cornerInDegrees)

                    k = (end - start) / (corner - start)
                    if np.isnan(k) or np.isinf(k):
                        p = points[end]
                    else:
                        p = result[-1] + k*(points[corner] - result[-1])
                        amplified = True
                else:
                    p = points[end]

            elif amplified:
                p = result[-1] + (points[end] - points[start])

            p[np.isnan(p)] = 0
            result.append(p)
        return result, corner
            

