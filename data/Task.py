import numpy as np
from abc import *

from bisect import bisect_left

maxVelocity = 3 # updated by slide bar, render.CanvasPanel

class TaskFactory():
    def __init__(self):
        pass
    def create(self, pastRootTrajectory, pastDirections, userInput):
        if userInput is None:
            return None

        if userInput['name'] == 'position':
            return PositionTask(userInput['position'], userInput['obstacles'])
        elif userInput['name'] == 'direction':
            return DirectionTask(userInput['direction'])
        elif userInput['name'] == 'trajectory':
            return TrackingTask(pastRootTrajectory, userInput['trajectory'],\
                pastDirections, userInput['amplifyCurve'], userInput['startFromNearest'],\
                userInput['limitedUpdate'], userInput['joystick'])
        return None


class Task(metaclass=ABCMeta):
    def __init__(self):
        self.name = None
        self.obstacleMap = None
        self._meta = {}

    @abstractmethod
    def getObservation(self, currentState):
        return

class PositionTask(Task):
    def __init__(self, pos, obstacles):
        super().__init__()
        assert(len(pos) == 2)
        self.name = 'position'
        self._meta['position'] = pos
        self.obstacleMap = obstacles

    def getObservation(self, feature, localFrame): # runtime
        x,z = self._meta['position']
        goal = np.linalg.inv(localFrame) @ np.array([x,0,z,1])
        character = np.append(feature[:15], goal[[0,2]])

        if self.obstacleMap:
            terrain = self.obstacleMap.getLocalMap(localFrame)
            return {'character':character, 'terrain':terrain}

        return character

class DirectionTask(Task):
    def __init__(self, keyflags):
        super().__init__()
        assert(type(keyflags) == dict)
        self.name = 'direction'
        self._meta = keyflags

    def getObservation(self, _, __):
        return self._meta


class TrackingTask(Task):
    def __init__(self, past, userTrajectory, directions,\
            amplifying, startFromNearest, limitedUpdate, joystick):
        super().__init__()
        self.name = 'tracking'
        
        self._meta = {}
        window = 120
        self._meta['past'] = past
        self._meta['user_trajectory'] = userTrajectory
        self._meta['window'] = window
        self._meta['direction_log'] = directions[-int(window/2):]
        self._meta['joystick'] = joystick

        # options
        self._meta['amplifying'] = amplifying
        self._meta['start_from_nearest'] = startFromNearest
        self._meta['limited_update'] = limitedUpdate


    def setFutureTrajectory(self, index, position=None, setLog=False):

        def parseFutureTrajectory(index, position):
            trajectory = self._meta['user_trajectory'] # includes past guideline
            if len(trajectory) < 2:
                return trajectory

            if position is None:
                position = self._meta['past'][-1]

            if self._meta['start_from_nearest']:
                dists = np.linalg.norm(trajectory - position, axis=1)
                return trajectory[np.argmin(dists):]

            MAX_MOVEMENT = 10
            if self._meta['limited_update']:
                seg = trajectory[index : index + MAX_MOVEMENT+1]
                dists = np.linalg.norm(seg - position, axis=1)
                minimum = np.argmin(dists)

                index = index + minimum
            else:
                index += 1

            index = min(len(trajectory) - 2, index)
            return index


        past = self._meta['past']
        amplifying = self._meta['amplifying']

        currentIndex = parseFutureTrajectory(index, position)
        future = self._meta['user_trajectory'][currentIndex:]

        if len(future) <= 2:
            queryPoints = future + (past[-1]*np.array([1,0,1]) - future[0])
            cornerPoint = None
            indices = [0]
        else:

            indices, extra = self._getTargetIndices(future)
            queryPoints, cornerPoint = self._amplifyCurve(amplifying, future, indices, extra['tangents'])

            # advance index at sharp corners
            if cornerPoint is not None:
                currentIndex += 2
                future = self._meta['user_trajectory'][currentIndex:]
                indices, extra = self._getTargetIndices(future)
                queryPoints, cornerPoint = self._amplifyCurve(amplifying, future, indices, extra['tangents'])


        # assert(len(queryPoints) <= 4)

        if setLog:
            self.indices = indices
            self._meta['future'] = future 
            self._meta['cornerPoint'] = cornerPoint
            self._meta['queryPoints'] = queryPoints
            self._meta['limited'] = indices[-1] != 0

        return currentIndex


    def getObservation(self, _, localFrame):

        past, queryPoints = self._logAndGuidelineInLocalFrame(localFrame)
        direction_log = self._directionInLocalFrame(localFrame)
        inputTrajectory = self._futurePointsInLocalFrame(localFrame)

        pastTrajectory = np.delete(past, 1, axis=1)
        return pastTrajectory, queryPoints, self._meta['window'], direction_log, inputTrajectory, self._meta['joystick'], self.indices

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
        maxIndex = indices[-1] # of future trajectory

        segment = points[: maxIndex + 1]
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
        maxIndex = bisect_left(segLengths, queryLength) # of velocity-limited future trajectory
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

        CORNER_LIMIT = np.radians(80)

        # CASE: User do not want amplifying OR Curve is too smooth to amplify
        hasCorner = False
        stack_theta = []
        for i in range(len(indices) - 1):
            theta = np.arccos(np.clip(np.dot(tangents[indices[i]], tangents[indices[i+1]]), -1, 1))
            stack_theta.append(theta)
            if (theta > CORNER_LIMIT and not hasCorner):
                hasCorner = True

        if (not flag) or (not hasCorner):
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
                # theta = np.arccos(np.clip(np.dot(tangents[start], tangents[end]), -1, 1))
                theta = stack_theta[i]

                if theta > CORNER_LIMIT: # sharp corner exists
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
            

