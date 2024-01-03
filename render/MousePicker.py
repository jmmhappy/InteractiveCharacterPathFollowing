import numpy as np
import time
from bisect import bisect_right

import scipy.ndimage.filters as filters

class MousePicker:
    def __init__(self, canvasSize):
        self.trajectory = []
        self.temp = []
        self.smoothTrajectory = []

        self.timeLog = []
        self.tempTimeLog = []
        self.initTime = 0

        self.scaled = False

        self.canvasWidth, self.canvasHeight = canvasSize
        
        # Store a copy for addIntersection()
        self.modelviewMatrix = None 
        self.rayWorld = None

    def pickRay(self, x, y, projMatrix, modelviewMatrix):
        self.modelviewMatrix = modelviewMatrix

        # Window Coordinates -> Normalized Device Coordinates(NDS)
        x = (2.0 * x) / self.canvasWidth - 1.0
        y = 1.0 - (2.0 * y) / self.canvasHeight
        rayNds = np.array([x,y,1]) # z=1 is a placeholder
        
        # NDS ->  Clip Coordinates
        rayClip = np.append(rayNds[:2], [-1.0, 1.0])
        # Clip Coordinates -> Eye Coordinates
        rayEye = np.linalg.inv(projMatrix) @ rayClip
        rayEye = np.append(rayEye[:2], [-1.0, 0.0]) # Non vector?
        # Eye Coordinates -> World Coordinates
        rayWorld = (np.linalg.inv(modelviewMatrix) @ rayEye)[:3]
        # Normalize
        rayWorld = rayWorld / np.sqrt(sum(rayWorld * rayWorld))

        self.rayWorld = rayWorld
        return rayWorld

    def getIntersection(self): # with y=0
        if self.rayWorld is None:
            return None

        rayWorld = self.rayWorld
        # Get Intersection with y=0 plane
        normalPlane = [0,1,0]
        camPos = (np.linalg.inv(self.modelviewMatrix.T)[3])[:3]
        d = camPos[1]
        t = -d/rayWorld[1] # intersection btw xz plane and ray

        if t <= 0:
            print("Intersect beyond the camera")
            return None
        elif d == 0:
            print("Ray perpendicular to the camera")
            return None

        # Store a trajectory
        insec = camPos + rayWorld*t
        return insec

    def addIntersection(self):

        if self.tempTimeLog == []:
            self.initTime = time.time()
        self.tempTimeLog.append(time.time() - self.initTime)

        insec = self.getIntersection()
        self.temp.append(insec)
     
    def getTimeLog(self):
        return self.timeLog

    def getTrajectoryOnMove(self):
        return self.temp

    def getTrajectory(self, smoothInput):
        if smoothInput:
            if len(self.smoothTrajectory) == 0:
                self.smoothTrajectory = self.smooth()
            return self.smoothTrajectory
        return self.trajectory

    def setTrajectory(self, t):
        self.temp = t

    def reset(self, clean=False):
        if clean:
            self.smoothTrajectory = []
            self.trajectory = []
            self.timeLog = []


    def isEmpty(self):
        return len(self.trajectory) == 0

    def adjust(self, pos, localControl):
        assert(pos is not None)

        self.startQueryIndex = 0

        if localControl:
            vector = np.array(pos - self.temp[0]) * [1,0,1]
        else:
            vector = np.zeros(3)

        self.trajectory = self.temp + vector
        self.timeLog = self.tempTimeLog

        self.smoothTrajectory = []
        self.temp = []
        self.tempTimeLog = []

    def scale(self, pos, controlflag, scaleflag):

        def lerp(p1, p2, t):
            return p1 + (p2-p1) * t

#        self.trajectory = np.load('simpleTrajectory.npy')
#        self.timeLog = np.load('simpleTimelog.npy')
#        self.temp = self.trajectory
#
#        np.save('simpleTrajectory', self.trajectory)
#        np.save('simpleTimelog', self.timeLog)

        points = np.delete(self.trajectory, 1, axis=1)
        timeLog = np.array(self.timeLog)
        
        assert(len(timeLog) == len(points))
        if self.scaled or not scaleflag:
            return

        # scale
        INTERVAL = 1/150
        start = timeLog[0]
        scaledPoints, scaledTimeLog = [], []

        for i in range(len(points) - 1):
            targets = np.arange(start, timeLog[i+1], INTERVAL)

            if len(targets) == 0:
                scaledPoints.append(points[i])
                continue

            for t in targets:
                ratio = (t - timeLog[i]) / (timeLog[i+1] - timeLog[i])
                #assert(0 <= ratio and ratio < 1)
                scaledPoints.append(lerp(points[i], points[i+1], ratio))
            scaledTimeLog.extend(targets)
            start = targets[-1] + INTERVAL
            #assert(timeLog[i+1] < start)


        if not controlflag: # global control enabled
            pos = pos[[0,2]] # Y up
            bridge = []
            seg = np.array(scaledPoints[:30])
            meanVelocity = np.mean(np.linalg.norm(seg[1:]-seg[:-1], axis=1))
            distance = np.linalg.norm(pos - seg[0])
            for t in np.arange(0,1,meanVelocity/distance):
                bridge.append((1-t)*pos + t*seg[0])
            scaledPoints = np.vstack((bridge, scaledPoints))


        self.trajectory = np.insert(scaledPoints, 1, 0, axis=1)
        self.timeLog = np.asarray(scaledTimeLog)
        

    def smooth(self):
        if len(self.trajectory) == 0:
            return []
        points = np.asarray(self.trajectory)

        sig = 3
        half = 9 # filter width = 2*half + 1

        # Extend
        first = (points[0] - points[1]) * half + points[0]
        last = (points[-1] - points[-2]) * half + points[-1]

        chunkFront, chunkBack = [], []
        for i in range(half):
            t = (1/half) * i
            chunkFront.append((1-t)*first + t*points[0])
            t += 1/half
            chunkBack.append((1-t)*points[-1] + t*last)
        points = np.concatenate((chunkFront, points, chunkBack))

        # Filter
        x = filters.gaussian_filter1d(points[:,0], sigma=sig, truncate=half)[half:-half]
        z = filters.gaussian_filter1d(points[:,2], sigma=sig, truncate=half)[half:-half]
        points = np.stack((x, np.zeros(x.shape), z)).T

        return points

