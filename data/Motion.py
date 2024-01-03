from data.Skeleton import Skeleton
from data.Posture import Posture

import numpy as np
import scipy.ndimage
import re

import util.bvhParser as parser

class Motion():
    def __init__(self, bvhpath=None, targetFPS=30):
        if not bvhpath:
            return

        self.name = bvhpath.split('/')[-1].split('.')[-2]  
        print('importing ', self.name, '...')

        with open(bvhpath, 'r') as bvhFile:
            text = bvhFile.read()
            skeletonInText, motionInText = text.split('MOTION', 1)
            skeletonInText = re.sub('[{|\t+]', '', skeletonInText).split()
            self.skeleton = Skeleton(skeletonInText)

            frameTime, frames = parser.parseMotion(motionInText.split('\n'))

        try:
            with open(bvhpath + '.conL.txt', 'r') as contactfile:
                left = parser.parseContact(contactfile.read(), len(frames))
                left = self.downsizing(left, 1/frameTime, targetFPS)
            with open(bvhpath + '.conR.txt', 'r') as contactfile:
                right = parser.parseContact(contactfile.read(), len(frames))
                right = self.downsizing(right, 1/frameTime, targetFPS)
            contacts = list(zip(left, right))

        except Exception:
            contacts = None
            print('Contact file with a same name does not exist.')

        frames = self.downsizing(frames, 1/frameTime, targetFPS)
        self.postures = self.generatePostures(frames, contacts)
        self.frameTime = 1/targetFPS

        self.cnt = 0

#        self.smoothedTrajectory, self.smoothedDirections = self.smoothTrajectoryAndDirections(n=9, name='GAUSSIAN')
#        self.smoothedTrajectory, self.smoothedDirections = self.smoothTrajectoryAndDirections(n=9, name='UNIFORM')
#        self.smoothedTrajectory, self.smoothedDirections = self.smoothTrajectoryAndDirections(n=9, name=None)


    def downsizing(self, array, FPS, targetFPS):
        assert(FPS > targetFPS)
        step = int(FPS/targetFPS)
        return array[1::step]



    def copy(self):
        result = Motion()
        result.skeleton = self.skeleton
        result.frameTime = self.frameTime
        copyPoseList = []
        for pose in self.postures:
            copyPose = Posture()
            copyPose.setPosture(pose.position.copy(), pose.orientations.copy())
            copyPoseList.append(copyPose)
        result.postures = copyPoseList
        return result

    def generatePostures(self, frames, contacts):
        postures = []
        scale = self.skeleton.getScale()

        if contacts:
            assert(len(frames) == len(contacts))
            for frame, contact in zip(frames, contacts):
                lc, rc = contact
                postures.append(Posture(frame, lc, rc, scale))
        else:
            for frame in frames:
                postures.append(Posture(frame, None, None, scale))
        return postures

    def reset(self):
        self.cnt = 0

    def setToPosture(self, i):
        self.cnt = i
    def setToNextPosture(self):
        self.cnt += 1

    def timeWarping(self, ratio):
        def trim(ratio):
            assert(ratio > 1)
            self.postures = self.postures[::int(ratio)]

        def extend(ratio):
            assert(ratio > 0 and ratio < 1)
            ratio = int(1/ratio)

            newPostureLength = len(self.postures) * ratio - (ratio - 1)
            for i in range(0, newPostureLength - 1, ratio):
                for j in range(1, ratio):
                    self.postures.insert(i+j, self.postures[i].interpolate(self.postures[i+1], j/ratio))

        if ratio > 1:
            trim(ratio)
        elif ratio > 0:
            extend(ratio)
        else:
            print('Invalid time warp ratio.')

    def keyFrameWarping(self, targetPose, WARP_RANGE):
        def s(t):
            return np.sin(t * (np.pi / WARP_RANGE))

        for i, pose in enumerate(self.postures):
            if i >= WARP_RANGE:
                return

            ori = pose.interpolateOnlyOrientations(targetPose, s(i))
            pose.setPosture(pose.getPosition(), ori)

#    def stitchFrom(self, anotherMotion, STITCH_RANGE):
#        def s(t):
#            return t / STITCH_RANGE
#
#        fromPose = anotherMotion.getLastPosture()
#        toPose = self.postures[0]
#        stitchStartPoint = toPose.getPosition().copy() # will be overwritten
#
#        rootAlignVector, rootAlignMatrix = fromPose.rootAlignment(toPose)
#
#        for i, pose in enumerate(self.postures):
#            if i < STITCH_RANGE:
#                ori = fromPose.interpolateOnlyOrientations(pose, s(i))
#                ori[0] = pose.getOrientations()[0]
#            else:
#                ori = pose.getOrientations()
#            ori[0] = rootAlignMatrix @ ori[0]
#            pos = pose.stitchedPosition(stitchStartPoint, rootAlignVector, rootAlignMatrix)
#            pose.setPosture(pos, ori)
#
#        return rootAlignVector, rootAlignMatrix


    def blending(self, anotherMotion, BLEND_RANGE):

# UNIFORM BLENDING
#        def s(t):
#            return 0.5 * np.cos(np.pi / BLEND_RANGE * t) + 0.5

        def T(t, f1=2, f2=0.5):
            t2 = t*t
            t3 = t2*t

            h1 = -2*t3 + 3*t2
            h2 = t3 -2*t2 + t
            h3 = t3 - t2
            return h1 + h2*f1 + h3*f2

        copy1 = self.copy()
        copy2 = anotherMotion.copy()

        copy1.timeWarping(len(copy1.postures)/BLEND_RANGE) # walk * 0.5
        copy2.timeWarping(len(copy2.postures)/BLEND_RANGE) # sneak * 2

        result = []
        lastPosition = copy1.postures[-1].position
        lastOrientation = copy1.postures[-1].orientations[0]
        for i in range(1, BLEND_RANGE - 1):
            t = i / BLEND_RANGE
            left = copy1.postures[i]
            leftPrevFrame = copy1.postures[i-1].orientations[0].copy()
            leftPrevFrame[:3,3] = copy1.postures[i-1].position

            right = copy2.postures[i]
            rightPrevFrame = copy2.postures[i-1].orientations[0].copy()
            rightPrevFrame[:3,3] = copy2.postures[i-1].position

            pose = left.interpolate(right, T(t)) # a new posture
            pose.position = (np.linalg.inv(leftPrevFrame) @ np.append(left.position, 1)\
                    + np.linalg.inv(rightPrevFrame) @ np.append(right.position, 1))[:3] * 0.5
            pose.position += lastPosition
            pose.orientations[0] = (np.linalg.inv(leftPrevFrame) @ left.orientations[0])\
                @ (np.linalg.inv(rightPrevFrame) @ right.orientations[0]) @ lastOrientation

            lastPosition = pose.position # update
            lastOrientation = pose.orientations[0]

            result.append(pose)

        self.postures = result
        return self



    def getFrameTime(self):
        return self.frameTime
    def getSkeletonRoot(self):
        return self.skeleton.getRoot()

    def getPostures(self):
        return self.postures
    def getCurrentPosture(self):
        return self.postures[self.cnt]

    def getPrevPosture(self):
        return self.postures[max(self.cnt - 1, 0)]
    def getLastPosture(self):
        return self.postures[-1]
    def getFuturePosture(self, dt):
        if self.cnt + dt < len(self.postures):
            return self.postures[self.cnt + dt]
        return None

#    def getFutureSmoothedPositionAndDirection(self, dt):
#        assert(self.cnt + dt < len(self.smoothedTrajectory))
#        return self.smoothedTrajectory[self.cnt + dt], self.smoothedDirections[self.cnt + dt]

    def inFirstPosture(self):
        return self.cnt == 0
    def inLastPosture(self):
        return self.cnt == len(self.postures) - 1


    '''
    def smoothTrajectoryAndDirections(self, n, name):

        assert(n%2==1)
        half = int(n/2)

        # generate
        rootTrajectory, directions = [], []
        for p in self.postures:
            rootTrajectory.append(p.getPosition()) # Y up
            directions.append(p.getLocalFrame()[:3, 2])

        if name == None: 
            print('INFO:: No filter given to smooth motion feature')
            return rootTrajectory, directions

        # extend line
        line = np.array(rootTrajectory, dtype=np.float32)

        first = (line[0] - line[1]) * n + line[0]
        last = (line[-1] - line[-2])* n + line[-1]

        prev, after = [], []
        for i in range(0, half):
            t = (1/n) * i
            prev.append((1-t)*first + t*line[0])
            t += 1/n
            after.append((1-t)*line[-1] + t*last)
        rootTrajectory = np.concatenate((prev, line, after))

        # extend directions
        directions = [directions[0]]*half + directions + [directions[-1]]*half
        directions = np.array(directions, dtype=np.float32)


        if name == 'UNIFORM':
            # filter
            weights = np.array([1/n for _ in range(n)])

            # filter line
            x = scipy.ndimage.filters.convolve1d(rootTrajectory[:, 0], weights)[half : -half]
            zeros = np.zeros(x.shape)
            z = scipy.ndimage.filters.convolve1d(rootTrajectory[:, 2], weights)[half : -half]

            # filter directions
            sin = scipy.ndimage.filters.convolve1d(directions[:, 0], weights)[half : -half]
            cos = scipy.ndimage.filters.convolve1d(directions[:, 2], weights)[half : -half]

        elif name == 'GAUSSIAN':
            # NEED TO MANUALLY SPECIFY PARAMETERS
            s,t = 3,4

            x = scipy.ndimage.filters.gaussian_filter1d(rootTrajectory[:, 0],\
                sigma=s, truncate=t)[half : -half]
            zeros = np.zeros(x.shape)
            z = scipy.ndimage.filters.gaussian_filter1d(rootTrajectory[:, 2],\
                sigma=s, truncate=t)[half : -half]


            sin = scipy.ndimage.filters.gaussian_filter1d(directions[:, 0],\
                sigma=s, truncate=t)[half : -half]
            cos = scipy.ndimage.filters.gaussian_filter1d(directions[:, 2],\
                sigma=s, truncate=t)[half : -half]
        
        else:
            print('INFO:: Invalid filter name, try again.')
            quit()


        smoothedTrajectory = np.vstack((x,zeros,z)).T

        directions = np.vstack((sin, zeros, cos)).T
        size = np.linalg.norm(directions, axis=1)
        smoothedDirections = directions / size[:,None]
        return smoothedTrajectory, smoothedDirections

    '''
