import numpy as np
from scipy.spatial import KDTree

from data.Motion import Motion
from data.Posture import Posture
from data.Trajectory import qt_factory 
from data.Task import TaskFactory
from motion.MotionMatcher import MotionMatcher

from bisect import bisect_right
from util.rotations import log

taskFactory = TaskFactory()
LEFT_FOOT = 'LeftFoot'
RIGHT_FOOT = 'RightFoot'

class MotionStream:
    def __init__(self, data, xPos=0):
        self.motions = data['motions']
        if xPos == 0:
            self.startXPos = (0,0)
#            self.startXPos = (-10,10)
#        elif xPos == 1:
#            self.startXPos = (-10,10)
#        elif xPos == 2:
#            self.startXPos = (-10,-10)

        postures = []
        for m in self.motions:
            postures.extend(m.getPostures())
        self.postures = np.array(postures)
        self.lastPostureLengths = np.cumsum([len(m.getPostures()) for m in self.motions])

        self.totalPosturesCount = data['total']
        self.motionMatcher = data['motionMatcher']

        self.UPDATE_TERM = 6
        self.useNetwork = False 

        self.playReset()

    def playReset(self, random=False):
        self.currentIndex = np.random.randint(0, self.totalPosturesCount) if random else 0

        self.lastQueriedFutureTrajectory = None
        self.lastUpdated = 0

        initPose = self.postures[0]

        # x,z=0,0
        x, z = self.startXPos

        poseFromOrigin = Posture()
        poseFromOrigin.setPosture(np.array([x, initPose.getPosition()[1], z]), [np.identity(4)])

        rootAlignVector, rootAlignMatrix = poseFromOrigin.rootAlignment(self.postures[self.currentIndex])

        self.futureTrajectoryIndex = 0


        global LEFT_FOOT, RIGHT_FOOT
        skel = self.motions[0].getSkeletonRoot()
        nodePos = dict(initPose.forwardKinematics(skel))
        self.previousContacts = {
            'left':(False, nodePos[LEFT_FOOT], 0, None),
            'right':(False, nodePos[RIGHT_FOOT], 0, None)
        }

        self.log = {
            'root':[np.array([0,0,0])],
            'direction':[np.array([0,0,1,1])],
            'frame number':[self.currentIndex],
            'posture':[self.postureRootAlignment(self.postures[self.currentIndex], self.postures[self.currentIndex].getPosition(), rootAlignVector, rootAlignMatrix)],

            'query':[[]],

            'taskInfo':[{
                'cornerPoint':None,
                'queryPoints':[], 
                'limited':False,
                'future':[]}],

#            's_posture':[self.postureRootAlignment(self.postures[0])],
#            'contacts':[(self.previousContacts['left'], self.previousContacts['right'])],
        }


    def poseUpdate(self, frameNum, userInput, action=None, stitch=True):

        def needMotionMatching():
            return frameNum > 0

        global taskFactory

        STITCH_RANGE = 6
        motion = self.getCurrentMotion()

        rootAlignVector, rootAlignMatrix = np.zeros(3), np.identity(4)

        if self.motionMatcher is not None and needMotionMatching():

            task = taskFactory.create(self.log['root'], self.log['direction'], userInput)
            if task is None:
                return [self.currentIndex]

            poseBeforeQuery = self.log['posture'][-1]

            self.currentIndex = self.matchByQuery(frameNum, task, motion, action, userInput["extendedSearch"])
            poseFromDB = self.postures[self.currentIndex]
            rootAlignVector, rootAlignMatrix = poseBeforeQuery.rootAlignment(poseFromDB)

            stitchStartPoint = poseFromDB.getPosition()

            # calculate the difference between m2^0 and m1^N
            poseAligned = self.postureRootAlignment(poseFromDB, stitchStartPoint, rootAlignVector, rootAlignMatrix)
            p_diff = poseBeforeQuery.getPosition()[1] - poseAligned.getPosition()[1]
            o_diff = [log(o2.T @ o1) for o1, o2 in zip(poseBeforeQuery.getOrientations(), poseAligned.getOrientations())]
            difference = (p_diff, o_diff)

            indices = []
            for i in range(STITCH_RANGE):
                poseFromDB, self.currentIndex = self.nextPosture()
                if stitch:
                    poseToRender = self.postProcess(poseFromDB, stitchStartPoint,\
                            rootAlignVector, rootAlignMatrix, difference, (i+1)/STITCH_RANGE)
                else:
                    poseToRender = self.postProcess(poseFromDB, stitchStartPoint, rootAlignVector, rootAlignMatrix)

                if task and task.name == 'tracking':
                    self.updateLog(self.currentIndex, poseToRender, task._meta)
                else:
                    self.updateLog(self.currentIndex, poseToRender, None)
                indices.append(self.currentIndex)
            return indices

        poseFromDB, self.currentIndex = self.nextPosture()
        poseToRender = self.postProcess(poseFromDB, self.log['root'][-1], rootAlignVector, rootAlignMatrix)

        self.updateLog(self.currentIndex, poseToRender, None)
        return [self.currentIndex]


    def matchByQuery(self, frameNum, task, motion, action, extendedSearch):
        assert(self.log['frame number'][-1] == frameNum)

        pose = self.log['posture'][-1]
        localFrame = pose.getLocalFrame()

        query = self.motionMatcher.getFeature(frameNum).copy() # normalized
        network = self.motionMatcher.network if self.useNetwork else None

        assert(task is not None)

        if task.name == 'tracking':
            self.futureTrajectoryIndex = task.setFutureTrajectory(self.futureTrajectoryIndex, None, True)

        obs = task.getObservation(query, localFrame)
        qt = qt_factory(task.name, self.motionMatcher.FUTURE_TERM)

        if action is not None:
            assert(self.useNetwork is True)
            action = self.motionMatcher.denormalize(action)

            action[2:4] /= np.linalg.norm(action[2:4])
            action[6:8] /= np.linalg.norm(action[6:8])
            action[10:12] /= np.linalg.norm(action[10:12])

            query[-12:] = self.motionMatcher.normalize(action)

            ft = action
            _, nextFrameNum = self.motionMatcher.findByQuery(query)
        else:
            ft = qt.calculate(obs, network)
            ft = qt.format(ft)
            query[15:] = self.motionMatcher.normalize(ft)


            # import time
            # now = time.time()

            flag = extendedSearch["flag"]
            K = extendedSearch["K"]
            L = extendedSearch["L"]

            if not flag or (K == 1 and L == 1):
                # k = 1, l = 1, original motion matching
                _, nextFrameNum = self.motionMatcher.findByQuery(query)

            else:
                _, nextFrameNum = self.queryLoop(query, task, qt, self.futureTrajectoryIndex, pose, K, L)

            # print('k=%d, L=%d'%(K,L), time.time() - now)

           
        self.lastQueriedFutureTrajectory = [localFrame @ m for m in qt.verboseFormat(ft)]

        if frameNum == nextFrameNum:
            return frameNum + 1
        return nextFrameNum



    def queryLoop(self, query, task, qt, futureIndex, pose, K, L, loop=1):
        minDist = np.float("inf")
        nextFrameNum = 0

        if L == loop:
            return self.motionMatcher.findByQuery(query, 1)

        for d, f in zip(*self.motionMatcher.findByQuery(query, K)):
            nextQuery, nextFutureIndex, nextPose = self.findNextQuery(f, task, qt, futureIndex, pose)

            dist = d + self.queryLoop(nextQuery, task, qt, nextFutureIndex, nextPose, K, L, loop + 1)[0] # only distance
            if minDist > dist:
                minDist = dist
                nextFrameNum = f
        return minDist, nextFrameNum


    def findNextQuery(self, frame, task, qt, futureIndex, pose):
        rootAlignVector, rootAlignMatrix = pose.rootAlignment(self.postures[frame])
        stitchStartPoint = self.postures[frame].getPosition()

        # calculate frame+6th position & root rotation
        poseAligned = self.postureRootAlignment(self.postures[frame + 6], stitchStartPoint,\
                rootAlignVector, rootAlignMatrix)
        localframe = poseAligned.getLocalFrame()

        futureIndex = task.setFutureTrajectory(futureIndex, localframe[:3,3])
        obs = task.getObservation(None, localframe)
        ft = qt.format(qt.calculate(obs, None))

        query = self.motionMatcher.getFeature(frame + 6).copy()
        query[15:] = ft
        return query, futureIndex, poseAligned


    def postProcess(self, poseFromDB, stitchStartPoint, rootAlignVector, rootAlignMatrix, difference=None, stitchRatio=None):
        # Root align
        skeleton = self.getSkeletonRoot()
        pose = self.postureRootAlignment(poseFromDB, stitchStartPoint, rootAlignVector, rootAlignMatrix)
        contacts = poseFromDB.getFootContacts()
        footNames = (LEFT_FOOT, RIGHT_FOOT)

        # Stitch
        if stitchRatio:
            pose.interpolate(difference, stitchRatio) 

        if contacts[0] == None:
            return pose

        # IK catch foot sliding
        CONTACT_STITCH_RANGE = 5

        nodePos = dict(pose.forwardKinematics(skeleton))

        for i, key in enumerate(['left', 'right']):
            prevContact, startPosition, count, difference = self.previousContacts[key]

            if not prevContact and contacts[i]:
                count = 0 # new phase

                p = nodePos[footNames[i]]
                startPosition = (pose.getLocalFrame() @ np.append(p,1))[:3]

            if prevContact and contacts[i]:
                pose = pose.inverseKinematics(footNames[i], startPosition, skeleton)

            if prevContact and not contacts[i]:
                assert(count == 0)
                ikpose = pose.inverseKinematics(footNames[i], startPosition, skeleton)
                difference = [log(o2.T @ o1) for o1, o2 in zip(ikpose.getOrientations(), pose.getOrientations())]
                count += 1


            # Contact stitch
            if not contacts[i] and 0 < count:
                if count < CONTACT_STITCH_RANGE:
                    pose.interpolateOnlyOrientations(difference, count/CONTACT_STITCH_RANGE)
                    count += 1
                elif count == CONTACT_STITCH_RANGE:
                    count = 0

            self.previousContacts[key] = (contacts[i], startPosition, count, difference)

#        self.log['s_posture'].append(pose)
#        self.log['contacts'].append((self.previousContacts['left'], self.previousContacts['right'])) 
        return pose


    def postureRootAlignment(self, pose, stitchStartPoint, rootAlignVector, rootAlignMatrix):
        pos = pose.stitchedPosition(stitchStartPoint, rootAlignVector, rootAlignMatrix)
        ori = pose.orientations.copy()

        alignedRootFrame = rootAlignMatrix @ pose.orientations[0]
        ori[0] = alignedRootFrame 

        pose = Posture()
        pose.setPosture(pos, ori)
        return pose

            # POSE: INERTIAL BLENDING
#                frametime = self.getFrameTime()

#                count = int(STITCH_RANGE * ratio)
#                lastQueriedIndex = self.currentIndex - count
#
#                rotations = {
#                    '0':self.postures[lastQueriedIndex+1].getOrientations(),
#                    '-1':self.postures[lastQueriedIndex].getOrientations(),
#                    '-2':self.postures[max(0, lastQueriedIndex-1)].getOrientations(),
#                    't':self.postures[self.currentIndex].getOrientations(),
#                }
#                mat = poseToMix.inertialBlending(rotations, count*frametime, STITCH_RANGE*frametime)
#                
#                for i, m in enumerate(mat):
#                    ori[i][:3,:3] = m
#
#                ori[0] = rootAlignMatrix @ ori[0] 
#                pose.setPosture(pos, ori)
#



    def updateLog(self, n, p, taskInfo, short=False):
        self.log['frame number'].append(n)
        self.log['root'].append(p.getPosition()*[1,0,1])
        self.log['posture'].append(p)
        self.log['direction'].append(p.getLocalFrame()[:,2])

        if taskInfo is None:
            taskInfo = self.log['taskInfo'][-1] # save duplicates
        self.log['taskInfo'].append(taskInfo)

        self.log['query'].append(self.lastQueriedFutureTrajectory)

        if short:
            self.log['root'] = self.log['root'][-10:]
            self.log['posture'] = self.log['posture'][-10:]
            self.log['direction'] = self.log['direction'][-10:]

    def emptyLog(self, count):
        assert(count > 0)
        self.setPostureIndex(self.log['frame number'][-count-1])
        for key in self.log:
            self.log[key] = self.log[key][:-count]

# -----------------------------------------

    def setNetwork(self, value, network=None):
        self.useNetwork = value
        if self.motionMatcher is None:
            return False
        if network is not None:
            self.motionMatcher.network = network
        return True

    def setFeatureWeights(self, values):
        if self.motionMatcher:
            assert(len(values) == 5)

            # weights: foot pos, foot vel, hip vel, future trajectory pos, ori
            wfp, wfv, whv, wftp, wfto = values
            weights = np.concatenate(([wfp]*6, [wfv]*6, [whv]*3, [wftp, wftp, wfto, wfto]*3))
            weights = np.array(weights, dtype=np.float64)

            assert(len(weights) == 27)
            self.motionMatcher.setFeatureWeights(weights)
            return True
        return False

    def setPostureIndex(self, value):
        self.currentIndex = value
       
    def nextPosture(self):
        self.currentIndex = np.clip(self.currentIndex + 1, a_min=0, a_max=self.totalPosturesCount - 1)
        return self.postures[self.currentIndex], self.currentIndex

    def setCurrentFrameNum(self, value):
        self.currentIndex = value
    def getCurrentFrameNum(self):
        return self.currentIndex
    def getCurrentMotion(self):
        idx = bisect_right(self.lastPostureLengths, self.currentIndex)
        motion = self.motions[idx]
        mIdx = self.currentIndex - self.lastPostureLengths[idx]

        motion.setToPosture(mIdx)
        return motion
    def getLog(self, key):
        return self.log.get(key)

    def setFutureTrajectoryIndex(self, value):
        self.futureTrajectoryIndex = value
    def getFutureTrajectoryIndex(self):
        return self.futureTrajectoryIndex

# -------------------------------------------------------------------------------------
    def getSkeletonRoot(self):
        return self.motions[0].getSkeletonRoot()
    def getFrameTime(self):
        return self.motions[0].frameTime
    def getTotalFrameCount(self):
        return self.totalPosturesCount 
    def getCharacterPosition(self):
        return self.log['root'][-1]
