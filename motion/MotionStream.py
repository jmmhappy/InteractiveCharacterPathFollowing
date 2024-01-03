import numpy as np
from scipy.spatial import KDTree

from data.Motion import Motion
from data.Posture import Posture
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

        self.stitchStartPoint = postures[0].getPosition() * [0,1,0]
        self.rootAlignVector, self.rootAlignMatrix = np.zeros(3), np.identity(4)

        self.useNetwork = False 

        self.playReset()

    def playReset(self, random=False):
        self.currentIndex = np.random.randint(0, self.totalPosturesCount) if random else 0

        self.lastQueriedFutureTrajectory = None
        self.lastOriginalFutureTrajectory = None
        self.lastUpdated = 0

        initPose = self.postures[0]

        poseFromOrigin = Posture()
        
        # x,z=0,0
        x, z = self.startXPos

        poseFromOrigin.setPosture(np.array([x, initPose.getPosition()[1], z]), [np.identity(4)])

        self.setRootAlignment(poseFromOrigin, self.postures[self.currentIndex])


        global LEFT_FOOT, RIGHT_FOOT
        skel = self.motions[0].getSkeletonRoot()
        nodePos = dict(initPose.forwardKinematics(skel, footOnly=True))
        self.previousContacts = {
            'left':(False, nodePos[LEFT_FOOT], 0, None),
            'right':(False, nodePos[RIGHT_FOOT], 0, None)
        }

        self.log = {
            'root':[np.array([0,0,0])],
            'direction':[np.array([0,0,1,1])],
            'frame number':[self.currentIndex],
            'posture':[self.postureRootAlignment(self.postures[self.currentIndex])],

            'query':[[]],
            'original':[[]],

            'taskInfo':[{
                'cornerPoint':None,
                'queryPoints':[], 
                'limited':False,
                'future':[]}],

#            's_posture':[self.postureRootAlignment(self.postures[0])],
#            'contacts':[(self.previousContacts['left'], self.previousContacts['right'])],
        }


    def poseUpdate(self, frameNum, userInput, action=None, stitch=True):

        global taskFactory

        STITCH_RANGE = 6
        motion = self.getCurrentMotion()

        if self.motionMatcher is not None:

            task = taskFactory.create(self.log['root'], self.log['direction'], userInput)

            poseBeforeQuery = self.log['posture'][-1]

            self.currentIndex = self.matchByQuery(frameNum, task, motion, action)
            poseFromDB = self.postures[self.currentIndex]
            self.setRootAlignment(poseBeforeQuery, poseFromDB)

            # calculate the difference between m2^0 and m1^N
            poseAligned = self.postureRootAlignment(poseFromDB)
            p_diff = poseBeforeQuery.getPosition()[1] - poseAligned.getPosition()[1]
            o_diff = [log(o2.T @ o1) for o1, o2 in zip(poseBeforeQuery.getOrientations(), poseAligned.getOrientations())]
            difference = (p_diff, o_diff)

            indices = []
            for i in range(STITCH_RANGE):
                poseFromDB, self.currentIndex = self.nextPosture()
                if stitch:
                    poseToRender = self.postProcess(poseFromDB, difference, (i+1)/STITCH_RANGE)
                else:
                    poseToRender = self.postProcess(poseFromDB)

                if task and task.name == 'tracking':
                    self.updateLog(self.currentIndex, poseToRender, task._meta)
                else:
                    self.updateLog(self.currentIndex, poseToRender, None)
                indices.append(self.currentIndex)
            return indices

        poseFromDB, self.currentIndex = self.nextPosture()
        poseToRender = self.postProcess(poseFromDB)

        self.updateLog(self.currentIndex, poseToRender, None)
        return [self.currentIndex]


    def matchByQuery(self, frameNum, task, motion, action):

        assert(self.log['frame number'][-1] == frameNum)
        assert(task)

        pose = self.log['posture'][-1]
        localFrame = pose.getLocalFrame()

        query = self.motionMatcher.getFeature(frameNum).copy() # normalized
        network = self.motionMatcher.network if self.useNetwork else None

        originalAction = query[15:]
        originalAction = self.motionMatcher.denormalize(originalAction)

        if action is None: # runtime
            obs = task.getObservation(query, localFrame)
            obs = motion if obs is None else obs
            action = task.action(obs, network)

        if task.isReinforcementLearning():
            assert(self.useNetwork is True)
            action = self.motionMatcher.denormalize(action)

            # action[2:4] /= np.linalg.norm(action[2:4])
            # action[6:8] /= np.linalg.norm(action[6:8])
            # action[10:12] /= np.linalg.norm(action[10:12])

            action[3:5] /= np.linalg.norm(action[3:5])
            action[8:10] /= np.linalg.norm(action[8:10])

        query[15:] = self.motionMatcher.normalize(action)

        self.lastQueriedFutureTrajectory = task.getLogInFrames(localFrame, action)
        self.lastOriginalFutureTrajectory = task.getLogInFrames(localFrame, originalAction)
        nextFrameNum = self.motionMatcher.findByQuery(query)

        if frameNum == nextFrameNum: # queried self
            return frameNum + 1
        return nextFrameNum


    def postProcess(self, poseFromDB, difference=None, stitchRatio=None):
        # Root align
        skeleton = self.getSkeletonRoot()
        pose = self.postureRootAlignment(poseFromDB)
        contacts = poseFromDB.getFootContacts()
        footNames = (LEFT_FOOT, RIGHT_FOOT)

        # Stitch
        if stitchRatio:
            pose.interpolate(difference, stitchRatio) 
        else:
            return pose

        if contacts[0] == None:
            return pose

        # IK catch foot sliding
        CONTACT_STITCH_RANGE = 5

        for i, key in enumerate(['left', 'right']):
            prevContact, startPosition, count, difference = self.previousContacts[key]

            if not prevContact and contacts[i]:
                count = 0 # new phase

                nodePos = dict(pose.forwardKinematics(skeleton, footOnly=True))
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


    def postureRootAlignment(self, pose):
        pos = pose.stitchedPosition(self.stitchStartPoint, self.rootAlignVector, self.rootAlignMatrix)
        ori = pose.orientations.copy()

        alignedRootFrame = self.rootAlignMatrix @ pose.orientations[0]
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
#                ori[0] = self.rootAlignMatrix @ ori[0] 
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
        self.log['original'].append(self.lastOriginalFutureTrajectory)

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

    def setRootAlignment(self, destinationPose, targetPose):
        self.stitchStartPoint = targetPose.getPosition()
        self.rootAlignVector, self.rootAlignMatrix = destinationPose.rootAlignment(targetPose)

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

# -------------------------------------------------------------------------------------
    def getSkeletonRoot(self):
        return self.motions[0].getSkeletonRoot()
    def getFrameTime(self):
        return self.motions[0].frameTime
    def getTotalFrameCount(self):
        return self.totalPosturesCount 
    def getCharacterPosition(self):
        return self.log['root'][-1]
