import numpy as np
from scipy.spatial import cKDTree
from bisect import bisect_right

from data.Trajectory import qt_factory 
from util.matrixFormulas import *

# FEATURE_DIM = 27
IGNORE_NUM = 10

class MotionMatcher():
    def __init__(self, motions, FUTURE_TERM, leftFootName, rightFootName):
        print("=== Creating Motion Matcher ===")
        self.motions = motions
        self.FUTURE_TERM = FUTURE_TERM

        self.leftFootName = leftFootName
        self.rightFootName = rightFootName

        features, nodePositions = self._calculateFeatures()
        self.features, self.mu, self.sig = self._normalizeFeatures(features)
        self.nodePositions = nodePositions
        print('Calculation Done.')

        tree_features = None # ignore euler extrapolated ft 
        lastFrameNums = np.cumsum([len(m.getPostures()) for m in motions])
        for i, n in enumerate(lastFrameNums):
            if i < 1:
                tree_features = list(self.features[: n - IGNORE_NUM])
            else:
                prev = lastFrameNums[i-1]
                tree_features += list(self.features[prev : n - IGNORE_NUM])

        self.tree = cKDTree(np.array(tree_features, dtype=np.float32))
        self.tree_features = tree_features
        self.lastFeatureNums = np.cumsum([len(m.getPostures()) - IGNORE_NUM for m in motions])
        assert(len(tree_features) == self.lastFeatureNums[-1])
        print('KD Tree Generated.')
        print('Now Ready!')
       
        self.network = None


    def normalize(self, vector):
        # assert(len(vector)==12)
        return (vector - self.mu[15:]) / self.sig[15:]
    def denormalize(self, normalizedVector):
        # assert(len(normalizedVector)==12)
        return normalizedVector * self.sig[15:] + self.mu[15:]

    def getFeature(self, index):
        return self.features[index]

    def getNodePositions(self, index):
        return self.nodePositions[index]

    def findByQuery(self, query, k=1):
        if k == 1:
            d, i = self.tree.query(query, k=1)
            cnt = bisect_right(self.lastFeatureNums, i)
            return d, i + IGNORE_NUM * cnt 

        distances, indices = self.tree.query(query, k=k)
        return distances, [i + IGNORE_NUM * bisect_right(self.lastFeatureNums, i) for i in indices]


    def _calculateFeatures(self):
        features = None
        nodePositions = []
        for motion in self.motions:
            root = motion.getSkeletonRoot()
            motion.reset()
            for pose in motion.getPostures():
                qt = qt_factory(None, self.FUTURE_TERM)
                t = qt.format(qt.calculate(motion))

                nodePositions.append(dict(pose.forwardKinematics(root)))

                if features is None:
                    features = np.array(self._composeFeature(motion, t, nodePositions))
                else:
                    features = np.vstack((features, self._composeFeature(motion, t, nodePositions)))

                motion.setToNextPosture()

        assert(len(features) == len(nodePositions))
        return features, nodePositions


    def _normalizeFeatures(self, features):

        mu = np.mean(features, axis=0)
        sig = np.std(features, axis=0)

        # Overwrite self.features with normed value
        features = np.array([(f - mu) / sig for f in features], dtype=np.float32)

        return features, mu, sig


    def _composeFeature(self, motion, trajectory, nodePositions):
        def globalToLocal(localFrame, point):
            return (np.linalg.inv(localFrame) @ np.append(point, 1))[:3]

        prevPosture = motion.getPrevPosture()
        posture = motion.getCurrentPosture()
        root = motion.getSkeletonRoot()

        localFrame = posture.getLocalFrame()

        FPS = int(1 / motion.getFrameTime())

        leftFootPos = nodePositions[-1][self.leftFootName]
        rightFootPos = nodePositions[-1][self.rightFootName]
        i = max(-2, -len(nodePositions))
        prevLeftFootPos = nodePositions[i][self.leftFootName]
        prevRightFootPos = nodePositions[i][self.rightFootName]

        localFrame[:3,3] = np.zeros(3) # only orientation for velocities

        leftFootVel = leftFootPos - prevLeftFootPos
        rightFootVel = rightFootPos - prevRightFootPos
        leftFootVel = globalToLocal(localFrame, leftFootVel) * FPS
        rightFootVel = globalToLocal(localFrame, rightFootVel) * FPS

        hipVel = posture.getPosition() - prevPosture.getPosition()
        hipVel = globalToLocal(localFrame, hipVel) * FPS

        futureTrajectory = trajectory 

        feature = np.concatenate((leftFootPos, rightFootPos, leftFootVel, rightFootVel,\
                    hipVel, futureTrajectory))
        # assert(len(feature) == FEATURE_DIM)
        return feature


