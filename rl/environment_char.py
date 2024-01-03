import numpy as np
import gym

import random

from motion.MotionStream import MotionStream
from util.bvhMenu import load

MAX_GOAL_DISTANCE = 10 # Position

ACTION_SPACE = 10 # 12

class Environment(gym.Env):

    data = None

    def __init__(self, env_config):
        super().__init__()

        if not Environment.data:
            Environment.data = load("pfnnContFeature.bin")
            # Environment.data = load("pfnn.bin")
#            Environment.data = load('edin_loco.bin')

        self.mstream = MotionStream(Environment.data)
        self.mstream.useNetwork = True
        assert(self.mstream.motionMatcher is not None)

        self.action_space = gym.spaces.Box(low=np.full(ACTION_SPACE, -16, dtype=np.float64),\
            high=np.full(ACTION_SPACE, 16, dtype=np.float64), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.full(17, -np.inf, dtype=np.float64),\
                high=np.full(17, np.inf, dtype=np.float64), dtype=np.float64)

        self.goal = np.array([0,0])
        self.setGoal(np.zeros(2))

        self.timesteps = 0

        self.w = {'goal':1, 'penalty_a':1, 'penalty_u':1}


    # overridden
    def reset(self):
        self.setGoal(np.zeros(2))
        self.timesteps = 0
        self.mstream.playReset(random=True)
        return self.observation_space.sample() 

    # overridden
    def step(self, action):
        self.timesteps += 1
        mstream = self.mstream

        userInput = {
            'name':'position',
            'position':self.goal,

            'obstacles':None
        }
        idx = mstream.poseUpdate(mstream.getCurrentFrameNum(), userInput, action, stitch=False)[-1]
        pose = mstream.getLog('posture')[-1]

        position = pose.getPosition()[[0,2]]
        localFrame = pose.getLocalFrame()
        goal = (np.linalg.inv(localFrame) @ np.array([self.goal[0], 0, self.goal[1], 1]))[[0,2]] # in local

        obs = self.mstream.motionMatcher.features[idx][:15]
        obs = np.append(obs, goal)
        obs = np.array(obs, dtype=np.float64)

        # Position
        reward, done = self.getReward(position, action, idx, threshold=0.5)

        return obs, reward, done, {"taskReward":-np.linalg.norm(self.goal - position), "initialDistance":np.linalg.norm(self.goal)}


    def getReward(self, state, action, poseIndex, threshold):
        def l2norm(v):
            return np.sqrt(sum(v*v))

        reward, done = 0, False

        dist = l2norm(self.goal - state)
        reward += -dist * self.w['goal']

        matcher = self.mstream.motionMatcher
        actualAction = matcher.features[poseIndex][-ACTION_SPACE:]
        reward += -np.linalg.norm(actualAction - action) * self.w['penalty_a']

        denormed = matcher.denormalize(action) 
        penalty = 0
        # for i in range(0,12,4):
            # direction = denormed[i+2:i+4]
            # penalty += abs(l2norm(direction) - 1) # make direction closer to a unit vector

        for i in [3,8]:
            direction = denormed[i:i+2]
            penalty += abs(l2norm(direction) - 1)

        reward += -penalty * self.w['penalty_u']

        # Position
        if dist < threshold:
            done = True
            self.reset()

        return reward, done

    def setGoal(self, pos):
        global MAX_GOAL_DISTANCE
        # Position
        self.goal = pos + [random.uniform(-MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE) for _ in range(2)] 

    def getGoal(self):
        return self.goal

