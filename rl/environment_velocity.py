import numpy as np
import gym

import random

from motion.MotionStream import MotionStream
from util.bvhMenu import load

MAX_GOAL_DISTANCE = 3.2767 # Target velocity

class Environment(gym.Env):

    data = None

    def __init__(self, env_config):
        super().__init__()

        if not Environment.data:
            Environment.data = load("pfnn.bin")
#            Environment.data = load('edin_loco.bin')

        self.mstream = MotionStream(Environment.data)
        self.mstream.useNetwork = True
        assert(self.mstream.motionMatcher is not None)

        self.action_space = gym.spaces.Box(low=np.full(12, -16, dtype=np.float64),\
            high=np.full(12, 16, dtype=np.float64), dtype=np.float64)
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
            'name': "joystick",
            "joystick":self.goal,

            'obstacles':None
        }
        idx = mstream.poseUpdate(mstream.getCurrentFrameNum(), userInput, action, stitch=False)[-1]
        pose = mstream.getLog('posture')[-1]

        position = pose.getPosition()[[0,2]]
        localFrame = pose.getLocalFrame()[:3,:3] # only rotation
        goal = (np.linalg.inv(localFrame) @ np.array([self.goal[0], 0, self.goal[1]]))[[0,2]] # in local

        obs = self.mstream.motionMatcher.features[idx][:15]
        obs = np.append(obs, goal)
        obs = np.array(obs, dtype=np.float64)

        # Target velocity
        prevPosition = mstream.getLog('posture')[-2].getPosition()[[0,2]]
        rootVelocity = (-prevPosition + position) * 30 # frametime

        reward, done = self.getReward(rootVelocity, action, idx, threshold=0.1) # 10%

        return obs, reward, done, {}


    def getReward(self, state, action, poseIndex, threshold):
        def l2norm(v):
            return np.sqrt(sum(v*v))

        reward, done = 0, False

        dist = l2norm(self.goal - state)
        reward += -dist * self.w['goal']

        matcher = self.mstream.motionMatcher
        actualAction = matcher.features[poseIndex][-12:]
        reward += -np.linalg.norm(actualAction - action) * self.w['penalty_a']

        denormed = matcher.denormalize(action) 
        penalty = 0
        for i in range(0,12,4):
            direction = denormed[i+2:i+4]
            penalty += abs(l2norm(direction) - 1) # make direction closer to a unit vector
        reward += -penalty * self.w['penalty_u']

        # Target velocity
        if np.random.uniform(0,1) < threshold:
            self.setGoal(None)

        return reward, done

    def setGoal(self, pos):
        global MAX_GOAL_DISTANCE

        # Target velocity
        th = np.random.uniform(0, 2*np.pi)
        rad = np.random.uniform(0.5, MAX_GOAL_DISTANCE)
        self.goal = np.array([np.cos(th), np.sin(th)]) * rad

    def getGoal(self):
        return self.goal

