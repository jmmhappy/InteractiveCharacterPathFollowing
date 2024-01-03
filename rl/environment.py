import numpy as np
import gym

from data.Task import Task
import random

from motion.MotionStream import MotionStream
from util.bvhMenu import load

MAX_GOAL_DISTANCE = 5

ACTION_SPACE = 10 # 12

class Environment(gym.Env):

    data = None

    def __init__(self, env_config):
        super().__init__()

        if not Environment.data:
            # Environment.data = load('edin_loco.bin')
            Environment.data = load('pfnnContFeature.bin')

        self.mstream = MotionStream(Environment.data)
        self.obstacles = env_config['map']
        MAP_SIZE = env_config['map_size']

        self.mstream.useNetwork = True
        assert(self.mstream.motionMatcher is not None)

        self.action_space = gym.spaces.Box(low=np.full(ACTION_SPACE, -16, dtype=np.float64),\
            high=np.full(ACTION_SPACE, 16, dtype=np.float64), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            'character':gym.spaces.Box(low=np.full(17, -np.inf, dtype=np.float64),\
                high=np.full(17, np.inf, dtype=np.float64), dtype=np.float64),
            'terrain':gym.spaces.Box(low=False, high=True, shape=(2, MAP_SIZE, MAP_SIZE), dtype=bool)
        })

        self.goal = np.array([0,0])
        self.setGoal(np.zeros(2))

        self.timesteps = 0

        self.w = {
            'goal':.1,
            'penalty_u':.1,
            'penalty_a':.1,
        }


    # overridden
    def reset(self):
        self.setGoal(np.zeros(2))
        self.timesteps = 0
        self.mstream.playReset(random=True)
        self.obstacles.reset()
        return self.observation_space.sample() 

    # overridden
    def step(self, action):
        self.timesteps += 1
        mstream = self.mstream

        userInput = {
            'name':'position',
            'position':self.goal,
            'obstacles':self.obstacles
        }

        # get localMap of previous terrain
        pose = mstream.getLog('posture')[-1]
        localFrame = pose.getLocalFrame()
        previousTerrain = self.obstacles.getLocalMap(localFrame) # return 1(channel) * MAP_SIZE(w) * MAP_SIZE(h)


        self.obstacles.move(0.2) # 6/30

        idx = mstream.poseUpdate(mstream.getCurrentFrameNum(), userInput, action, stitch=False)[-1]
        pose = mstream.getLog('posture')[-1]

        position = pose.getPosition()[[0,2]]
        localFrame = pose.getLocalFrame()
        goal = (np.linalg.inv(localFrame) @ np.array([self.goal[0], 0, self.goal[1], 1]))[[0,2]]

        obs = self.mstream.motionMatcher.features[idx][:15]
        obs = np.append(obs, goal)
        obs = np.array(obs, dtype=np.float64)

        reward, done = self.getReward(position, action, idx)

        # map wrt current character
        currentTerrain = self.obstacles.getLocalMap(localFrame) # return 1(channel) * MAP_SIZE(w) * MAP_SIZE(h)
        terrain = np.concatenate((previousTerrain, currentTerrain)) # 2 * MAP_SIZE * MAP_SIZE

        nodePos = mstream.motionMatcher.getNodePositions(idx)
        if self.obstacles.collision(pose, nodePos):
            reward = -10 * (1000 - self.timesteps)
            done = True
        
        return {'character':obs, 'terrain':terrain}, reward, done, {} 


    def getReward(self, pos, action, poseIndex):
        def l2norm(v):
            return np.sqrt(sum(v*v))

        reward, done = 0, False

        dist = l2norm(self.goal - pos)
        reward += -dist * self.w['goal']

        matcher = self.mstream.motionMatcher
        actualAction = matcher.features[poseIndex][15:]
        diff = np.linalg.norm(actualAction - action)
        reward += -diff * self.w['penalty_a']

        denormedAction = matcher.denormalize(action)
        penalty = 0
#        for i in range(0,12,4):
#            direction = denormedAction[i+2:i+4]
#            penalty += abs(l2norm(direction) - 1) # make direction closer to a unit vector

        for i in [3,8]:
            direction = denormedAction[i:i+2]
            penalty += abs(l2norm(direction) - 1)

        reward += -penalty * self.w['penalty_u']

        if dist < 0.5:
            reward = 1000
            done = True

        return reward, done

    def setGoal(self, pos):
        global MAX_GOAL_DISTANCE
        theta = random.uniform(0,np.pi*2)
        self.goal = pos + np.array([np.cos(theta), np.sin(theta)]) * MAX_GOAL_DISTANCE
        #self.goal = pos + [random.uniform(-MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE) for _ in range(2)] 
    def getGoal(self):
        return self.goal

