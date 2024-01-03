import numpy as np

# Y-UP

MIN_DISTANCE = 2 # no hit with pose
MAX_DISTANCE = 10
MAX_LENGTH = 3

SIZE = 16 
RANGE = 6

M_L = [] 
for i in np.arange(-RANGE/2, RANGE/2, RANGE/SIZE):
    for j in np.arange(-RANGE/2, RANGE/2, RANGE/SIZE):
        M_L.append([i, 0, j, 1])
M_L = np.array(M_L).T

NUM_OBSTACLES = 30
MAX_VELOCITY = 0

class ObstacleMap:
    def __init__(self):
        self.center = np.array([self._setRandomLocation() for _ in range(NUM_OBSTACLES)])
        self.velocity = np.random.uniform(0, -MAX_VELOCITY, (NUM_OBSTACLES, 2))
        self.velocity[:,0] = 0 # moving direction fixed
        self.length = np.random.randint(1, MAX_LENGTH, (NUM_OBSTACLES, 2))

    def reset(self):
        self.center = np.array([self._setRandomLocation() for _ in range(NUM_OBSTACLES)])
        self.velocity = np.random.uniform(0, -MAX_VELOCITY, (NUM_OBSTACLES, 2))
        self.velocity[:,0] = 0 # moving direction fixed
        self.length = np.random.randint(1, MAX_LENGTH, (NUM_OBSTACLES, 2))

    def move(self, delta):
        self.center += self.velocity * delta
#        mask = np.array(self.center[:, 1] < 0) # is z out of range?
#        self.center[mask, 1] = MAX_DISTANCE # filp to left

    def collision(self, pose, node_pos):

        positions_local = np.array(list(node_pos.values())).T
        positions = pose.getLocalFrame() @ np.vstack((positions_local, np.ones(positions_local.shape[1])))

        positions = (positions[[0,2]]).T # 31 x 2

        low, high = self._getObstaclesNearBy(pose.getPosition()[[0,2]]) # Y up
        for l, h in zip(low, high): # each obj
            comparison = ((l <= positions) & (positions <= h)).T
            if np.any(comparison[0] & comparison[1]):
                return True

        return False

    def getRenderInfo(self):
        return {'center':self.center, 'length':self.length}

    def _setRandomLocation(self):
        center = np.array([0,0])
        while abs(center[0]) < MIN_DISTANCE and abs(center[1]) < MIN_DISTANCE:
            center = np.random.uniform(-MAX_DISTANCE, MAX_DISTANCE, 2)
            minus = np.around(np.random.random(2))
            center = center * np.power([-1,1], minus)
        return center


    def _getObstaclesNearBy(self, pos):
        low, high = self.center - self.length*0.5, self.center + self.length*0.5
        
        half = RANGE/2
        adjacents = (((pos-half) < high) & (low < (pos+half))).T
        adjacents = adjacents[0] & adjacents[1]
        return low[adjacents], high[adjacents]

