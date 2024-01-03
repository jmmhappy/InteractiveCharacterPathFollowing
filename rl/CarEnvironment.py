import gym
import numpy as np
import dartpy as dart

class Environment(gym.Env):
    def __init__(self, env_config):
        super().__init__()

        self.world = dart.utils.SkelParser.readWorld("/home/jmm/motionmatching/skel/4legger.skel")
        self.world.setGravity([0,-9.81,0])
        self.world.setTimeStep(0.001)

        self.skel = self.world.getSkeleton(1) 
        self.dofs = self.skel.getNumDofs() 

        self.Kp = np.identity(self.dofs) * 400
        self.Kd = np.identity(self.dofs) * 40

        for i in range(6):
            self.Kp[i,i] = 0.0
            self.Kd[i,i] = 0.0

        self.torques = np.zeros(self.dofs)
        
        self.timestep = self.world.getTimeStep()
        self.targetPosition = np.array([3,0,0])

        self.action_space = gym.spaces.Box(low=np.full(self.dofs-6, -np.inf, dtype=np.float64),\
            high=np.full(self.dofs-6, np.inf, dtype=np.float64), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.full(self.dofs, -np.inf, dtype=np.float64),\
            high=np.full(self.dofs, np.inf, dtype=np.float64), dtype=np.float64)

    def reset(self):
        self.world.reset()
        self.skel.resetPositions()
        self.timestep = self.world.getTimeStep()
        self.Kp = np.identity(self.dofs) * 400
        self.Kd = np.identity(self.dofs) * 40
        for i in range(6):
            self.Kp[i,i] = 0.0
            self.Kd[i,i] = 0.0
        self.torques = np.zeros(self.dofs)
#        self.setTargetPosition()

        return self.observation_space.sample()

    def step(self, action):

        # PD control
        q = self.skel.getPositions()
        dq = self.skel.getVelocities()

        constraint_forces = self.skel.getConstraintForces()

        q_d = q.copy() # only target joint angles
        q_d[6:] = action

        # SPD tracking
        invM = np.linalg.inv(self.skel.getMassMatrix() + self.Kd * self.timestep)
        p = np.matmul(-self.Kp, q + dq * self.timestep - q_d)
        d = np.matmul(-self.Kd, dq)
        ddq = np.matmul(invM, -self.skel.getCoriolisAndGravityForces() + p + d + constraint_forces)

        self.torques = p + d + np.matmul(-self.Kd, ddq) * self.timestep
        self.torques[:6] = 0
        self.skel.setForces(self.torques * 0.8)

        # external force?

        for _ in range(33):
            self.world.step() # 0.001 seconds

        q = self.skel.getPositions()
        done = False

        if np.isnan(np.sum(q)):
            done = True
            return np.zeros(self.dofs), -500, done, {}

        reward = -np.linalg.norm(q[3:6] - self.targetPosition)
        if abs(reward) < 0.3:
            done = True

        return q, reward, done, {}

    def setTargetPosition(self):
        self.targetPosition += np.random.uniform(-5,5,3)
