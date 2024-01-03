import numpy as np

class Particle:
    def __init__(self, initialPosition, m):
        self.x = initialPosition
        self.v = np.zeros(3)
        self.F = np.zeros(3)
        self.m = m

    def clearForce(self):
        self.F = np.zeros(3)

    def getState(self):
        return np.concatenate((self.x, self.v))
    def getDeriv(self):
        return np.concatenate((self.v, self.F / self.m))

    def setState(self, state):
        self.x = state[:3]
        self.v = state[3:]


class ParticleSystem:
    def __init__(self):
        self.timeElapsed = 0
        self.forces = []
        self.particles = [] 

    def derivEvalLoop(self, dt):
        self.clearForces()
        self.applyForces()
        return self.getDeriv()

    def addForce(self, force):
        self.forces.append(force)
    def addParticle(self, particle):
        self.particles.append(particle)
    def getParticles(self):
        return self.particles
    def addTime(self, dt):
        self.timeElapsed += dt

    def clearForces(self):
        for particle in self.particles:
            particle.clearForce()

    def applyForces(self):
        for force in self.forces:
            force.apply()

    def getState(self):
        result = np.array([])
        for particle in self.particles:
            result = np.append(result, particle.getState())
        return result

    def getDeriv(self):
        result = np.array([])
        for particle in self.particles:
            result = np.append(result, particle.getDeriv())
        return result

    def setState(self, state):
        assert(len(state) == len(self.particles)*6)
        for i, particle in enumerate(self.particles):
            particle.setState(state[i*6 : i*6+6])
