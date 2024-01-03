import numpy as np

from data.Particle import ParticleSystem, Particle
from data.Force import Gravity, VisciousDrag, DampedSpring

class ParticleSolver:
    def __init__(self):
        self.sys = ParticleSystem()

    def simulate(self, dt, method='Euler'):

        state = self.sys.getState()
        nextState = self.integrate(state, dt, method) # this is just a candidate
        nextState = self.solveCollision(state, nextState, dt, method)

        self.sys.setState(nextState)
        self.sys.addTime(dt)

    def reset(self):
        self.sys.clearForces()
        self.sys.setState(np.zeros(len(self.sys.getParticles)*6))


    def addDotObject(self, vertice, m=1):
        p = Particle(initialPosition=np.array(vertice), m=m)
        self.sys.addParticle(p)
        self.sys.addForce(Gravity(p))

    def addSpringObject(self, startPoint, vertices, KD, KS, m=1):
        vertices += startPoint
        for i, v in enumerate(vertices):
            self.addDotObject(v)

        particles = self.sys.getParticles()[-len(vertices):]
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles[i+1:]):
                self.sys.addForce(DampedSpring(p1, p2, KD, KS))

    def addMouseParticle(self, point, targetParticle, KD, KS, m=1):
        self.mouseParticle = Particle(initialPosition=np.array(point), m=m)
        self.sys.addForce(DampedSpring(self.mouseParticle, targetParticle, KD, KS))

    def moveMouseParticle(self, dx, dy):
        self.mouseParticle.x += [dx, dy, 0]

    def removeMouseParticle(self):
        self.sys.forces.pop()

    def nearestParticle(self, point):
        dist = 99999
        target = None
        for p in self.sys.getParticles():
            d = sum((p.x-point) * (p.x-point))
            if dist > d:
                dist = d
                target = p
        return target

    # ---------------------------------------------------------------------

    def solveCollision(self, prevState, state, dt, method):

        def contact(v, N): # assume collision
            EPSILON = 2.5
            if abs(np.dot(v,N)) < EPSILON:
                return True
            return False

        def collision(pos, N):
            EPSILON = 0.03
            pointOnGround = np.zeros(3)
            if np.dot((pos - pointOnGround), N) < EPSILON:
                return True
            return False

        def rollBack(pos, prevPos, N):
            pointOnGround = np.zeros(3)
            distToGround = np.dot(N, pos[1] - pointOnGround)
            previousDistToGround = np.dot(N, prevPos[1] - pointOnGround)

            if distToGround < 0:
                assert(previousDistToGround > 0)
                ratio = abs(previousDistToGround)/(abs(distToGround) + abs(previousDistToGround))
                return ratio
            
            print('Rollback called without a collision!')
            print('You need bigger contact epsilon.')
            return -1 

        ELASTIC = 0.7
        maxRatio = -1
        indices = []
        for i in range(0, len(state), 6):

            N = np.array([0,1,0])
            x, v = state[i:i+3], state[i+3:i+6]

            if collision(x, N):
                vN = np.dot(N, v)*N
                vT = v - vN

                if contact(v, N):
                    state[i+3:i+6] = vT
                else:
                    ratio = rollBack(x, prevState[i:i+3], N)
                    if ratio > 0:
                        maxRatio = max(ratio, maxRatio)
                        indices.append(i)

        if maxRatio == -1: # no collisions
            return state

        tmp = self.integrate(prevState, ratio*dt, method)

        for i in indices:
            vN = np.dot(N, v)*N
            vT = v - vN

            if contact(v,N):
                tmp[i+3:i+6] = vT
            else:
                tmp[i+3:i+6] = -(vN*ELASTIC + vT) # flip velocity

        return self.integrate(tmp, (1-ratio)*dt, method)



    def integrate(self, state, dt, method):
        def euler(state, dt):
            deriv = self.sys.derivEvalLoop(dt)
            return state + deriv * dt
        def midpoint(state, dt):
            halfDeriv = self.sys.derivEvalLoop(dt*0.5)
            return state + halfDeriv * dt
        def rungeKutta4(state, dt):
            pass
        def implicitEuler(state, dt):
            pass

        if method == 'Euler':
            state = euler(state, dt)
        elif method == 'Midpoint':
            state = midpoint(state, dt)
        elif method == 'RK4':
            state = rungeKutta4(state, dt)
        elif method == 'Implicit':
            state = implicitEuler(state, dt)

        return state

    def getParticlePositions(self): # canvas
        return [p.x for p in  self.sys.particles]

    def getSpringPairPositions(self):
        result = []
        for f in self.sys.forces:
            if isinstance(f, DampedSpring):
                result.append((f.p1.x, f.p2.x))
        return result
