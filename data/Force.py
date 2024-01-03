import numpy as np

class Gravity:
    def __init__(self, p, G=-9.8):
        self.G = G
        self.p = p # target particle

    def apply(self):
        self.p.F[1] += self.p.m * self.G

class VisciousDrag:
    def __init__(self, p, k):
        self.k = k
        self.p = p

    def apply(self):
        self.p.F -= self.k * self.p.v

class DampedSpring:
    def __init__(self, p1, p2, kd, ks):
        self.p1 = p1
        self.p2 = p2
        self.kd = kd
        self.ks = ks

        self.restLength = np.sqrt(sum((p1.x-p2.x)*(p1.x-p2.x)))

    def apply(self):
        dx = self.p2.x - self.p1.x 
        dv = self.p2.v - self.p1.v
        dxLen = np.sqrt(sum(dx*dx))

        # spring, inertia
        f1 = -(self.ks*(dxLen - self.restLength) + self.kd*(dv*(dx/dxLen))) * (dx/dxLen)
        f2 = (-f1)

        self.p1.F += f1
        self.p2.F += f2
