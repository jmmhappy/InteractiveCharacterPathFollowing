from render.RendererInterface import RendererInterface
from render.drawPrimitives import *

import ParticleSolver

import numpy as np

class ParticleRenderer(RendererInterface):
    def __init__(self, name):
        self.FRAMETIME = 1/30
        self.solver = ParticleSolver.ParticleSolver()
        if name == 'cube':
            vertices = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1]])
            self.solver.addSpringObject([0,3,1], vertices, KD=-20, KS=-10, m=10)
        elif name == 'dot':
            self.solver.addDotObject([0,5,1])

    def getFocusPoint(self):
        return np.zeros(3)

    def render(self, shadow, option):
        color = (255,0,0)
        for pp in self.solver.getParticlePositions():
            drawCube(color, pp, w=0.1, h=0.1, l=0.1)

        for p1,p2 in self.solver.getSpringPairPositions():
            drawLine(color, p1, p2)

    def update(self, userInput):
        self.solver.simulate(self.FRAMETIME, 'Euler')#'Midpoint')
        return 0 # frame counter always 0

    def reset(self):
        self.solver.reset()
