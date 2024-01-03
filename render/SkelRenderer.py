#import dartpy as dart

from render.RendererInterface import RendererInterface
from render.drawPrimitives import *

import numpy as np

class SkelRenderer(RendererInterface):
    def __init__(self, skelPath, frameTime=1/30):
        self.FRAMETIME = frameTime # CanvasBase needs to start timer

        world = dart.utils.SkelParser.readWorld(skelPath)
        world.setGravity([0, -9.81, 0])
        world.setTimeStep(frameTime)

        self.timestep = world.getTimeStep()

        skel = world.getSkeleton(1)

        self.dofs = skel.getNumDofs()
        self.skel = skel
        self.world = world

        self.Kp = np.identity(self.dofs) * 400
        self.Kd = np.identity(self.dofs) * 40

        for i in range(6):
            self.Kp[i,i] = 0.0
            self.Kd[i,i] = 0.0

        self.torques = np.zeros(self.dofs)
        self.q_d = self.skel.getPositions() # desired pose
#        self.q_d[3:] = 0.1

    def getFocusPoint(self):
        return self.skel.getPosition()[:3]

    def render(self, shadow, _): # Draw
        # TODO shadow mapping

        q = self.skel.getPositions()
        #self.skel.getDof('joint 1_y').setPosition(2)

        for i in range(self.skel.getNumBodyNodes()):
            bn = self.skel.getBodyNode(i)
            j = self.skel.getJoint(i)

            shape = bn.getShapeNode(0).getShape()
            #shapeType = shape.getType()
            #if 'BoxShape' == shapeType:
            w,h,l = shape.getSize()

            glPushMatrix()
            glMultTransposeMatrixf(bn.getTransform().matrix())
            drawCube((255,0,255), (0,0,0), w, h, l)
            glPopMatrix()


    def update(self, userInput=None): # Timer

        # PD control
        q = self.skel.getPositions()
        dq = self.skel.getVelocities()

        constraint_forces = self.skel.getConstraintForces()

        # SPD tracking
        invM = np.linalg.inv(self.skel.getMassMatrix() + self.Kd * self.timestep)
        p = np.matmul(-self.Kp, q + dq * self.timestep - self.q_d)
        d = np.matmul(-self.Kd, dq)
        ddq = np.matmul(invM, -self.skel.getCoriolisAndGravityForces() + p + d + constraint_forces)

        self.torques = p + d + np.matmul(-self.Kd, ddq) * self.timestep
        self.torques[:6] = 0
        self.skel.setForces(self.torques * 0.8)

        # external force?

        self.world.step()
        return self.timestep

    def reset(self):
        self.world.reset()
