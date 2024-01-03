import dartpy as dart

import os

from render.RendererInterface import RendererInterface
from render.drawPrimitives import *

import numpy as np

from rl.CarEnvironment import Environment
from rl.CarAgent import Agent

class SkelRenderer(RendererInterface):
    def __init__(self, skelPath, frameTime=1/30):
        self.FRAMETIME = frameTime # CanvasBase needs to start timer
        
        if skelPath[0] != '/': # relative path
            skelPath = os.getcwd() + "/" + skelPath

        self.env = Environment({})
        self.skel = self.env.skel
        self.network = None
#        self.network = Agent("/home/jmm/ray_results/PPO_minicar_env_2022-06-16_10-03-26qj_nd9l7/checkpoint_000011/checkpoint-11")

        self.q_d = self.skel.getPositions() # desired pose
        self.q_d[6:] = 1

    def getFocusPoint(self):
        return self.skel.getPosition()[3:6]

    def render(self, shadow, options, observation): # Draw
        def drawSkeleton(isShadow):
            for i in range(self.skel.getNumBodyNodes()):
                bn = self.skel.getBodyNode(i)
                j = self.skel.getJoint(i)

                glPushMatrix()
                glMultTransposeMatrixf(bn.getTransform().matrix())

                shape = bn.getShapeNode(0).getShape()
                shapeType = shape.getType()
                if shapeType == 'BoxShape':
                    w,h,l = shape.getSize()
                    color = (100,100,100) if isShadow else (0,255,0)
                    drawCube(color, (0,0,0), w, h, l)

                elif shapeType == 'CylinderShape':
                    color = (100,100,100) if isShadow else (255,255,255)
                    drawCylinder(color, (0,0,0), shape.getRadius(), shape.getHeight())

                glPopMatrix()



        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, (70,100,50,1))

        q = self.skel.getPositions()
        #self.skel.getDof('joint 1_y').setPosition(2)

        drawSkeleton(False)
        glDisable(GL_LIGHTING)

        #shadow
        glPushMatrix()
        glMultMatrixf(shadow)
        drawSkeleton(True)
        glPopMatrix()

    def update(self, userInput=None): # Timer
        if self.network:
            obs = self.skel.getPositions()
            action = self.network.action(obs)
        else:
            action = self.q_d[6:]

        self.env.step(action)

    def reset(self):
        self.env.reset()
