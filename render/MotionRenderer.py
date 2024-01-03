
import sys

import wx
import wx.glcanvas as glcanvas

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    haveOpenGL = True
except ImportError:
    haveOpenGL = False

from render.RendererInterface import RendererInterface
from render.MousePicker import MousePicker
from render.drawPrimitives import *

from motion.MotionStream import MotionStream

import numpy as np

# Mesh drawing
from util.matrixFormulas import *

class MotionRenderer(RendererInterface): 
    def __init__(self, db):
        super().__init__(db.getFrameTime())#self.FRAMETIME 

        self.TOTAL_FRAMECOUNT = db.getTotalFrameCount()
#        self.COLOR = (0,0,155)#np.asarray(np.random.rand(3) * 255, dtype=int)
#        self.COLOR = (158,15,15) # red
#        self.COLOR = (0,155,0) # green
        self.COLOR = (200,100,30) # orange
#        self.COLOR = (10,10,30) # purple

        self.stack = []
        self.motionStream = db 

        self.instant = np.array([0,0,1])

        self.reset()

    def update(self, userInput):

        frameNum = self.motionStream.getCurrentFrameNum()
        if self.stack == []:
            frameNums = self.motionStream.poseUpdate(frameNum, userInput)
            self.stack.extend(frameNums)

        frameNum = self.stack.pop(0)
        return frameNum

    def emptyStack(self):
        if self.stack == []:
            return

        # In case index at rewind(past)
        dummy = self.stack[0]
        startIndexWithoutDummy = len(self.stack) - 1 - self.stack[::-1].index(dummy)
        framesToRemove = len(self.stack) - startIndexWithoutDummy - 1
        if framesToRemove > 0:
            self.motionStream.emptyLog(framesToRemove)
        self.stack = []

    def forceQuery(self, instant):
        theta = np.arccos(np.clip(np.dot(instant, self.instant), -1, 1))
        if theta >= np.pi/2:
            self.emptyStack()
        self.instant = instant


    def setCurrentFrameNum(self, value):#slider
        self.motionStream.setCurrentFrameNum(value)
    def getFrameNumberLog(self):
        return self.motionStream.getLog('frame number')
    def getRootTrajectoryLog(self):
        return self.motionStream.getLog('root')

    def getFocusPoint(self):
        return self.motionStream.getLog('root')[-len(self.stack) - 1]

    def getCurrentLocalFrame(self):
        return self.motionStream.getLog('posture')[-len(self.stack) - 1].getLocalFrame()

    def render(self, shadow, options, observation):
        # options(dictionary): meshDrawing(bool), logDrawing(bool)

        postures = self.motionStream.getLog('posture')

        def drawCharacter(color, skel, pose):
            self.drawBvh(skel, pose, options['meshDrawing'], color) 

            if options['logDrawing']:
                for i in range(2, len(postures)-1, LOG_STRIDE):
                    pose = postures[i]
                    self.drawBvh(skel, pose, options['meshDrawing'], color)

        def drawGoal():

            goal = observation['target']
            if goal is None:return

            x,z = goal
            if np.linalg.norm(goal - self.getFocusPoint()[[0,2]]) < 0.5:
                drawCube((0,0,0), np.array([x,1,z]), w=.1,h=2,l=.1)
                return

            drawCube((255,0,0), np.array([x,1,z]), w=0.1, h=2, l=0.1)

        glEnable(GL_LIGHTING)

        LOG_STRIDE = 20
        skel = self.motionStream.getSkeletonRoot()
        pose = postures[-len(self.stack) - 1]

        drawCharacter(self.COLOR, skel, pose)
        drawGoal()

        glDisable(GL_LIGHTING)

        if options["r_arrows"]:
            futureTrajectory = self.motionStream.getLog('query')[-len(self.stack)-1]
            # self.drawArrows(futureTrajectory)
            self.drawArrows([futureTrajectory[0]], (0,0,215)) # left
            self.drawArrows([futureTrajectory[1]], (150,150,215))

            futureTrajectory = self.motionStream.getLog('original')[-len(self.stack)-1]
            self.drawArrows([futureTrajectory[0]], (215,0,0))
            self.drawArrows([futureTrajectory[1]], (215,150,150))


        if observation["joystick"] is not None:
            x, z = self.getFocusPoint()[[0,2]]
            v_x, v_z = observation["joystick"]
            size = np.linalg.norm(observation["joystick"])
            _x, _, _z = np.cross([0,1,0], np.array([v_x,0,v_z])/size)
            frame = np.array([[_x,0,v_x,x],
                              [0,1,0,0.001],
                              [_z,0,v_z,z],
                              [0,0,0,1]])
            drawQuadArrow((0,0,0), frame)

        # shadow
        glPushMatrix()
        glTranslatef(0,-0.001,0) # below trajectory higher than the grid
        glMultMatrixf(shadow)
        drawCharacter((150,150,150), skel, pose)
        glPopMatrix()

        # Options
        taskInfo = self.motionStream.getLog('taskInfo')[-len(self.stack)-1]
        points = taskInfo['future']

        if options["r_desiredPoint"] and len(points):
            drawCircle((0,0,0), points[0][0], points[0][2], 2.1)

        for i, q in enumerate(taskInfo['queryPoints'][1:]):
            if options["r_maxLimitPoint"] and i == 2 and taskInfo['limited']:
                drawCircle((255,0,255), q[0], q[2], 2.1)
            if options["r_queryPoints"]:
                drawCircle((0,0,237), q[0], q[2], 2.1)
            if options["r_nonlimitedQueryPoints"] and i *10 < len(points):
                p = points[i*10]
                drawCircle((155,155,155), p[0], p[2], 2.1)
                

        if options["r_cornerPoint"] and taskInfo["cornerPoint"]:
            p = points[taskInfo["cornerPoint"]]
            drawCircle((0,176,80), p[0], p[2], 2.1)


    def setFeatureWeights(self, values):
        if not self.motionStream.setFeatureWeights(values):
            print("No motion matcher in this renderer:", self.name)

    def setNetwork(self, value):
        if not self.motionStream.setNetwork(value):
            print("No motion matcher in this renderer:", self.name)

    def drawBvh(self, node, pose, meshDrawing, color=(0,0,155)):
#255,102,255
#201,125,73
        glPushMatrix()

        x,y,z = np.array(node.offset)
        glTranslatef(x,y,z)

        if node.isRoot: # root align 
            self.nodeIndex = -1

            x,y,z = pose.getPosition()
            glTranslatef(x,y,z)


        if 'End' not in node.name:
            self.nodeIndex += 1

            orientation = pose.getOrientations()[self.nodeIndex]
            glMultTransposeMatrixf(orientation)

#            if node.isRoot:
#                glBegin(GL_LINES)
#                glColor3ub(255, 0, 0)
#                glVertex3fv(np.array([0., 0., 0.]))
#                glVertex3fv(np.array([1., 0., 0.]))
#                glColor3ub(0, 255, 0)
#                glVertex3fv(np.array([0., 0., 0.]))
#                glVertex3fv(np.array([0., 1., 0.]))
#                glColor3ub(0, 0, 255)
#                glVertex3fv(np.array([0., 0., 0.]))
#                glVertex3fv(np.array([0., 0., 1.]))
#                glEnd()

        for c in node.children:
            if meshDrawing:
                length = np.sqrt(sum(c.offset*c.offset))
                if length:
                    glPushMatrix()

                    T = np.identity(4)
                    T[:3,3] = c.offset / 2
                    glMultTransposeMatrixf(T @ c.rotationFromParent)

                    drawCube(color, (0,0,0), w=0.05, h=0.05, l=length)

                    glPopMatrix()
            else:
                glBegin(GL_LINES)
                glColor3ub(255, 255, 0)
                glVertex3fv([0.0, 0.0, 0.0])
                glVertex3fv(c.offset)
                glEnd()


            self.drawBvh(c, pose, meshDrawing, color)

        glPopMatrix()

    def drawArrows(self, frames, color=(0,0,237)):
        if frames is None:
            return
        for i, frame in enumerate(frames):
            frame[1,3] =0.001# On ground
            #drawQuadArrow((81*i,82*i,237), frame)
            drawQuadArrow(color, frame)
            #x,z = frame[[0,2],3]
            #drawCircle((0,0,235), x,z, scale=3)

    def reset(self):

        self.instant = np.array([0,0,1])

        self.emptyStack()
        self.motionStream.playReset()
        self.update({'name':'default'})

    def showCachePosture(self, back):
        postures = self.motionStream.getLog('posture')

        if back:
            if len(postures) - 1 > len(self.stack) + 1:
                self.stack.insert(0, 0)
        elif len(self.stack) > 2: # forth
            if self.stack[0] == self.stack[1]:
                self.stack.pop(0)

        return 0 

