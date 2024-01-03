
import sys

import wx
import wx.glcanvas as glcanvas

try:
    from OpenGL.GL import *
#    from OpenGL.GLUT import *
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
        self.COLOR = (0,0,155)#np.asarray(np.random.rand(3) * 255, dtype=int)
#        self.COLOR = (158,15,15) # red
#        self.COLOR = (0,155,0) # green
#        self.COLOR = (200,100,30) # orange
#        self.COLOR = (10,10,30) # purple

        self.stack = []
        self.motionStream = db 

        self.instant = np.array([0,0,1])

        self.goal = None#np.zeros(2) 

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
    def setFutureTrajectoryIndex(self, value):
        self.motionStream.setFutureTrajectoryIndex(value)
    def getFutureTrajectoryIndex(self):
        return self.motionStream.getFutureTrajectoryIndex()
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
            if not options['characterDrawing']:
                return

            self.drawBvh(skel, pose, options['meshDrawing'], color) 

            if options['logDrawing']:
                for i in range(2, len(postures)-1, LOG_STRIDE):
                    pose = postures[i]
                    self.drawBvh(skel, pose, options['meshDrawing'], color)

        def drawGoal():
            if self.goal is None:return
#            if sum(self.goal) == 0:
#                self.setGoal(observation['map'])

            x,z = self.goal
            if np.linalg.norm(self.goal - self.getFocusPoint()[[0,2]]) < 0.5:
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
            self.drawArrows(futureTrajectory)

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

        if options["r_rootTrajectory"]:
            self.drawTrajectory()


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

    def drawArrows(self, frames):
        if frames is None:
            return
        for i, frame in enumerate(frames):
            frame[1,3] =0.001# On ground
            #drawQuadArrow((81*i,82*i,237), frame)
            drawQuadArrow((0,0,237), frame)
            #x,z = frame[[0,2],3]
            #drawCircle((0,0,235), x,z, scale=3)

    def drawTrajectory(self):
        def drawLines(color, points):
            r,g,b = color
            glColor3ub(r,g,b)
            glLineWidth(5)
            glBegin(GL_LINES)
            for i in range(0, len(points)-1):
                p1, p2 = points[i:i+2]
                glVertex3fv(p1)
                glVertex3fv(p2)
            glEnd()
            glLineWidth(1)

        r,g,b = self.COLOR
        drawLines((r,g,b), self.motionStream.getLog('root')[:-len(self.stack) - 1])


    def reset(self):

        self.instant = np.array([0,0,1])

        self.emptyStack()
        self.motionStream.playReset()
        self.update(None)

    def showCachePosture(self, back):
        postures = self.motionStream.getLog('posture')

        if back:
            if len(postures) - 1 > len(self.stack) + 1:
                self.stack.insert(0, 0)
        elif len(self.stack) > 2: # forth
            if self.stack[0] == self.stack[1]:
                self.stack.pop(0)

        return 0 

    def setGoal(self, obstacles):
        radius = 10

        if not obstacles:
            return np.random.uniform(-radius, radius, 2)

        info = obstacles.getRenderInfo()
        center, length = info['center'], info['length']
        low, high = center - length/2 , center + length/2

        while True: 
            goal = [np.random.uniform(-radius, radius), np.random.uniform(3,radius)]
            intersection = ((low < goal) & (goal < high)).T
            intersection = intersection[0] & intersection[1]
            if np.sum(intersection) == 0: # no intersection
                break
        self.goal += goal
