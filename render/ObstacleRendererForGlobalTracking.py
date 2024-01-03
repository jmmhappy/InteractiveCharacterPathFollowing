from render.drawPrimitives import *
from render.RendererInterface import RendererInterface

from util.rotations import log

from data.ObstacleForGlobalTracking import ObstacleMap

class ObstacleRenderer(RendererInterface):
    def __init__(self, frametime):
        super().__init__(frametime)
        self.obstacles = ObstacleMap()

    def getFocusPoint(self):
        return None
    def reset(self):
        return self.obstacles.reset()
    def update(self, userInput=None):
        self.obstacles.move(self.FRAMETIME)
        return self.obstacles

    def render(self, shadow, options, observation):
        localframe = observation['localframe']
        verticalPosition = localframe[2,3]

        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, (70,100,50,1))

        self.drawObstacles((200,200,200))

        color = (255,0,0)
        if verticalPosition >= 18:
            color = (0,255,0)
        # draw goal line
        drawCube(color, (0,1,19), w=20,h=2,l=.2)

        glDisable(GL_LIGHTING)

        # shadow
        glPushMatrix()
        glTranslatef(0, -0.001, 0)
        glMultMatrixf(shadow)

        gray = (100,100,100)
        self.drawObstacles(gray)
        glPopMatrix()


    def drawObstacles(self, color):
        info = self.obstacles.getRenderInfo()
        center_2d, length = info['center'], info['length']
        for p, l in zip(center_2d, length):
            height = 1.5
            center = (p[0], height/2, p[1])
            drawCube(color, center, w=l[0], h=height, l=l[1])

