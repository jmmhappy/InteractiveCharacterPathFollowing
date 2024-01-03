from render.drawPrimitives import *
from render.RendererInterface import RendererInterface

from data.Obstacle import SIZE, RANGE
mapVertices = []
for i in np.arange(-RANGE/2, RANGE/2, RANGE/SIZE):
    for j in np.arange(-RANGE/2, RANGE/2, RANGE/SIZE):
        delta = 0.05
        mapVertices.append([i+delta, 0, j+delta])
        mapVertices.append([i-delta, 0, j+delta])
        mapVertices.append([i-delta, 0, j-delta])
        mapVertices.append([i+delta, 0, j-delta])
mapVertices = np.array(mapVertices, dtype=np.float32)

from util.rotations import log

class ObstacleRenderer(RendererInterface):
    def __init__(self, obstacles, frametime):
        super().__init__(frametime)
        self.obstacles = obstacles

    def getFocusPoint(self):
        return None
    def reset(self):
        return self.obstacles.reset()
    def update(self, userInput=None):
        self.obstacles.move(self.FRAMETIME)
        return self.obstacles

    def render(self, shadow, options, observation):
        localframe = observation['localframe']
        localframe[1][3] = 0 # Y up, projection

        glEnable(GL_LIGHTING)
        self.drawObstacles((200,200,200))

        glDisable(GL_LIGHTING)

        self.drawCollisionMap(localframe)

        # shadow
        glPushMatrix()
        glTranslatef(0, -0.001, 0)
        glMultMatrixf(shadow)
        self.drawObstacles((100,100,100))
        glPopMatrix()


    def drawObstacles(self, color):
        info = self.obstacles.getRenderInfo()
        center_2d, length = info['center'], info['length']
        for p, l in zip(center_2d, length):
            height = 1.5
            center = (p[0], height/2, p[1])
            drawCube(color, center, w=l[0], h=height, l=l[1])

    def drawCollisionMap(self, localFrame):
        global mapVertices
        terrain = self.obstacles.getLocalMap(localFrame)
        terrain = np.squeeze(terrain)
        mapColors = np.full((len(mapVertices), 3), (0,1,0)) # all green
        
        glPushMatrix() 
        x,y,z,_ = localFrame[:,3]
        glTranslatef(x,y,z)
        theta = log(localFrame)[1] # Y-up
        glRotatef(np.degrees(theta),0.,1.,0.)

        for i, flag in enumerate(terrain.flatten()):
            if flag:
                mapColors[i*4:i*4+4] = [(1,0,0)]*4 # red, GL_QUADS
        mapColors = np.array(mapColors, dtype=np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, False, mapColors.tobytes())
        glVertexPointer(3, GL_FLOAT, False, mapVertices.tobytes())
        glDrawArrays(GL_QUADS, 0, len(mapVertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glPopMatrix()


