# Requirement: wxPython version '4.1.0 gtk3 (phoenix) wxWidgets 3.1.4'

from OpenGL.GL import *
import numpy as np

# CUBE
#cubeVertices = np.array([
#    [1,1,1],
#    [1,1,-1],
#    [-1,1,-1],
#    [-1,1,1],
#    [-1,-1,1],
#    [1,-1,1],
#    [1,-1,-1],
#    [-1,-1,-1]], np.float32)
#cubeIndices = [0,1,2, 2,3,0, 0,4,5, 0,3,4, 1,0,5, 1,5,6, 2,3,4, 2,4,7, 1,7,6, 1,2,7, 7,4,6, 4,5,6]

cubeVertices = np.array([
    1, 1, 1,  -1, 1, 1,  -1,-1, 1,   1,-1, 1,   # (front)
    1, 1, 1,   1,-1, 1,   1,-1,-1,   1, 1,-1,   # (right)
    1, 1, 1,   1, 1,-1,  -1, 1,-1,  -1, 1, 1,   # (top)
    -1, 1, 1,  -1, 1,-1,  -1,-1,-1,  -1,-1, 1,   # (left)
    -1,-1,-1,   1,-1,-1,   1,-1, 1,  -1,-1, 1,   # (bottom)
    1,-1,-1,  -1,-1,-1,  -1, 1,-1,   1, 1,-1     # back
    ], np.float32)
cubeVertices *= 0.5
cubeNormals = np.array([
    0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,   # (front)
    1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,   # (right)
    0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,   # (top)
    -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,   # (left)
    0,-1, 0,   0,-1, 0,   0,-1, 0,   0,-1, 0,   # (bottom)
    0, 0,-1,   0, 0,-1,   0, 0,-1,   0, 0,-1  # (back)
    ], np.float32)
cubeIndices = [0,1,2, 2,3,0,
        4,5,6, 6,7,4, 
        8,9,10, 10,11,8,
        12,13,14, 14,15,12,
        16,17,18, 18,19,16,
        20,21,22, 22,23,20]

# QUAD
quadVertices = np.array([
    [0.5,0,0.5],
    [0.5,0,-0.5],
    [-0.5,0,-0.5],
    [-0.5,0,0.5]], np.float32)
quadIndices = [1,2,0, 0,2,3]

# Quad arrow
unit = np.pi / 2
quadBase = []
radius = 0.1
length = 0.4
#for i in range(4):
#    quadBase.append([round(radius*np.cos(unit*i),3), round(radius*np.sin(unit*i),3), 0])
#quadBase.append([0,0,length])
quadBase = [[.1,0,length/2],[-.1,0,length/2],[0,0,length],\
            [.02,0,.2], [.02,0,0], [-.02,0,0],[-.02,0,.2]]
quadBase = np.array(quadBase, np.float32)
#quadArrowIndices = [0,3,2, 2,1,0, 0,1,4, 1,2,4, 2,3,4, 3,0,4]
quadArrowIndices = [0,1,2, 3,5,6, 3,4,5]

#Circle
radius = 1 
CIRCLE_SIZE = 20
unit = 2*np.pi / CIRCLE_SIZE
circleVertices = [[0,0,0]]
for i in range(CIRCLE_SIZE):
    circleVertices.append([radius * np.cos(unit * i), 0, radius * np.sin(unit * i)])
circleVertices = np.array(circleVertices, np.float32)
circleIndices = np.append(np.arange(CIRCLE_SIZE+1), 1)

# Grid
height = -0.01
# ca,cb = (255/255, 255/255, 255/255), (210/255, 240/255, 255/255) # blue
#ca,cb = (255/255, 255/255, 255/255), (230/255, 235/255, 200/255) # green
ca,cb = (255/255, 255/255, 255/255), (242/255, 238/255, 222/255) # orange
#ca,cb = (255/255, 255/255, 255/255), (242/255, 242/255, 242/255) # red
#ca,cb = (255/255, 255/255, 255/255), (242/255, 242/255, 255/255) # purple
#ca,cb = (66/255, 70/255, 66/255), (53/255, 61/255, 62/255)
color = ca

gridColors = []
gridVertices = []
for i in range(-50, 50):
    for j in range(-50, 51):
        color = cb if color == ca else ca
        gridColors.append([color]*4)
        gridVertices.append([i, height, j])
        gridVertices.append([i+1, height, j])
        gridVertices.append([i+1, height, j+1])
        gridVertices.append([i, height, j+1])
gridColors = np.array(gridColors, np.float32)
gridVertices = np.array(gridVertices, np.float32)

def drawMeshGrid():
    global gridVertices, gridColors
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, False, gridColors.tobytes())
    glVertexPointer(3, GL_FLOAT, False, gridVertices.tobytes())
    glDrawArrays(GL_QUADS, 0, len(gridVertices))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

# Cylinder
up = circleVertices.copy()[1:] + [0, .5, 0] # Y up
down = circleVertices.copy()[1:] + [0, -.5, 0]
#up = np.vstack((up, up[-1]))
#down = np.vstack((down, down[-1]))
sideNormals = []
sideVertices = []
for i in range(len(up)):
    j = (i+1) % len(up)

    p1,p2,p3 = up[i], down[i], down[j]
    sideVertices.extend([p1,p2,p3])
    s,r = p2-p1, p3-p1
    normal = np.cross(s,r)
    sideNormals.extend([-normal]*3)

    p1,p2,p3 = down[j], up[j], up[i]
    sideVertices.extend([p1,p2,p3])
    s,r = p2-p1, p3-p1
    normal = np.cross(s,r)
    sideNormals.extend([-normal]*3)

sideVertices = np.array(sideVertices, dtype=np.float32)
sideNormals = np.array(sideNormals, dtype=np.float32)
sideIndices = np.array([[6*i+j for j in range(6)] for i in range(len(up))]).flatten()

def drawCylinder(color, centerPoint, radius, height):
    global sideVertices, sideNormals, sideIndices
    global circleVertices, circleNormals, circleIndices

    r,g,b = color
    glColor3ub(r,g,b)

    glRotatef(90, 1,0,0) # dartpy cylinder
    glScalef(radius, height, radius)

    glPushMatrix()
    x,y,z = centerPoint
    glTranslatef(x,y,z)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    glVertexPointer(3, GL_FLOAT, False, sideVertices.tobytes())
    glNormalPointer(GL_FLOAT, False, sideNormals.tobytes())
    glDrawElementsui(GL_TRIANGLES, sideIndices)

    glTranslatef(0,.5,0)
    glVertexPointer(3, GL_FLOAT, False, circleVertices.tobytes())
    glNormalPointer(GL_FLOAT, False, np.array([0,1,0]*len(circleVertices)).tobytes())
    glDrawElementsui(GL_TRIANGLE_FAN, circleIndices)
    glTranslatef(0,-1,0)
    glVertexPointer(3, GL_FLOAT, False, circleVertices.tobytes())
    glNormalPointer(GL_FLOAT, False, np.array([0,-1,0]*len(circleVertices)).tobytes())
    glDrawElementsui(GL_TRIANGLE_FAN, circleIndices)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

    glPopMatrix()


def drawCube(color, centerPoint, w=0.5, h=0.5, l=0.5):
    global cubeVertices, cubeNormals, cubeIndices

    # width, height, position xyz(center)
    r,g,b, = color
    glColor3ub(r,g,b)

    glPushMatrix()
    x,y,z = centerPoint
    glTranslatef(x,y,z)
    glScalef(w,h,l)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, False, cubeVertices.tobytes())
    glNormalPointer(GL_FLOAT, False, cubeNormals.tobytes())
    glDrawElementsui(GL_TRIANGLES, cubeIndices)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

    glPopMatrix()


def drawCircle(color, x, z, scale=1):
    r,g,b = color
    glColor3ub(r,g,b)
    glPushMatrix()
    glTranslatef(x,0,z)
    if scale != 1:
        glTranslatef(0,0.001*scale,0)
        glScalef(scale, 0, scale)
    glScalef(0.02, 0, 0.02) # default size

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, False, circleVertices.tobytes())
    glDrawElementsui(GL_TRIANGLE_FAN, circleIndices)
    glDisableClientState(GL_VERTEX_ARRAY)
    glPopMatrix()

def drawQuad(color, x, z, w=0.2, h=0.2):
    global quadVertices, quadIndices

    # width, height, position x, z(center)
    # quad lying on a plane(XZ Plane)
    r,g,b, = color
    glColor3ub(r,g,b)

    glPushMatrix()
    glTranslatef(x,0,z)
    glScalef(w, 0, h)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, False, quadVertices.tobytes())
    glDrawElementsui(GL_TRIANGLES, quadIndices)
    glDisableClientState(GL_VERTEX_ARRAY)
    glPopMatrix()

def drawQuadArrow(color, frame):
    global quadBase, quadArrowIndices
    glPushMatrix()

    glMultTransposeMatrixf(frame)

    r,g,b = color
    glColor3ub(r,g,b)

   #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, False, quadBase.tobytes())
    glDrawElementsui(GL_TRIANGLES, quadArrowIndices)
    glDisableClientState(GL_VERTEX_ARRAY)

    #glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glPopMatrix()

def drawLine(color:tuple, startPoint:tuple, endPoint:tuple):
    glBegin(GL_LINES)
    r,g,b = color
    glColor3ub(r,g,b)
    glVertex3fv(startPoint)
    glVertex3fv(endPoint)
    glEnd()
