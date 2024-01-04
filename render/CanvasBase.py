import random
import sys
import time

import wx, wx.adv
import wx.glcanvas as glcanvas

try:
    from OpenGL.GL import *
#    from OpenGL.GLUT import *
    haveOpenGL = True
except ImportError:
    haveOpenGL = False


from render.MousePicker import MousePicker
from render.drawPrimitives import *
from render.MotionRenderer import MotionRenderer

import numpy as np
from itertools import combinations

class CanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent):
        attribList = (glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_DEPTH_SIZE, 24)
        glcanvas.GLCanvas.__init__(self, parent,  attribList=attribList)
        self.init = False
        self.context = glcanvas.GLContext(self)

        self.SetMinSize((800,800))
        # self.SetMinSize((1800,1800))
        # self.SetMinSize((3300,2000))
        self.isPlaying = False
        self.options = {
            'characterDrawing':True,
            'meshDrawing':True, 
            'logDrawing':False, 
            'record':False,
            'limitedUpdate':False,
            'startFromNearest':False,
            'amplifyCurve':False,
            'extendedSearchParam':{'K':1, 'L':1},
            'extendedSearch':False,
            'forceQuery':False,
            'scalePoints':False,
            'smoothInput':False,
            'localControl':False,

            'joystick':None,

            # render options
            'r_desiredPoint':False,
            'r_nonlimitedQueryPoints':False,
            'r_maxLimitPoint':False,
            'r_arrows':False,
            'r_queryPoints':False,
            'r_cornerPoint':False,
            'r_rootTrajectory':False
        }

        # reinforcement learning
        self.observation = {'target':None, 'map':None, 'localframe':np.identity(4)} 

        # --- Recorders ---

        size = self.GetSize()
        width, height = size.width, size.height
        self.screenInfo = {
            'screen': wx.ScreenDC(),
            'memory': wx.MemoryDC(),
            'bitmap': wx.Bitmap(width, height)
        }

        # --- Renderers ---
        self.renderers = []

        self.characterNames = []
        self.pickers = []

        self.characterInAction = -1

        self.slider = None # pointer copy from panel
        self.taskLabel = None # static txt copy from panel 

        # --- Projection Shadow ---
        shadow = np.identity(4)

        ground = np.array([0,1,0,0])
        light = np.array([.7,1,.5,0])

        dot = np.dot(ground, shadow)
        shadow[:,0] -= light[0] * ground
        shadow[:,1] -= light[1] * ground
        shadow[:,2] -= light[2] * ground
        shadow[:,3] -= light[3] * ground

        self.shadow = shadow


        # --- Camera Constants ---
        self.panX = self.panY = 0
        self.rotX = 30
        self.rotY = -15
        self.zoom = -40

        self.isFocusOn = False

        self.lastx = self.x = 0
        self.lasty = self.y = 0
        self.size = None
        self.isPanning = False 
        self.isRotating = False
        self.isPicking = False

        # --- Key Flags ---
        self.dkeyflags = {'left':False, 'right':False, 'up':False, 'down':False}
        self.ctrlKeyPressed = False

        # --- BINDINGS ---

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.timer = wx.Timer(self, id=1)
        self.Bind(wx.EVT_TIMER, self.OnTimer)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_RIGHT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

        # joystick
        self.stick = wx.adv.Joystick(wx.JOYSTICK1)
        self.stick.SetCapture(self)
        self.Bind(wx.EVT_JOYSTICK_EVENTS, self.OnJoystick)
        self.stick.SetMovementThreshold(1)

        # instant tangent at current position, of 2dim input trajectory
        self.instant = np.array([0,0,1])

    def InitGL(self):
        def gluPerspective(fovy, aspect, near, far):
            # fovy in radian
            h = 2 * near * np.tan(fovy/2)
            top = h; bottom = -h
            left = aspect * bottom; right = aspect * top
            glFrustum(left,right,bottom,top,near,far)

        # set viewing projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        size = self.GetSize()
        aspect = size.width / size.height
        gluPerspective(np.pi/10, aspect, 1, 100)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)


    def OnJoystick(self, evt):
        if evt.IsButton():
            return

        if evt.IsMove():
            position = np.array(evt.GetPosition())
            length = np.linalg.norm(position)
            x,z = position / length

            self.options['joystick'] = np.array([x,0,-z])


    def OnTimer(self, event):

        def setTaskFromInput(rendererID):
            name = 'None' 
            inputTrajectory = self.pickers[rendererID].getTrajectory(self.options['smoothInput'])
            characterPosition = renderer.getFocusPoint()

            if self.observation['target'] is not None:
                self.renderers[rendererID].goal = self.observation['target']
                self.observation['target'] = None
                name = 'position'

            if sum(self.dkeyflags.values()) > 0:
                name = 'direction'

            if not self.pickers[rendererID].isEmpty():
                name = 'trajectory'
                inputTrajectory = self.pickers[rendererID].getTrajectory(self.options['smoothInput'])

                if self.options['forceQuery']:
                    index = self.renderers[rendererID].getFutureTrajectoryIndex()
                    p1, p2 = inputTrajectory[index:index+2]

                    instant = -p1 + p2
                    d = np.sqrt(sum(instant*instant))
                    if d > 0:
                        instant /= d
                    self.renderers[rendererID].forceQuery(instant)


            userInput = {
                'name':name,
                'position':self.renderers[rendererID].goal,
                'obstacles':self.observation['map'],
                'direction':self.dkeyflags,
                'trajectory':inputTrajectory,

                'amplifyCurve':self.options['amplifyCurve'],
                'joystick':self.options['joystick'],
                'startFromNearest':self.options['startFromNearest'],
                'limitedUpdate' :self.options['limitedUpdate'],
                'extendedSearch':{
                    "flag":self.options['extendedSearch'],
                    "K":self.options['extendedSearchParam']['K'],
                    "L":self.options['extendedSearchParam']['L'],
                },
            }

            self.taskLabel.SetLabel('Task: ' + name)
            return userInput

        self.SetFocus() # panel -> canvas

        characters = []
        for i, renderer in enumerate(self.renderers):
            userInput = None
            if isinstance(renderer, MotionRenderer):
                userInput = setTaskFromInput(i)
                characters.append(i)

            #if i == self.characterInAction:
                nextFrameNum = renderer.update(userInput)
                self.slider.SetValue(nextFrameNum)

                self.observation['localframe'] = renderer.getCurrentLocalFrame()

                if self.observation['map']:#and userInput['name'] == "position":
                    pose = renderer.motionStream.getLog('posture')[-1]
                    nodePos = renderer.motionStream.motionMatcher.getNodePositions(nextFrameNum)
                    if self.observation['map'].collision(pose, nodePos):
                        renderer.reset()
                        break
            else:
                renderer.update(userInput)

        for a,b in combinations(characters,2):
            p1 = self.renderers[a].getFocusPoint()
            p2 = self.renderers[b].getFocusPoint()
            if np.linalg.norm(p1-p2) < 0.5: # two characters collide
                self.renderers[a].reset()
                self.pickers[a].reset(True)
                self.renderers[b].reset()
                self.pickers[b].reset(True)

        self.Refresh(False)

    def OnIdle(self, event):
        def capture():
            x,y = self.GetParent().ClientToScreen((0,0))
            size = self.GetSize()
            w,h = size.width, size.height

            mem = self.screenInfo['memory']
            bmp = wx.Bitmap(w,h)#self.screenInfo['bitmap']

            mem.SelectObject(bmp)
            mem.Blit(
                0, 0,
                w, h,
                self.screenInfo['screen'],
                x, y
            )
            mem.SelectObject(wx.NullBitmap)
            bmp.SaveFile('screenshots/%s.png'%str(time.time()), wx.BITMAP_TYPE_PNG)

        if self.isPlaying and self.options['record']:
            capture() 


    def addRenderer(self, r, label):
        self.renderers.append(r)

        if not isinstance(r, MotionRenderer): # Obstacle renderer
            self.observation['map'] = r.obstacles
            self.pickers.append(None)
            return

        self.characterNames.append(label)
        self.pickers.append(MousePicker(canvasSize=self.GetMinSize()))

        it = len(self.renderers) - 1
        self.markCharacterSelection(it)


    def markCharacterSelection(self, it):
        assert(isinstance(self.renderers[it], MotionRenderer))

        ch, nextCh = self.renderers[self.characterInAction], self.renderers[it]
        if ch != nextCh and ch.COLOR == nextCh.COLOR:
            nextCh.COLOR = (0,255,0)
        else:
            ch.COLOR, nextCh.COLOR = nextCh.COLOR, ch.COLOR

        self.characterInAction = it 

#        for i, name in enumerate(self.characterNames):
#            if isinstance(self.renderers[i], MotionRenderer):
#                r,g,b = self.renderers[i].COLOR
#                if i == self.characterInAction:
#                    name.SetFont(wx.Font().MakeItalic())
#                else:
#                    name.SetFont(wx.Font())
#                name.SetForegroundColour(wx.Colour(r,g,b))


    def renderAll(self):
        for r in self.renderers:
            r.render(self.shadow, self.options, self.observation)


    def getFrameNumberLog(self):
        return self.renderers[self.characterInAction].getFrameNumberLog()

    def getFocusPoint(self):
        return self.renderers[self.characterInAction].getFocusPoint()

    def getFrameTime(self):
        return self.renderers[self.characterInAction].FRAMETIME

    def getMouseTrajectory(self):
        return self.pickers[self.characterInAction].getTrajectory(False)
    def setMouseTrajectory(self, t):
        picker = self.pickers[self.characterInAction]
        picker.setTrajectory(t)

        characterPosition = self.getFocusPoint()
        picker.adjust(characterPosition, self.options['localControl'])

    def setFrameNum(self, value):
        try:
            self.renderers[self.characterInAction].setCurrentFrameNum(value) # bvh
        except Exception:
            pass # particle sys

    def playReset(self):
        assert(len(self.renderers) == len(self.pickers))
        for i in range(len(self.renderers)):
            if self.pickers[i]:
                self.renderers[i].reset()
                self.pickers[i].reset()

        self.Refresh(False)


    def stopTimer(self):
        self.isPlaying = False
        self.timer.Stop()

    def startTimer(self):
        self.isPlaying = True
        interval = round(self.getFrameTime() * 1000)
        if self.options['record']:
            interval *= 10 # slow enough to Bitmap() every frame

        self.timer.Start(interval)
        self.Refresh(False)
        self.SetFocus() # panel -> canvas


    def OnPaint(self, event):
        #dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

    def OnDraw(self):
        def lookAtMatrix(eye, at, up):
            forward = eye - at
            forward = forward / np.sqrt(sum(forward * forward))
            side = np.cross(forward, up)
            side = side / np.sqrt(sum(side * side))
            camUp = np.cross(side, forward)
            pos = np.array([np.dot(-eye,side), np.dot(-eye,camUp), np.dot(-eye,forward)])
            M = np.identity(4)
            M[:3,0] = side
            M[:3,1] = camUp
            M[:3,2] = forward
            M[3,:3] = pos
            return M

        def cameraTransform():
            glTranslatef(0, 0, self.zoom)

            glTranslatef(self.panX, self.panY, 0)
            glRotatef(self.rotX, 1.0, .0, .0)
            glRotatef(self.rotY, .0, 1., .0)


        # ------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#        glClearColor(255,255,255,1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.options["joystick"] is not None:
            self.drawJoystickInput()

        if self.isFocusOn:
            # Track only xz-plane(Y-UP)
            at = self.getFocusPoint() * [1,0,1]
            eye = np.array([10,10, 0])
            up = np.array([0,1,0])
            glMultMatrixf(lookAtMatrix(eye, at, up))

        else:
            if self.size is None:
                self.size = self.GetClientSize()
            w, h = self.size
            w = max(w, 1.0)
            h = max(h, 1.0)
            xScale = 90.0 / w
            yScale = 90.0 / h

            if self.isRotating:
                self.rotX += (self.y - self.lasty) * yScale
                self.rotY += (self.x - self.lastx) * xScale

            if self.isPanning:
                xScale *= 0.1
                yScale *= 0.1
                self.panX += (self.x - self.lastx) * xScale
                self.panY -= (self.y - self.lasty) * yScale

            cameraTransform()

        self.drawGrid()
        # self.drawFrame()
        self.renderAll()

        self.drawTrajectories()

        self.SwapBuffers()


    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key == wx.WXK_LEFT: 
            self.dkeyflags['left'] = True
        elif key == wx.WXK_RIGHT:
            self.dkeyflags['right'] = True
        elif key == wx.WXK_UP:
            self.dkeyflags['up'] = True
        elif key == wx.WXK_DOWN:
            self.dkeyflags['down'] = True

        elif key == wx.WXK_CONTROL:
            self.ctrlKeyPressed = True
            self.observation['target'] = None

        elif key == wx.WXK_ESCAPE:
            self.pickers[self.characterInAction].reset(clean=True)
            self.renderers[self.characterInAction].setFutureTrajectoryIndex(0) # reset picker index
        elif key == wx.WXK_SPACE:
            if self.isPlaying:
                self.stopTimer()
            else:
                self.startTimer()
        elif key == wx.WXK_BACK:
            self.rewind(back=True)
        elif key == wx.WXK_TAB:
            self.rewind(back=False)

        elif key == wx.WXK_SHIFT:
            it = (self.characterInAction + 1) % len(self.renderers)
            while not isinstance(self.renderers[it], MotionRenderer):
                it = (it + 1) % len(self.renderers) 

            self.markCharacterSelection(it)
            self.Refresh(False)


    def rewind(self, back=True):
        for r in self.renderers:
            try: # if Motion
                count = r.showCachePosture(back)
                self.Refresh(False)
                return count
            except Exception: # else particle
                return -1

    def OnKeyUp(self, evt):
        key = evt.GetKeyCode()
        if key == wx.WXK_LEFT:
            self.dkeyflags['left'] = False
        elif key == wx.WXK_RIGHT:
            self.dkeyflags['right'] = False
        elif key == wx.WXK_UP:
            self.dkeyflags['up'] = False
        elif key == wx.WXK_DOWN:
            self.dkeyflags['down'] = False
        elif key == wx.WXK_CONTROL:
            self.ctrlKeyPressed = False

    def setOptions(self, key, value):
        if self.options.get(key) is not None:
            self.options[key] = value

    def setExtendedSearch(self, val):
        K, L = val.values()
        print("Set extended motion matching parameters in K = %d, L = %d"%(K,L))
        self.options['extendedSearchParam']['K'] = K
        self.options['extendedSearchParam']['L'] = L

    def setNetwork(self, value):
        for r in self.renderers:
            if isinstance(r, MotionRenderer):
                r.setNetwork(value)

    def setFeatureWeights(self, values):
        for r in self.renderers:
            try:
                r.setFeatureWeights(values) # only works when type: MotionRenderer
            except Exception as e:
                print(e)

    def setCameraFocus(self, value):
        assert(isinstance(value, bool))
        self.isFocusOn = value
        self.Refresh(False)
        self.SetFocus()

    def setRecorder(self, value):
        self.options['record'] = True


    def setSlider(self, slider):
        self.slider = slider
    def setTaskLabel(self, taskLabel):
        self.taskLabel = taskLabel

    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        self.InitGL()
        event.Skip()


    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)


    def OnMouseDown(self, evt):
        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()

        if evt.RightIsDown() and self.ctrlKeyPressed:
            picker = self.pickers[self.characterInAction]
            picker.pickRay(self.x, self.y, \
                np.array(glGetFloatv(GL_PROJECTION_MATRIX)).T,\
                np.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
            point = picker.getIntersection()

            self.observation['target'] = tuple(point[[0,2]])
            self.Refresh(False)

    def OnMouseUp(self, evt):
        self.ReleaseMouse()
        if self.isPicking:
            characterPosition = self.getFocusPoint()
            self.pickers[self.characterInAction].adjust(characterPosition, self.options['localControl'])
            self.renderers[self.characterInAction].setFutureTrajectoryIndex(0) # reset picker index
            self.pickers[self.characterInAction].scale(characterPosition, self.options['localControl'], self.options["scalePoints"])
            self.renderers[self.characterInAction].emptyStack()
        self.isPanning = self.isRotating = self.isPicking = False

    def OnMouseWheel(self, evt):
        self.zoom += evt.GetWheelRotation() * 0.01 # 1.2 or -1.2
        self.Refresh(False)

    def OnMouseMotion(self, evt):
        if evt.Dragging():
            self.isPanning = evt.LeftIsDown() and not self.ctrlKeyPressed
            self.isRotating = evt.RightIsDown()
            self.isPicking = evt.LeftIsDown() and self.ctrlKeyPressed

            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = evt.GetPosition()

            # Mouse picker
            if self.isPicking:
                picker = self.pickers[self.characterInAction]
                picker.pickRay(self.x, self.y, \
                    np.array(glGetFloatv(GL_PROJECTION_MATRIX)).T,\
                    np.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
                picker.addIntersection()

            self.Refresh(False)

    def drawGrid(self):
        height = -0.01
        ca, cb = (66, 70, 66), (53, 61, 62)
        color = ca

        if self.options['meshDrawing']:
            drawMeshGrid()
        else:
            glBegin(GL_LINES)
            glColor3ub(255,255,255)
            for i in range(-50, 50):
                glVertex3fv(np.array([-50, height, i]))
                glVertex3fv(np.array([50, height, i]))
                glVertex3fv(np.array([i, height, -50]))
                glVertex3fv(np.array([i, height, 50]))
            glEnd()

    def drawFrame(self):
        glBegin(GL_LINES)
        glColor3ub(255, 0, 0)
        glVertex3fv(np.array([0., 0., 0.]))
        glVertex3fv(np.array([10., 0., 0.]))
        glColor3ub(0, 255, 0)
        glVertex3fv(np.array([0., 0., 0.]))
        glVertex3fv(np.array([0., 10., 0.]))
        glColor3ub(0, 0, 255)
        glVertex3fv(np.array([0., 0., 0.]))
        glVertex3fv(np.array([0., 0., 10.]))
        glEnd()

    def drawTrajectories(self):
        def drawLines(color, points):
            r,g,b = color
            glColor3ub(r,g,b)
            glLineWidth(1)
            glBegin(GL_LINES)
            for i in range(0, len(points)-1):
                p1, p2 = points[i:i+2]
                glVertex3fv(p1)
                glVertex3fv(p2)
            glEnd()
            glLineWidth(1)

        # def drawLinesWithColors(colors, points):
            # assert(len(colors) == len(points))
            # # glLineWidth(5)
            # glBegin(GL_LINES)
            # for i in range(0, len(points)-1):
                # p1, p2 = points[i:i+2]
                # r, g, b = colors[i]
                # r = 255 - b

                # glColor3ub(r,g,b)
                # glVertex3fv(p1)
                # glVertex3fv(p2)
            # glEnd()
            # glLineWidth(1)

        for renderer, picker in zip(self.renderers, self.pickers):

            glLineWidth(5)
            drawLines((255,28,0), picker.getTrajectoryOnMove()) # of mouse
            glLineWidth(1)

            if picker.isEmpty():
                continue
            mouseTrajectory = np.array(picker.getTrajectory(self.options['smoothInput']))
            mouseTrajectory[:, 1] = 0.00
            drawLines((255,28,0), mouseTrajectory)

            for i in range(0, len(mouseTrajectory)):
                p = mouseTrajectory[i]
                drawCircle((255,0,0), p[0], p[2])

#            comparison = picker.getTrajectory(False)
#            drawLines((100,100,100), comparison)
#            for p in comparison:
#                drawCircle((100,100,100), p[0], p[2])

#        if len(mouseTrajectory): # speed coloring
#            dists = np.linalg.norm(mouseTrajectory[1:] - mouseTrajectory[:-1], axis=1)
#            colors = np.clip(dists * 1000, 0, 255)
#            colors_3d = np.zeros(mouseTrajectory.shape, dtype=int)
#            colors_3d[:-1,2] = colors
#            drawLinesWithColors(colors_3d, mouseTrajectory)


    # CIRCLE_SIZE = 20
    # unit = 2*np.pi / CIRCLE_SIZE
    verticalCircleVertices = [[0,0,0]]
    for i in range(CIRCLE_SIZE):
        verticalCircleVertices.append([np.cos(unit*i), np.sin(unit*i), 0])
    verticalCircleVertices = np.array(verticalCircleVertices, np.float32).tobytes()
    verticalCircleIndices = np.append(np.arange(CIRCLE_SIZE+1), 1)

    def drawJoystickInput(self):
        def drawVerticalCircle():
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, False, CanvasBase.verticalCircleVertices)
            glDrawElementsui(GL_TRIANGLE_FAN, CanvasBase.verticalCircleIndices)
            glDisableClientState(GL_VERTEX_ARRAY)

        glPushMatrix()
        glTranslatef(-.4,-.4,-2)

        glPushMatrix()
        glColor3ub(0,255,0)
        glScalef(0.02,0.02,0)
        x, _, y = self.options["joystick"] * 4 

        glTranslatef(x,y,0)
        drawVerticalCircle()
        glPopMatrix()

        glColor3ub(200,200,200)
        glScalef(0.1,0.1,0)
        drawVerticalCircle()

        glPopMatrix()

