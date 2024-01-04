import wx
import wx.lib.scrolledpanel as scrolledpanel
from render.CanvasBase import CanvasBase
from render.MotionRenderer import MotionRenderer

from collections import OrderedDict

import data.Task

class CanvasPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        box = wx.BoxSizer(wx.VERTICAL)

        # Canvas
        topBox = wx.BoxSizer(wx.HORIZONTAL)

        canvas = CanvasBase(self)
        self.canvas = canvas
        panelWidth = parent.GetSize().GetWidth() - canvas.GetMinSize().GetWidth()
        panelHeight = canvas.GetMinSize().GetHeight()
        topBox.Add(canvas, 0, wx.EXPAND|wx.ALL)
        

        # Options 
        optionBox = wx.BoxSizer(wx.VERTICAL)

        scrolledPanel = scrolledpanel.ScrolledPanel(self, wx.ID_ANY,\
                size=(panelWidth, panelHeight))
        self.scrolledPanel = scrolledPanel
        scrolledPanel.SetupScrolling()
        scrolledPanel.SetSizer(optionBox)

        self.addResetAndSaveButtons(optionBox)
        self.addFocusCharacterCheckBox(optionBox)
        self.addRenderMethodRadioBoxes(optionBox)
        self.addLogRadioBoxes(optionBox)
        self.addQueryOptionCheckBoxes(optionBox)
        self.addVelocitySlider(optionBox)
        self.addRenderOptionCheckBoxes(optionBox)
        self.addCharacterBoard(optionBox)

        self.K = self.L = None # text ctrls
        self.addExtendedSearchParameters(optionBox)

        optionBox.AddSpacer(20)
        taskLabel = wx.StaticText(scrolledPanel, label='')
        optionBox.Add(taskLabel, 0, wx.ALIGN_CENTER)

        topBox.Add(scrolledPanel)
        # topBox.Add(optionBox, 0)
        box.Add(topBox, 0)
        box.AddSpacer(20)

        # Slider
        slider = wx.Slider(self, value=0, minValue=0, maxValue=0,\
            style=wx.SL_HORIZONTAL|wx.SL_LABELS, name="slider")
        box.Add(slider, 0, wx.EXPAND|wx.ALL)
        self.Bind(wx.EVT_SCROLL, self.OnSliderChanged, slider)

        # Player 
        buttonBox = wx.BoxSizer(wx.HORIZONTAL)
        self.addPlayerButtons(buttonBox)
        box.Add(buttonBox, 0, wx.CENTER)

        # Show
        box.SetSizeHints(self)
        self.SetSizer(box)

        self.slider = slider

        self.canvas.setTaskLabel(taskLabel)
        self.canvas.setSlider(slider)
        # initialize K, L
        self.canvas.setExtendedSearch({'K':2, 'L':3})

    def getMouseTrajectory(self):
        return self.canvas.getMouseTrajectory()
    def setMouseTrajectory(self, t):
        self.canvas.setMouseTrajectory(t)

    def getSliderLog(self):
        return self.canvas.getFrameNumberLog()

    def OnRadioBox(self, evt):
        rb = evt.GetEventObject()
        selection = rb.GetStringSelection()

        if selection == 'None':
            self.canvas.setOptions('characterDrawing', False)
            return
        else:
            self.canvas.setOptions('characterDrawing', True)

        # Mesh drawing
        if selection == 'Line':
            self.canvas.setOptions('meshDrawing', False)
        elif selection == 'Mesh':
            self.canvas.setOptions('meshDrawing', True)

        # Log drawing
        elif selection == 'Current Only':
            self.canvas.setOptions('logDrawing', False)
        elif selection == 'With Log':
            self.canvas.setOptions('logDrawing', True)

        self.canvas.SetFocus()

    def OnButton(self, evt):
        bId = evt.GetId()
        if bId == 0: # Pause
            self.canvas.stopTimer()
        elif bId == 1: # Play
            self.canvas.startTimer()
        elif bId == 2: # Reset
            self.canvas.playReset()
#        elif bId == 3: # Save db with weights
#            dig = wx.TextEntryDialog(self, 'Enter a New Name for this database.')
#            if dig.ShowModal() == wx.ID_OK and len(dig.GetValue()) > 0:
#                name = dig.GetValue()
#            else:
#                wx.MessageDialog(self, 'Failed to Export!').ShowModal()
#                return False
#            dig.Destroy()
#
#            bvhMenu.saveMotionDB(name, db) 
#            wx.MessageDialog(self.frame, "Done!").ShowModal()
        elif bId == 4: # Prev
            rewindIndex = self.canvas.rewind(back=True)
            self.rewindText.SetLabel(str(rewindIndex))
        elif bId == 5: # Next
            rewindIndex = self.canvas.rewind(back=False)
            self.rewindText.SetLabel(str(rewindIndex))
#        elif bId == 6: # weights
#            self.canvas.setFeatureWeights([textctrl.GetValue() for textctrl in self.weights.values()])
        elif bId == 19: # extended search parameters
            self.canvas.setExtendedSearch({'K':int(self.K.GetValue()), 'L':int(self.L.GetValue())})

        self.canvas.SetFocus()
        self.canvas.Refresh(False)

    def OnFocusCheckBox(self, evt):
        cbId = evt.GetId()
        cbObj = evt.GetEventObject()
        if cbId == 1: # focus camera on character
            self.canvas.setCameraFocus(cbObj.GetValue())
        elif cbId == 2: # record by screenshots
            self.canvas.setRecorder(cbObj.GetValue())
        self.canvas.SetFocus()
        self.canvas.Refresh(False)

    def OnQueryCheckBox(self, evt):
        identifier = evt.GetId()
        obj = evt.GetEventObject()

        if identifier == 3:
            self.canvas.setOptions('limitedUpdate', obj.GetValue())
        elif identifier == 4:
            self.canvas.setOptions('startFromNearest', obj.GetValue())
        elif identifier == 7:
            self.canvas.setOptions('amplifyCurve', obj.GetValue())
        elif identifier == 11:
            self.canvas.setOptions("extendedSearch", obj.GetValue())
        elif identifier == 5:
            self.canvas.setOptions('forceQuery', obj.GetValue())
        elif identifier == 6:
            self.canvas.setNetwork(obj.GetValue())

        elif identifier == 10:
            self.canvas.setOptions("scalePoints", obj.GetValue())
        elif identifier == 8:
            self.canvas.setOptions("smoothInput", obj.GetValue())
        elif identifier == 9:
            self.canvas.setOptions("localControl", obj.GetValue())

        self.canvas.SetFocus()
        self.canvas.Refresh(False)

    def OnRenderCheckBox(self, evt):
        identifier = evt.GetId()
        obj = evt.GetEventObject()

        name = None
        if identifier == 12:
            name = "desiredPoint"
        elif identifier == 13: 
            name = "nonlimitedQueryPoints"
        elif identifier == 14: 
            name = "maxLimitPoint"
        elif identifier == 15: 
            name = "arrows"
        elif identifier == 16: 
            name = "queryPoints"
        elif identifier == 17: 
            name = "cornerPoint"
        elif identifier == 18:
            name = "rootTrajectory" 

        self.canvas.setOptions('r_'+name, obj.GetValue())
        self.canvas.SetFocus()
        self.canvas.Refresh(False)

    def OnSliderChanged(self, evt):
        pos = evt.GetPosition()
        self.canvas.setFrameNum(pos)
        self.slider.SetValue(pos)
        self.canvas.Refresh(False)
        self.canvas.SetFocus()

    def OnVelocitySliderChanged(self, evt):
        vel = evt.GetPosition() / 10
        data.Task.maxVelocity = vel

        self.canvas.SetFocus()

    def appendRenderer(self, name, renderer):
        self.Layout() # Refresh

        text = wx.StaticText(self.scrolledPanel, id=wx.ID_ANY, label=name)
        self.characterBox.Add(text)
        self.canvas.addRenderer(renderer, text)

        print('Added a renderer:')
        try:
            self.slider.SetMax(max(renderer.TOTAL_FRAMECOUNT, self.slider.GetMax()))
            print('type: Motion')
        except Exception:
            print('type: Something else(skel, particle system)')


    # -------------------------------------------------------------------------------

    def addPlayerButtons(self, buttonBox):
        buttonPrev = wx.Button(self, 4, 'Prev')
        buttonPlay = wx.Button(self, 1, 'Play') # id and name
        buttonPause = wx.Button(self, 0, 'Pause')
        buttonNext = wx.Button(self, 5, 'Next')

        self.rewindText = wx.StaticText(self,label="")

        buttonBox.Add(buttonPrev, 0)
        buttonBox.AddSpacer(10)
        buttonBox.Add(buttonPlay, 0)
        buttonBox.AddSpacer(10)
        buttonBox.Add(buttonPause, 0)
        buttonBox.AddSpacer(10)
        buttonBox.Add(buttonNext, 0)
        buttonBox.AddSpacer(10)
        buttonBox.Add(self.rewindText, 0)

        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonPlay)
        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonPause)
        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonPrev)
        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonNext)

    def addResetAndSaveButtons(self, optionBox):
        _parent = self.scrolledPanel
        buttonApply = wx.Button(_parent, 2, 'Reset')
#        buttonSave = wx.Button(_parent, 3, 'SaveDB')

        dbButtonBox = wx.BoxSizer(wx.HORIZONTAL)
        dbButtonBox.Add(buttonApply, 0)
#        dbButtonBox.AddSpacer(10)
#        dbButtonBox.Add(buttonSave, 0)

        optionBox.AddSpacer(20)
        optionBox.Add(dbButtonBox, 0, wx.ALIGN_CENTER)

        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonApply)
#        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonSave)


    def addRenderMethodRadioBoxes(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20
        renderMethodBox = wx.RadioBox(_parent, wx.ID_ANY, majorDimension=3, size=(width, -1),\
            label="Choose a render method:", style=wx.RA_SPECIFY_ROWS, choices=['None', 'Line', 'Mesh'])
        renderMethodBox.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)
        renderMethodBox.SetSelection(2)
        self.canvas.setOptions('meshDrawing', True)
        optionBox.AddSpacer(20)
        optionBox.Add(renderMethodBox, 0, wx.ALIGN_CENTER)

    def addFocusCharacterCheckBox(self, optionBox):
        _parent = self.scrolledPanel
        checkBoxFocusCharacter = wx.CheckBox(_parent, id=1, label='Camera focus on character/object.')
        checkBoxRecordScreenshots = wx.CheckBox(_parent, id=2, label='Record screenshots on timer.')

        self.Bind(wx.EVT_CHECKBOX, self.OnFocusCheckBox, checkBoxFocusCharacter)
        self.Bind(wx.EVT_CHECKBOX, self.OnFocusCheckBox, checkBoxRecordScreenshots)

        optionBox.AddSpacer(20)
        optionBox.Add(checkBoxFocusCharacter, 0)
        optionBox.Add(checkBoxRecordScreenshots, 0)


    def addVelocitySlider(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20
        optionBox.AddSpacer(20)
        optionBox.Add(wx.StaticText(_parent, wx.ID_ANY, label="Speed Limit:", size=(width, -1)))
        velocitySlider = wx.Slider(_parent, value=30, minValue=0, maxValue=100,\
            style=wx.SL_HORIZONTAL|wx.SL_LABELS, name="velocity slider")
        self.Bind(wx.EVT_SCROLL, self.OnVelocitySliderChanged, velocitySlider)
        optionBox.Add(velocitySlider, 0, wx.EXPAND|wx.ALL)
        optionBox.Add(wx.StaticText(_parent, label="0m/s\t\t\t\t\t\t\t10m/s", id=wx.ID_ANY))
        optionBox.AddSpacer(20)

        self.canvas.SetFocus()

    def addQueryOptionCheckBoxes(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20

        box = wx.StaticBoxSizer(wx.VERTICAL, _parent)

        box.Add(wx.StaticText(_parent, id=wx.ID_ANY, label="Compose motion matching\nquery options:", size=(width, -1)))
        box.AddSpacer(10)

        buttons = []
        buttons.append(("scalePoints", wx.CheckBox(_parent, id=10, label="Resample Input in 150Hz")))
        buttons.append(("smoothInput", wx.CheckBox(_parent, id=8, label="Smooth User-Drawn Path")))
        buttons.append(("limitedUpdate", wx.CheckBox(_parent, id=3, label="Compute Desired Position\nof Current Character")))
        buttons.append(("startFromNearest", wx.CheckBox(_parent, id=4, label="Use Nearest Path Point\nas Desired Position")))
        buttons.append(("amplifyCurve", wx.CheckBox(_parent, id=7, label="Modify Future Position\nfor Sharp Corners")))
        buttons.append(("extendedSearch",wx.CheckBox(_parent, id=11, label="Extended Motion Matching")))
        buttons.append(("forceQuery", wx.CheckBox(_parent, id=5, label="Force query after a strong curve")))
        buttons.append((None, wx.CheckBox(_parent, id=6, label="Use DirectionNet")))
        buttons.append(("localControl", wx.CheckBox(_parent, id=9, label="Local Control Mode")))

        for name, b in buttons:
            if b.GetId() not in [4,6,11]:
                b.SetValue(True)
                if name is not None:
                    self.canvas.setOptions(name, True)
            self.Bind(wx.EVT_CHECKBOX, self.OnQueryCheckBox, b)
            box.Add(b, 0)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)

    def addLogRadioBoxes(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20
        box = wx.RadioBox(_parent, wx.ID_ANY, majorDimension=2,\
            label="Pose rendering method:", size=(width, -1),\
            style=wx.RA_SPECIFY_ROWS, choices=['Current Only', 'With Log'])
        box.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)

        box.SetSelection(0)
        self.canvas.setOptions('logDrawing', False)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)


    def addRenderOptionCheckBoxes(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20
        box = wx.StaticBoxSizer(wx.VERTICAL, _parent)

        box.Add(wx.StaticText(_parent, id=wx.ID_ANY, label="What to render:"))
        box.AddSpacer(10)

        buttons = []
        buttons.append(wx.CheckBox(_parent, id=12, label="Desired Position")) 
        buttons.append(wx.CheckBox(_parent, id=13, label="User Given Positions"))
        buttons.append(wx.CheckBox(_parent, id=14, label="Max Velocity Limited Position")) 
        buttons.append(wx.CheckBox(_parent, id=15, label="Query Arrows")) 
        buttons.append(wx.CheckBox(_parent, id=16, label="Query Points")) 
        buttons.append(wx.CheckBox(_parent, id=17, label="Corner Point")) 
        buttons.append(wx.CheckBox(_parent, id=18, label="Root Trajectory Log")) 

        for i, b in enumerate(buttons):
            if i == 3:
                b.SetValue(True)
                self.canvas.setOptions("r_arrows", True)
            if i == 4:
                b.SetValue(True)
                self.canvas.setOptions("r_queryPoints", True)
            self.Bind(wx.EVT_CHECKBOX, self.OnRenderCheckBox, b)
            box.Add(b, 0)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)


    def addCharacterBoard(self, optionBox):
        width = self.scrolledPanel.GetSize().GetWidth() - 20
        box = wx.StaticBoxSizer(wx.VERTICAL, self.scrolledPanel)

        box.Add(wx.StaticText(self.scrolledPanel, id=wx.ID_ANY, label="Characters in action(SHIFT KEY):", size=(width,-1)))
        box.AddSpacer(10)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)

        self.characterBox = box

    def addExtendedSearchParameters(self, optionBox):
        _parent = self.scrolledPanel
        width = _parent.GetSize().GetWidth() - 20
        box = wx.StaticBoxSizer(wx.VERTICAL, _parent)
        
        self.K = wx.TextCtrl(_parent, wx.ID_ANY, value="2")
        self.L = wx.TextCtrl(_parent, wx.ID_ANY, value="3")
        
        box.Add(wx.StaticText(_parent, wx.ID_ANY, label="K and L\nfor Extended\nMotion Matching\n", size=(width,-1)))
        box.Add(wx.StaticText(_parent, wx.ID_ANY, label="Enter K:"))
        box.Add(self.K)
        box.Add(wx.StaticText(_parent, wx.ID_ANY, label="Enter L(1<=L<=3):"))
        box.Add(self.L)
        box.AddSpacer(20)

        button = wx.Button(_parent, 19, "Apply")
        box.Add(button)
        self.Bind(wx.EVT_BUTTON, self.OnButton, button)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)


#    def addFeatureWeightTextCtrl(self, optionBox):
#
#        box = wx.BoxSizer(wx.VERTICAL)
#
#        self.weights['foot pos'] = wx.TextCtrl(self, wx.ID_ANY, value='1.0')
#        self.weights['foot vel'] = wx.TextCtrl(self, wx.ID_ANY, value='1.0')
#        self.weights['hip vel'] = wx.TextCtrl(self, wx.ID_ANY, value='1.0')
#        self.weights['future traj pos'] = wx.TextCtrl(self, wx.ID_ANY, value='1.0')
#        self.weights['future traj ori'] = wx.TextCtrl(self, wx.ID_ANY, value='1.0')
#
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="===Feature Weights==="))
#        
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="Foot Position"))
#        box.Add(self.weights['foot pos'])
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="Foot Velocity"))
#        box.Add(self.weights['foot vel'])
#
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="Hip Velocity"))
#        box.Add(self.weights['hip vel'])
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="FutureTrajectory Position"))
#        box.Add(self.weights['future traj pos'])
#        box.Add(wx.StaticText(self, wx.ID_ANY, label="FutureTrajectory Orientation"))
#        box.Add(self.weights['future traj ori'])
#
#        box.AddSpacer(20)
#        button = wx.Button(self, 6, 'Apply Weights')
#        box.Add(button)
#        self.Bind(wx.EVT_BUTTON, self.OnButton, button)
#
#
#        optionBox.AddSpacer(20)
#        optionBox.Add(box, 0, wx.ALIGN_CENTER)
