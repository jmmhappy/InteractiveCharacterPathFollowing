import wx
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
        canvasSize = topBox.Add(canvas, 0, wx.EXPAND|wx.ALL).GetSize()

        # Options 
        optionBox = wx.BoxSizer(wx.VERTICAL)

        self.addResetAndSaveButtons(optionBox)
        self.addFocusCharacterCheckBox(optionBox)
        self.addRenderMethodRadioBoxes(optionBox)
        self.addLogRadioBoxes(optionBox)
        self.addQueryOptionCheckBoxes(optionBox)
        self.addVelocitySlider(optionBox)
        self.addRenderOptionCheckBoxes(optionBox)
        self.addCharacterBoard(optionBox)

        optionBox.AddSpacer(20)
        taskLabel = wx.StaticText(self, label='')
        optionBox.Add(taskLabel, 0, wx.ALIGN_CENTER)

        topBox.Add(optionBox, 0)
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

    def getMouseTrajectory(self):
        return self.canvas.getMouseTrajectory()
    def setMouseTrajectory(self, t):
        self.canvas.setMouseTrajectory(t)

    def getSliderLog(self):
        return self.canvas.getFrameNumberLog()

    def OnRadioBox(self, evt):
        rb = evt.GetEventObject()
        selection = rb.GetStringSelection()

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
        if identifier == 11:
            name = "desiredPoint"
        elif identifier == 12: 
            name = "nonlimitedQueryPoints"
        elif identifier == 13: 
            name = "maxLimitPoint"
        elif identifier == 14: 
            name = "arrows"
        elif identifier == 15: 
            name = "queryPoints"
        elif identifier == 16: 
            name = "cornerPoint"
        elif identifier == 17:
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

        text = wx.StaticText(self, id=wx.ID_ANY, label=name)
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
        buttonApply = wx.Button(self, 2, 'Reset')
#        buttonSave = wx.Button(self, 3, 'SaveDB')

        dbButtonBox = wx.BoxSizer(wx.HORIZONTAL)
        dbButtonBox.Add(buttonApply, 0)
#        dbButtonBox.AddSpacer(10)
#        dbButtonBox.Add(buttonSave, 0)

        optionBox.AddSpacer(20)
        optionBox.Add(dbButtonBox, 0, wx.ALIGN_CENTER)

        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonApply)
#        self.Bind(wx.EVT_BUTTON, self.OnButton, buttonSave)


    def addRenderMethodRadioBoxes(self, optionBox):
        renderMethodBox = wx.RadioBox(self, wx.ID_ANY, majorDimension=2, \
            label="Choose a render method:", style=wx.RA_SPECIFY_ROWS, choices=['Line', 'Mesh'])
        renderMethodBox.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)
        renderMethodBox.SetSelection(1)
        self.canvas.setOptions('meshDrawing', True)
        optionBox.AddSpacer(20)
        optionBox.Add(renderMethodBox, 0, wx.ALIGN_CENTER)

    def addFocusCharacterCheckBox(self, optionBox):
        checkBoxFocusCharacter = wx.CheckBox(self, id=1, label='Camera focus on character/object.')
        checkBoxRecordScreenshots = wx.CheckBox(self, id=2, label='Record screenshots on timer.')

        self.Bind(wx.EVT_CHECKBOX, self.OnFocusCheckBox, checkBoxFocusCharacter)
        self.Bind(wx.EVT_CHECKBOX, self.OnFocusCheckBox, checkBoxRecordScreenshots)

        optionBox.AddSpacer(20)
        optionBox.Add(checkBoxFocusCharacter, 0)
        optionBox.Add(checkBoxRecordScreenshots, 0)


    def addVelocitySlider(self, optionBox):
        optionBox.AddSpacer(20)
        optionBox.Add(wx.StaticText(self, wx.ID_ANY, label="Speed Limit:"))
        velocitySlider = wx.Slider(self, value=30, minValue=0, maxValue=100,\
            style=wx.SL_HORIZONTAL|wx.SL_LABELS, name="velocity slider")
        self.Bind(wx.EVT_SCROLL, self.OnVelocitySliderChanged, velocitySlider)
        optionBox.Add(velocitySlider, 0, wx.EXPAND|wx.ALL)
        optionBox.Add(wx.StaticText(self, label="0m/s\t\t\t\t\t\t\t10m/s", id=wx.ID_ANY))
        optionBox.AddSpacer(20)

        self.canvas.SetFocus()

    def addQueryOptionCheckBoxes(self, optionBox):

        box = wx.StaticBoxSizer(wx.VERTICAL, self)

        box.Add(wx.StaticText(self, id=wx.ID_ANY, label="Compose motion matching\nquery options:"))
        box.AddSpacer(10)

        buttons = []
        buttons.append(("scalePoints", wx.CheckBox(self, id=10, label="Resample Input in 150Hz")))
        buttons.append(("smoothInput", wx.CheckBox(self, id=8, label="Smooth User-Drawn Path")))
        buttons.append(("limitedUpdate", wx.CheckBox(self, id=3, label="Compute Desired Position\nof Current Character")))
        buttons.append(("startFromNearest", wx.CheckBox(self, id=4, label="Use Nearest Path Point\nas Desired Position")))
        buttons.append(("amplifyCurve", wx.CheckBox(self, id=7, label="Modify Future Position\nfor Sharp Corners")))
        buttons.append(("forceQuery", wx.CheckBox(self, id=5, label="Force query after a strong curve")))
        buttons.append((None, wx.CheckBox(self, id=6, label="Use DirectionNet")))
        buttons.append(("localControl", wx.CheckBox(self, id=9, label="Local Control Mode")))

        for name, b in buttons:
            if b.GetId() not in [4,6]:
                b.SetValue(True)
                if name is not None:
                    self.canvas.setOptions(name, True)
            self.Bind(wx.EVT_CHECKBOX, self.OnQueryCheckBox, b)
            box.Add(b, 0)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)

    def addLogRadioBoxes(self, optionBox):
        box = wx.RadioBox(self, wx.ID_ANY, majorDimension=2,\
            label="Pose rendering method:",\
            style=wx.RA_SPECIFY_ROWS, choices=['Current Only', 'With Log'])
        box.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)

        box.SetSelection(0)
        self.canvas.setOptions('logDrawing', False)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)


    def addRenderOptionCheckBoxes(self, optionBox):
        box = wx.StaticBoxSizer(wx.VERTICAL, self)

        box.Add(wx.StaticText(self, id=wx.ID_ANY, label="What to render:"))
        box.AddSpacer(10)

        buttons = []
        buttons.append(wx.CheckBox(self, id=11, label="Desired Position")) 
        buttons.append(wx.CheckBox(self, id=12, label="User Given Positions"))
        buttons.append(wx.CheckBox(self, id=13, label="Max Velocity Limited Position")) 
        buttons.append(wx.CheckBox(self, id=14, label="Query Arrows")) 
        buttons.append(wx.CheckBox(self, id=15, label="Query Points")) 
        buttons.append(wx.CheckBox(self, id=16, label="Corner Point")) 
        buttons.append(wx.CheckBox(self, id=17, label="Root Trajectory Log")) 

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
        box = wx.StaticBoxSizer(wx.VERTICAL, self)

        box.Add(wx.StaticText(self, id=wx.ID_ANY, label="Characters in action(SHIFT KEY):"))
        box.AddSpacer(10)

        optionBox.AddSpacer(20)
        optionBox.Add(box, 0, wx.ALIGN_CENTER)

        self.characterBox = box


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
