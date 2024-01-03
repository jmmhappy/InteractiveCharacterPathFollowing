import wx
import wx.glcanvas as glcanvas

try:
    from OpenGL.GL import *
    haveOpenGL = True
except ImportError:
    haveOpenGL = False

from render.CanvasPanel import CanvasPanel
from render.MotionRenderer import MotionRenderer
from render.ObstacleRenderer import ObstacleRenderer
from render.SkelRenderer import SkelRenderer
from render.ParticleRenderer import ParticleRenderer

import util.bvhMenu as bvhMenu

import sys, os, argparse
from datetime import datetime

import numpy as np

from networks.BiGRUNetwork import BiGRUNetwork
from motion.MotionStream import MotionStream

class RunApp(wx.App):
    def __init__(self, args):
        wx.App.__init__(self)
        
        #frame = wx.Frame(None, -1, "Run Application", size=(1050,1000))
        frame = wx.Frame(None, -1, "Run Application", size=(2060,2000))
        #frame = wx.Frame(None, -1, "Run Application", size=(3550,2200))
        frame.CreateStatusBar()

        # --- Menu Bar ---
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()

        exportBvhItem = fileMenu.Append(wx.ID_ANY, 'Export BVH file')
        self.Bind(wx.EVT_MENU, self.OnExportBvh, exportBvhItem)

        saveTrajectoryItem = fileMenu.Append(wx.ID_ANY, 'Save current mouse trajectory')
        self.Bind(wx.EVT_MENU, self.OnSaveTrajectory, saveTrajectoryItem)
        loadTrajectoryItem = fileMenu.Append(wx.ID_ANY, 'Load a mouse trajectory')
        self.Bind(wx.EVT_MENU, self.OnLoadTrajectory, loadTrajectoryItem)

        exitItem = fileMenu.Append(wx.ID_EXIT, 'Quit')
        self.Bind(wx.EVT_MENU, self.OnExitApp, exitItem)

        menubar.Append(fileMenu, '&File')

        graphMenu = wx.Menu()
        logGraphItem = graphMenu.Append(wx.ID_ANY, 'Show frame number log graph')
        self.Bind(wx.EVT_MENU, self.OnLogGraph, logGraphItem)
        menubar.Append(graphMenu, '&Graph')

        # --- Show frame ---
        frame.SetMenuBar(menubar)
        frame.Centre()
        frame.Show(True)
        frame.Bind(wx.EVT_CLOSE, self.OnCloseFrame)

        self.window = CanvasPanel(frame)
        self.frame = frame

        dataPath, weightPath = args.data, args.weights
        characters = int(args.characters) if args.characters else 1
        network = None

        if weightPath and os.path.isfile(weightPath):

            if args.agent == 'character':
                from rl.agent_char import Agent
                network = Agent(weightPath)
                print("Loaded character agent")
            elif args.agent == 'map':
                from rl.agent import Agent
                network = Agent(weightPath)

                config = network.trainer.get_config()['env_config']
                self.window.appendRenderer('obstacles', ObstacleRenderer(config['map'], 1/30))
                print("Loaded map agent")

            elif weightPath[-3:] == 'bin':
                network = BiGRUNetwork()
                network.load_weights(weightPath)
                print('Loaded pace network')

        # TEST!!
        from render.ObstacleRendererForGlobalTracking import ObstacleRenderer
        self.window.appendRenderer('obstacles', ObstacleRenderer(1/30))

        data = None
        if dataPath[-3:] == 'bin':
            data = bvhMenu.load(dataPath)
        else:
            _m, _c = bvhMenu.importBVH(dataPath)
            data = {
            'motions':_m, 'total':_c,
            'motionMatcher':None,
            }

        def composeCharacter(data, network):
            mstream = MotionStream(data)
            mstream.setNetwork(False, network)
            return mstream

        namePool = ['A', 'B', 'C']
        for i in range(characters):
            self.window.appendRenderer(namePool[i], MotionRenderer(composeCharacter(data, network)))



    def OnExportBvh(self, evt):
        # User input -> New name for this bvh.
        dig = wx.TextEntryDialog(self.frame, 'Enter a New Name for this BVH.')
        if dig.ShowModal() == wx.ID_OK and len(dig.GetValue()) > 0:
            name = dig.GetValue()
        else:
            wx.MessageDialog(self.frame, 'Failed to Export!').ShowModal()
            return False
        dig.Destroy()

        bvh = self.window.canvas.bvh

        # Create a new file
        if not bvhMenu.exportBVH(name, bvh):
            # Create popup
            wx.MessageDialog(self.frame, "Nothing to Export!").ShowModal()

        wx.MessageDialog(self.frame, "Done!").ShowModal()
        return True
   
    def OnSaveTrajectory(self, evt):
        name = '/mouse ' + str(datetime.now())
        np.save(os.getcwd() + name, self.window.getMouseTrajectory())

        wx.MessageDialog(self.frame, "Saved!").ShowModal()

    def OnLoadTrajectory(self, evt):
        with wx.FileDialog(self.frame, 'Set trajectory to ...') as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            self.window.setMouseTrajectory(np.load(fileDialog.GetPath()))
        wx.MessageDialog(self.frame, "Loaded!").ShowModal()

    def OnExitApp(self, evt):
        self.frame.Close(True)

    def OnCloseFrame(self, evt):
        if hasattr(self, 'window'):
            quit()

    def OnLogGraph(self, evt):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10,10))
        frames = self.window.getSliderLog()
        plt.margins(x=0.0)
        plt.ylim(0,120)
        plt.hist(frames, bins=20)
        plt.show()
         


#----------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=False, help='Database')
    parser.add_argument('--agent', required=False, help='Character or Map')
    parser.add_argument('--weights', required=False, help="Network weights")
    parser.add_argument('--characters', required=False, help="How many characters to control")
    args = parser.parse_args()

    if not haveOpenGL:
        wx.MessageBox('This sample requires the PyOpenGL package.', 'Sorry')
    else:
        app = RunApp(args)
        app.MainLoop()

if __name__ == '__main__':
    main(sys.argv)

