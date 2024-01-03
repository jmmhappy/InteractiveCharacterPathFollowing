import dartpy as dart

if __name__== "__main__":
    world = dart.utils.SkelParser.readWorld("/home/jmm/Projects/motionmatching/skel/4legger.skel")
    world.setGravity([0, -9.81, 0])

    node = dart.gui.osg.RealTimeWorldNode(world)

    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0.8, 0.0, 0.8], [0, -0.25, 0], [0, 0.5, 0])
    viewer.run()
