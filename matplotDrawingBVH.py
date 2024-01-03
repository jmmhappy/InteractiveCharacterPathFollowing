import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from util.bvhMenu import *
import numpy as np

from motion.MotionStream import MotionStream

db = importBVH('.')
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

line, = ax.plot([],[],[])

def update(frame):
    _, pose = db.poseUpdate(frame)
    points = pose.getAllNodePositions(db.getSkeleton().getRoot())
    line.set_data([points[0], points[2]])
    line.set_3d_properties(points[1])
    return line

ani = FuncAnimation(fig, update, frames=db.getTotalFrameCount(), interval=30)
plt.show()
