import matplotlib.pyplot as plt

from util.bvhMenu import *
import numpy as np

import sys

def main(path):
    form = path.split('.')[-1]
    if form == 'bin':
        data = load(path)
        motions = data['motions']
        total = data['total']
    else:
        motions, total = importBVH(path)

    print('Of total %d frames.'%total)
    positions = []
    for m in motions:
        for p in m.getPostures():
            positions.append(p.getPosition()*[1,0,1])
    positions.append(positions[-1])
    
    positions = np.asarray(positions)
    velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    velocities *= 30 # m/s in FPS 30

    print(max(velocities))
    plt.margins(x=0.0000)
    plt.hist(velocities, range=(0,4), align='left',bins=100)
    plt.show()

if __name__ == '__main__':
    _, dataPath = sys.argv
    main(dataPath)
