import os
import pickle
import numpy as np

from data.Motion import Motion
from motion.MotionMatcher import MotionMatcher
import pickle


def generate(dataPath, name, matcher=True):
    motions, totalPosturesCount = importBVH(dataPath)
    motionMatcher = MotionMatcher(motions, FUTURE_TERM=10, leftFootName='LeftFoot', rightFootName='RightFoot') \
        if matcher else None

    with open(name, 'wb') as f:
        pickle.dump({
        'motions':motions, 
        'total':totalPosturesCount, 
        'motionMatcher':motionMatcher
        }, f)
        print('Your file is saved.')


def load(dataPath):
    with open(dataPath, 'rb') as f:
        dataset = pickle.load(f)
        print('Load complete')
    return dataset

def importBVH(path, progressBar=None, targetFPS=30):

    if os.path.isfile(path):
        m = Motion(path, targetFPS)
        return [m], len(m.postures)

    motions = []
    count = 0

    fileList = bringBVHFiles(path) #path is directory
    length = len(fileList)
    for i in range(0, length):
        f = fileList[i]

        if progressBar:
            progressBar.Update(int((i+1)/length * 100))

        m = Motion(f, targetFPS)

        motions.append(m)
        count += len(m.postures)
    return motions, count 


def exportBVH(newFileName, bvh):
    if not bvh:
        return False # Nothing to export.
    
    newFile = open(newFileName + '.bvh', 'x')

    with open(bvh.name, 'r').read() as text:# Will NOT WORK!
        hierarchy = text[:text.find('MOTION')]

        newFile.write(hierarchy)
        newFile.write('MOTION\nFrames: ' + str(bvh.frameCount) +\
            '\nFrame Time: ' + str(bvh.frameTime) + '\n')

        for frame in bvh.frames:
            # write a line
            for data in frame:
                newFile.write(str(data))
            newFile.write('\n')
    newFile.close()

    return True
 
def bringBVHFiles(rootDir, result=[]):
    for f in os.listdir(rootDir):
        filepath = os.path.join(rootDir, f)

        if os.path.splitext(filepath)[-1] == '.bvh':
            result.append(filepath)
        elif os.path.isdir(filepath):
            bringBVHFiles(filepath, result)

    return result

