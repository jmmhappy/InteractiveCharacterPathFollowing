from util.bvhMenu import importBVH
import numpy as np
import sys
import os

def extractFilePaths(directory):
    result = []
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.splitext(path)[-1] == '.bvh':
            result.append(path)
    return result

def openfile(filepath):
    if filepath[-3:] == 'bvh':
        motions, total = importBVH(filepath, targetFPS=120)
        motion = motions[0]
        leftFoot = []
        rightFoot = []
        for p in motion.getPostures():
            positions = dict(p.forwardKinematics(motion.getSkeletonRoot()))
            localFrame = p.getLocalFrame()
            leftFoot.append((localFrame @ np.append(positions['LeftFoot'], 1))[:3])
            rightFoot.append((localFrame @ np.append(positions['RightFoot'], 1))[:3])

        np.savez(filepath, leftFoot=np.array(leftFoot, dtype=np.float32), rightFoot=np.array(rightFoot, dtype=np.float32)) 

    elif filepath[-3:] == 'npz':
        npzfile= np.load(filepath)
        leftFoot = npzfile['leftFoot']
        rightFoot = npzfile['rightFoot']

    else:
        print('Cannot find file %s'%filepath)
        quit()
    return leftFoot, rightFoot


def getVelocity(rawData):
    FRAMETIME = 1/30
    def velocity(d):
        tmp = np.array(d + [d[-1]], dtype=np.float32)
        return (tmp[1:] - tmp[:-1]) * (1/FRAMETIME)

    return np.linalg.norm(velocity(rawData), axis=1)

def getContact(position, velocity):

    MAX_HEIGHT = .70
    MAX_VELOCITY = 0.3

    def contact(foot, vel):
        contacts = []
        i = 0
        prev = None
        for p, v in zip(foot, vel):
            height = p[1] # Y UP
            if v < MAX_VELOCITY and height < MAX_HEIGHT:
                if prev != True: # 0->1
                    contacts.append(i)
                prev = True
            else:
                if prev == True: # 1->0
                    contacts.append(i)
                prev = False

            i += 1


        if len(contacts) % 2 == 1: # not even:
            contacts.append(len(foot))

        return contacts

    return contact(position, velocity)


def saveContactInFile(filepath, c, end):
    if filepath[-3:] == 'npz':
        filepath = filepath[:-4]

    def parseText(c):

        text = '%d\n'%(int(len(c)/2))

        for i in range(0, len(c), 2):
            start, end = c[i:i+2]
            text += '%d %d\n'%(start, end)

        return text
    
    name = filepath + end 
    with open(name, 'w') as f:
        f.write(parseText(c))


def main(filepath):
    if os.path.isdir(filepath):
        filepaths = extractFilePaths(filepath)
    else:
        filepaths = [filepath]


    for filepath in filepaths:

        leftFoot, rightFoot = openfile(filepath)

        leftFootVel = getVelocity(leftFoot)
        rightFootVel = getVelocity(rightFoot)
        leftContacts = getContact(leftFoot, leftFootVel)
        rightContacts = getContact(rightFoot, rightFootVel) 

        saveContactInFile(filepath, leftContacts, '.conL.txt')
        saveContactInFile(filepath, rightContacts, '.conR.txt')


if __name__ == '__main__':
    filepath =sys.argv[1]
    main(filepath)
