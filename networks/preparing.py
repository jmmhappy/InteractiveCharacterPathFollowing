import random
import sys

import numpy as np 
from data.Motion import Motion

def prepareDatasetByWindow(motionstream, WINDOW):

    valid_m_idx = []
    for i, m in enumerate(motionstream.motions):
        if len(m.postures) >= WINDOW:
            valid_m_idx.append(i)
    print('Leaving %d motions from total %d motions.'%(len(valid_m_idx), len(motionstream.motions)))

    train = []
    for i, idx in enumerate(valid_m_idx):
        m = motionstream.motions[idx]

        for j in range(len(m.postures) - WINDOW):
            train.append(_projection(m.postures[j : j + WINDOW], int(WINDOW/2) - 1))

        _progressBar(i+1, len(valid_m_idx))

    print()
#    np.random.shuffle(train) # TODO haven't tried yet
    return train # (number of segments) x (window size) x 4

def _projection(postures, pivot=None):
    _extract = []
    m_length = len(postures) # same as above

    startIndex = np.random.randint(0, m_length) if pivot is None else pivot

    localFrame_inv = np.linalg.inv(postures[startIndex].getLocalFrame()) # local frame of the start point
    for p in postures:
        position = (localFrame_inv @ np.append(p.getPosition(), 1))[[0, 2]]
        direction = (localFrame_inv @ p.getOrientations()[0])[[0,2], 2]
        _extract.append(np.concatenate((position, direction)))

    return np.array(_extract)

def _progressBar(now, total):
    percentage = int(now / total * 100)

    sys.stdout.write('\r') # rewrite startpoint
    sys.stdout.write('[%-20s], %3d%%' % ('='*(percentage//5), percentage))
    sys.stdout.flush()

   
