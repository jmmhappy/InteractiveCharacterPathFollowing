import numpy as np
import util.bvhParser as parser
from util.matrixFormulas import *
from data.Node import Node

class Skeleton():
    def __init__(self, skeletonInText):
        self.root = parser.parseNode(iter(skeletonInText), stack=[])

        #CMU
        self.SCALE = (1.0/0.45)*2.54/100.0

        #LaFAN1
        #self.SCALE =0.01
        #self.root.offset = np.zeros(3)

        self.applyScale()

    def getRoot(self):
        return self.root
    def getScale(self):
        return self.SCALE

    '''
    def setScale(self, scale):
        self.SCALE = scale
        applyScale()
        '''

    def applyScale(self, node=None):
        node = self.root if not node else node

        node.offset *= self.SCALE
        for child in node.children:
            self.applyScale(child)
