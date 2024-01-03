from util.matrixFormulas import *
import numpy as np

class Node:
    def __init__(self, name, offset, channels, parentNode=None, isRoot=False):
        self.name = name
        self.offset = np.array(offset)
        self.channels = channels
        self.children = [] # list of nodes
        self.parent = parentNode

        self.isRoot = isRoot
        self.rotationFromParent = self.offsetRotation()

        if parentNode:
            parentNode.addChild(self)

    def addChild(self, childNode):
        self.children.append(childNode)

    def offsetRotation(self):
        length = np.sqrt(sum(self.offset*self.offset))
        if length == 0:
            return None

        a = (0,0,1)
        b = tuple(self.offset / length)
        return rotationMatrixFrom2Vectors(a, b)
