import numpy as np
from data.Node import Node

def parseNode(it, stack):
    try:
        while True:
            word = next(it) 
            if word in ['ROOT', 'JOINT']:
                break
            elif word == 'End':
                next(it) # 'Site'
                next(it) # 'Offset'
                offset = []
                for _ in range(3):
                    offset.append(float(next(it)))
                stack.append(Node('End Site', offset, [], stack[-1])) # dummy node
            elif word == '}' and len(stack) > 1: # not a root
                stack.pop()

        name = next(it)
        offset = []
        channels = []

        assert(next(it) == 'OFFSET')
        offset.append(float(next(it)))
        offset.append(float(next(it)))
        offset.append(float(next(it)))

        assert(next(it) == 'CHANNELS')
        length = int(next(it))
        for i in range(length):
            channels.append(next(it))
        
        if word == 'ROOT':
            stack.append(Node(name, offset, channels, None, True))
        else: # JOINT
            stack.append(Node(name, offset, channels, stack[-1]))

        # RECURSION 
        parseNode(it, stack)

    except StopIteration:
        return stack[-1] 

    return stack[-1] # node that is just created

def parseMotion(text):

    frameTime = float(text[2].split()[2]) # from line "Frame Time: #####"
    frames = [np.array(line.split(), dtype=np.float32) for line in text[3:]]
    if len(frames[-1]) == 0:
        frames.pop()
    return frameTime, np.array(frames, dtype=np.float32)

def parseContact(text, length):
    contact = np.zeros(length, dtype=bool)
    intervals = text.splitlines()
    count = int(intervals.pop(0))
    for i in range(count):
        start, end = intervals[i].split()
        contact[int(start):int(end)] = 1
    return contact

def printCheckTree(node):
    print(node.name)
    for c in node.children:
        printCheckTree(c)


