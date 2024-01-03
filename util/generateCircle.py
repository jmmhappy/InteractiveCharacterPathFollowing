import numpy as np

def accelerate(param,length):
    assert(param > 1)
    result=[]
    i=1
    idx=0
    while idx < length:
        result.append(idx)
        idx += i
        i *= param
    return np.array(result, int)

def divide(length):
    result = []
    one = np.arange(0,length/3, 1)
    two = np.arange(length/3, length/3*2, 2)
    three = np.arange(length/3*2 , length, 3)
    result = np.hstack((one, two, three))
    return np.array(result, int)

def makecircle(theta):
    x=np.array([np.cos(t) for t in theta])
    z=np.array([np.sin(t) for t in theta])
    y=np.zeros(x.shape)
    return np.vstack((x,y,z)).T

def accelCircle(alpha):
    time = np.arange(0,1000) * 0.005
    theta = .5*time*time*alpha
#    if alpha == 1: theta[:356] 2pi, theta[:435] 3pi
    return theta


from bisect import bisect_left
def main():
    theta = accelCircle(8)

    index = bisect_left(theta, np.pi*2)
    twopi = theta[:index] # 6.2424
    index = bisect_left(theta, np.pi*3)
    threepi = theta[:index] # 9.375

    c = makecircle(threepi)
    c *= 5 #radius
    
    print((np.linalg.norm(c[1:]-c[:-1], axis=1)*30))

    np.save("accelCircle_4", c)

