from networks.preparing import *
from networks.BiGRUNetwork import BiGRUNetwork

from motion.MotionStream import MotionStream

import sys, getopt
import pickle

import numpy as np

def main(data, meta):

    network = BiGRUNetwork()

    if meta_output['window'] == 0:
        print('No window size!')
        print('Training terminated')
        return

    trainset = prepareDatasetByWindow(data, meta['window'])
    network.train(meta['name'], trainset, n_epoch=2000) #This will save weights automatically

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, "d:o:w:") # _: leftovers
    except getopt.GetoptError:
        print("trainNetwork.py -d <data file> -o <output name> -w <window size>")
        quit()

    meta_output = {'window': 120, 'name': 'default'}

    for opt, arg in opts:
        if opt == '-d':
            with open(arg, 'rb') as f:
                data = pickle.load(f)
                data = MotionStream(data)
        elif opt == '-o':
            meta_output['name'] = arg
        elif opt == '-w':
            meta_output['window'] = int(arg)

    main(data, meta_output)
