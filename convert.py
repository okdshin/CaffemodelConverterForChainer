import pickle
import os
import argparse
from chainer.links.caffe import CaffeFunction

def main():
    parser = argparse.ArgumentParser(description='Master')
    parser.add_argument('-modelpath', type=str, help='path of target caffe model')
    args = parser.parse_args()
    vgg = CaffeFunction(args.modelpath)
    pickle.dump(vgg, open(os.path.splitext(os.path.basename(args.modelpath))[0]+".pkl", 'wb'))

if __name__ == "__main__":
    main()

