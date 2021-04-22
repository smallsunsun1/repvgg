import argparse
import yaml
import numpy as np 
import tensorflow as tf 

parser = argparse.ArgumentParser(description="tensorflow imagenet training")
parser.add_argument("--data", type=str, help="path to training data", default="data")
parser.add_argument("--arch", type=str, help="model architecture", default="RepVGG-A0")
parser.add_argument("--workers", type=int, help="number of threads", default=10)
parser.add_argument("--epochs", type=int, help="number of epochs to run", default=120)
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')


if __name__ == "__main__":
    pass