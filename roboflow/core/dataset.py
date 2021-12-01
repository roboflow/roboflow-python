import os
from roboflow.config import *
import sys

class Dataset():
    def __init__(self, name, version, model_format, location):
        self.name = name
        self.name = version
        self.model_format = model_format
        self.location = location
        #resolution
        #num_train, num_val, num_test
        #class_names

    #def describe

    #def visualize

    #def sort_by_sim

    #def sample

    #def query