import numpy as np
import tensorflow as tf
import os
import sys
import json
import random

sys.path.append("./src/")
from utils.tf_dl_utils import TfRNNCell, TfTraining
from proto import io
from proto import CoreNLP_pb2
from proto import dataset_pb2
from proto import training_dataset_pb2
from learning_baseline.agent import Agent
from learning_baseline.agent import QaData


class RnnAgent(Agent):
    '''
    Recurrent neural networks using lstm/rnn cells.

    The config structure should contain the following info:
    nVocab
    nEmbedDim
    nHidden
    cellType
    keepProb
    '''
    def __init__(self, nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev):
        '''
        loss type has to be either "max-margin" or "nce" 
        '''
        Agent.__init__(self, floatType, idType, lossType)
        self.nEmbedDim = nEmbedDim
        self.nHidden = nHidden
        self.nLayer = nLayer
        self.cellType = cellType
        self.keepProb = keepProb
        self.floatType = floatType
        self.idType = idType
        self.lossType = lossType
        self.initStdDev = initStdDev
         

    def LoadInitWordVec(self, fileName):
        '''
        Note the first two embedding are preserved for <pad> and <unk>
        The <pad> should be kept zero. nVocab did not count <pad>
        '''
        self.initEmbedding = np.load(fileName) / 3
        self.initEmbedding[0, :] = 0.0  
        if self.floatType == tf.float32:
            self.initEmbedding = self.initEmbedding.astype(np.float32)
        else:
            self.initEmbedding = self.initEmbedding.astype(np.float64)      
        self.nVocab = self.initEmbedding.shape[0] - 1


    def ConstructGraph(self, batchSize):
        pass


    def GetNextTrainBatch(self):
        pass


