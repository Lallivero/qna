import numpy as np
import tensorflow as tf
import os
import sys
import json

sys.path.append("./src/")
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.data_processor import DataProcessor
from utils.squad_utils import LoadJsonData, ParseJsonData
from utils.loss_func import NCELoss, MaxMarginLoss
from utils.tf_dl_utils import TfRNNCell, TfTraining


class NaiveRnnAgent(object):
    '''
    Recurrent neural networks using lstm cells and 
    noise-contrastive-estimation loss layer. The negative
    samples may come.

    The config structure should contain the following info:
    nVocab
    nEmbedDim
    nHidden
    cellType
    keepProb
    '''
    def __init__(self, nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType):
        '''
        loss type has to be either "max-margin" or "nce" 
        '''
        self.nVocab = None
        self.nEmbedDim = nEmbedDim
        self.nHidden = nHidden
        self.nLayer = nLayer
        self.cellType = cellType
        self.keepProb = keepProb
        self.floatType = floatType
        self.idType = idType
        self.lossType = lossType
        self.qMaxLen = None
        self.aMaxLen = None
        self.trainVar = None
        # a list serving for debuging and run-time assertion
        self.debug = []


    def LoadDataset(self, datasetPath):
        with open(os.path.join(datasetPath), "r") as fp:
            self.data = json.load(fp)
        # TODO get max sequence length for q and a
        self.qMaxLen = 3
        # self.qMaxLen = None
        self.aMaxLen = 3


    def LoadVocab(self, vocabPath):
        with open(os.path.join(vocabPath, "word2id.json") ) as fp:
            self.wordToId = json.load(fp)
        with open(os.path.join(vocabPath, "id2word.json") ) as fp:
            self.idToWord = json.load(fp)
        self.nVocab = len(self.idToWord.keys() )


    def ConstructGraph(self, batchSize):
        '''
        the input comes in the format [maxSeqLen x batchSize]
        @return assertions: the dynamic assertions in the computational
        graph. Evaluate assertions as a list of tensors when session.run.
        It will trigger error messages indicating the violation of the assert.
        @return naScore, paScore: use paScores to get inference score. They
        are also for debuging the computational graph.
        '''
        floatType = self.floatType
        idType = self.idType
        self.batchSize = batchSize
        self.qRnnInput = \
            tf.placeholder(idType, [self.qMaxLen, batchSize] )
        self.qSeqLen = tf.placeholder(idType, [batchSize] ) 
        self.paRnnInput = \
            tf.placeholder(idType, [self.aMaxLen, batchSize] )
        self.paSeqLen = tf.placeholder(idType, [batchSize] )
        self.naRnnInput = \
            tf.placeholder(idType, [self.aMaxLen, None] )
        self.naSeqLen = tf.placeholder(idType, [None] )
        self.naGroupPos = tf.placeholder(idType, [batchSize + 1] )
        
        # get embedding for question and answer
        self.embedding = tf.get_variable("embedding", 
            shape=[self.nVocab, self.nEmbedDim], dtype=floatType)
        qRnnInput = tf.nn.embedding_lookup(self.embedding, self.qRnnInput)
        paRnnInput = tf.nn.embedding_lookup(self.embedding, self.paRnnInput)
        naRnnInput = tf.nn.embedding_lookup(self.embedding, self.naRnnInput)
        # aRnnInput goes in the format [maxSeqLen x nSample X nEmbeddingDim]
        aRnnInput = tf.concat(1, [paRnnInput, naRnnInput] )
        aSeqLen = tf.concat(0, [self.paSeqLen, self.naSeqLen] )
        qSeqLen = self.qSeqLen

        # # prepare forward and backward basic cell for both question and answer
        cellGenerator = TfRNNCell(self.nHidden,
            self.nLayer, self.keepProb, self.cellType)
        
        # get encoding for questions
        with tf.variable_scope("QRnn"):
            qCellFw = cellGenerator.GetCell()
            qCellBw = cellGenerator.GetCell()
            qInitStateFw = qCellFw.zero_state(batchSize, floatType)
            qInitStateBw = qCellFw.zero_state(batchSize, floatType)
            # note the input should be a list where each ele is a tensor for a step
            qRnnRes = tf.nn.bidirectional_rnn(cell_fw=qCellFw, cell_bw=qCellBw, 
                inputs=[qRnnInput[i, :, :] for i in xrange(self.qMaxLen) ],
                initial_state_fw=qInitStateFw, initial_state_bw=qInitStateBw, 
                dtype=floatType, sequence_length=qSeqLen)
            # note qRnnRes = (output, final_state_fw, final_state_bw)
            qCode = tf.concat(1, [qRnnRes[1], qRnnRes[2] ] )

        # # get encoding for answers (process postive and negative answer
        # # in a single answer rnn module)
        with tf.variable_scope("ARnn"):
            aCellFw = cellGenerator.GetCell()
            aCellBw = cellGenerator.GetCell()
            aInitStateFw = aCellFw.zero_state(tf.shape(aRnnInput)[1], floatType)
            aInitStateBw = aCellBw.zero_state(tf.shape(aRnnInput)[1], floatType)
            aRnnRes = tf.nn.bidirectional_rnn(cell_fw=aCellFw, cell_bw=aCellBw,
                inputs=[aRnnInput[i, :, :] for i in xrange(self.aMaxLen) ], 
                initial_state_fw=aInitStateFw, initial_state_bw=aInitStateBw, 
                dtype=floatType, sequence_length=aSeqLen)
            aCode = tf.concat(1, [aRnnRes[1], aRnnRes[2] ] )
            paCode = aCode[0:batchSize, :]
            naCode = aCode[batchSize:, :]

        # construct loss layer
        if self.lossType == "max-margin":
            self.loss, _, paScores, naScores, assertions = MaxMarginLoss(qCode, 
                paCode, naCode, self.naGroupPos, floatType, idType, batchSize)
        elif self.lossType == "nce":
            self.loss, _, paScores, naScores = NCELoss(qCode, paCode, naCode, 
                self.naGroupPos, floatType, idType, batchSize)
            assertions = []

        # get access to all the trainable variables
        self.trainVars = tf.trainable_variables()

        return assertions, paScores, naScores


    def GetNextBatchData(self):
        batchData = {self.qRnnInput : np.array( ( (1, 2, 3), (4, 5, 0) ) ).T, 
            self.qSeqLen : np.array( (3, 2) ),
            self.paRnnInput : np.array( ( (1, 2, 0), (4, 5, 6) ) ).T, 
            self.paSeqLen : np.array( (2, 3) ),
            self.naRnnInput : np.array( ( (2, 3, 4), (3, 4, 5), 
                (6, 0, 0), (9, 10, 0), (10, 11, 12) ) ).T,
            self.naSeqLen : np.array( (3, 3, 1, 2, 3) ),
            self.naGroupPos : np.array( (0, 2, 5) ) }
        return batchData


def main(_):
    nEmbedDim = 2
    nHidden = 2
    nLayer = 1
    cellType = "lstm"
    keepProb = 1.0
    floatType = tf.float32
    idType = tf.int64
    lossType = "max-margin"
    agent = NaiveRnnAgent(nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType)

    agent.qMaxLen = 3
    agent.aMaxLen = 3
    agent.nVocab = 13
    # set up agent loss op
    assertions, _, _ = agent.ConstructGraph(batchSize=2)

    maxIter = 100


    lrFunc = lambda iterNo: 0.00001
    trainer = TfTraining(model=agent, lrFunc=lrFunc, optAlgo="SGD", maxIter=maxIter)


    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.initialize_all_variables() )
            trainer.Run(sess)
            
            # feed = {agent.qRnnInput : np.array( ( (1, 2, 3), (4, 5, 0) ) ).T, 
            #     agent.qSeqLen : np.array( (3, 2) ),
            #     agent.paRnnInput : np.array( ( (1, 2, 0), (4, 5, 6) ) ).T, 
            #     agent.paSeqLen : np.array( (2, 3) ),
            #     agent.naRnnInput : np.array( ( (2, 3, 4), (3, 4, 5), 
            #         (6, 0, 0), (9, 10, 0), (10, 11, 12) ) ).T,
            #     agent.naSeqLen : np.array( (3, 3, 1, 2, 3) ),
            #     agent.naGroupPos : np.array( (0, 2, 5) ) }
            # res = sess.run([agent.loss, ] + assertions, feed_dict=feed)
            
            # inputDict = agent.GetNextBatchData()
            # sess.run(agent.loss, feed_dict=inputDict)

    print "done "


if __name__ == "__main__":
  tf.app.run()


rm src.zip
zip -r src.zip src
cl upload src.zip
cl run naive_rnn.py:src/learning-baseline/naive_rnn.py src:src "python naive_rnn.py" --request-docker-image tensorflow/tensorflow












