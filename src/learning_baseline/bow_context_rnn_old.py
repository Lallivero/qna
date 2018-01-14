import numpy as np
import tensorflow as tf
import os
import sys
import json
import random
import shutil

sys.path.append("./src/")
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.data_processor import DataProcessor
from utils.squad_utils import ReconstructStrFromSpan
from utils.loss_func import NCELoss, MaxMarginLoss
from utils.tf_dl_utils import TfRNNCell, TfTraining
from learning_baseline.context_rnn import ContextRnnAgent
from utils.evaluator import QaEvaluator


DEBUG = False

class BoWContextAgent(ContextRnnAgent):
    def __init__(self, nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev, 
        negSampleSize, articleLevel=True):
        ContextRnnAgent.__init__(self, nEmbedDim, nHidden, nLayer, cellType, 
            keepProb, floatType, idType, lossType, initStdDev, 
            negSampleSize, articleLevel=articleLevel)


    def ConstructGraph(self, l2Regularizer=0):
        floatType = self.floatType
        idType = self.idType
        batchSize = self.batchSize
        self.l2Regularizer = l2Regularizer

        # construct some additional variables for easy mean operations
        self.invQSeqLen = tf.placeholder(floatType, [batchSize, None], name="inv1")
        self.invPaSeqLen = tf.placeholder(floatType, [batchSize, None], name="inv2")
        self.invPcSeqLen = tf.placeholder(floatType, [batchSize, None], name="inv3")
        self.invNaSeqLen = tf.placeholder(floatType, [None, None], name="inv4")
        self.invNcSeqLen = tf.placeholder(floatType, [None, None], name="inv5")

        # get embedding for question and answer
        self.embedding = tf.concat(0, (tf.zeros( (1, self.nEmbedDim), dtype=floatType), 
            tf.Variable(tf.random_normal([self.nVocab, self.nEmbedDim], 
            stddev=self.initStdDev), name="embedding", dtype=floatType) ) )

        qRnnInput = tf.nn.embedding_lookup(self.embedding, self.qRnnInput)
        paRnnInput = tf.nn.embedding_lookup(self.embedding, self.paRnnInput)
        naRnnInput = tf.nn.embedding_lookup(self.embedding, self.naRnnInput)
        pcRnnInput = tf.nn.embedding_lookup(self.embedding, self.pcRnnInput)
        ncRnnInput = tf.nn.embedding_lookup(self.embedding, self.ncRnnInput)

        # average
        stepAxis = 1
        qCode = tf.reduce_sum(qRnnInput, reduction_indices=stepAxis) * self.invQSeqLen
        paCode = tf.reduce_sum(paRnnInput, reduction_indices=stepAxis) * self.invPaSeqLen
        pcCode = tf.reduce_sum(pcRnnInput, reduction_indices=stepAxis) * self.invPcSeqLen
        naCode = tf.reduce_sum(naRnnInput, reduction_indices=stepAxis) * self.invNaSeqLen
        ncCode = tf.reduce_sum(ncRnnInput, reduction_indices=stepAxis) * self.invNcSeqLen

       
        # Note the row corresponding to <pad> should be constant zeros (not trainable)
        self.projection = tf.Variable(tf.random_normal([2 * self.nEmbedDim, self.nEmbedDim], 
            stddev=self.initStdDev), name="projection", dtype=floatType)

        pCode = tf.concat(concat_dim=1, values=[paCode, pcCode] )
        nCode = tf.concat(concat_dim=1, values=[naCode, ncCode] )
        naCode = tf.matmul(nCode, self.projection)
        paCode = tf.matmul(pCode, self.projection)


        if DEBUG:
            paCode = tf.Print(paCode, (paCode, ), summarize=20, message="train paCode")
            qCode = tf.Print(qCode, (qCode, ), summarize=20, message="train qCode")


        paScores, naScores, _ = self.GetPaAndNaScores(qCode, paCode, 
            naCode, self.naGroupPos, floatType, idType, batchSize)


        # DEBUG
        # if DEBUG:
        paScores = tf.Print(paScores, (paScores, ), summarize=50, message="t graph paScore")
        # naScores = tf.Print(naScores, () )

        # construct loss layer
        if self.lossType == "max-margin":
            loss, paScores, naScores, assertions = \
                MaxMarginLoss(paScores, naScores, self.naGroupPos, floatType, idType, batchSize)
        elif self.lossType == "nce":
            loss, paScores, naScores = \
                NCELoss(paScores, naScores, self.naGroupPos, floatType, idType, batchSize)
            assertions = []
        else:
            raise Exception("Loss type is not implemented")

        if self.l2Regularizer == 0:
            self.loss = loss
        else:
            self.loss = loss + self.l2Regularizer * (tf.nn.l2_loss(self.embedding) \
                + tf.nn.l2_loss(self.projection) ) 

        self.trainVars = tf.trainable_variables()
        return assertions


    def ConstructEvalGraph(self):
        '''
        Do prediction for one sample
        '''
        floatType = self.floatType
        idType = self.idType
        # construct some additional variables for easy mean operations
        self.invQSeqLenEval = tf.placeholder(floatType, [None, None], name="invQSeqLenEval")
        self.invASeqLenEval = tf.placeholder(floatType, [None, None], name="invASeqLenEval")
        self.invCSeqLenEval = tf.placeholder(floatType, [None, None], name="invCSeqLenEval")
        
        # depends on the constructure of training graph
        qRnnInputEval = tf.nn.embedding_lookup(self.embedding, self.qRnnInputEval)
        aRnnInputEval = tf.nn.embedding_lookup(self.embedding, self.aRnnInputEval)
        cRnnInputEval = tf.nn.embedding_lookup(self.embedding, self.cRnnInputEval)

        # average
        stepAxis = 1
        qCode = tf.reduce_sum(qRnnInputEval, reduction_indices=stepAxis) * self.invQSeqLenEval
        aCode = tf.reduce_sum(aRnnInputEval, reduction_indices=stepAxis) * self.invASeqLenEval
        cCode = tf.reduce_sum(cRnnInputEval, reduction_indices=stepAxis) * self.invCSeqLenEval

        aCode = tf.concat(concat_dim=1, values=[aCode, cCode] )
        aCode = tf.matmul(aCode, self.projection)

        # DEBUG
        if DEBUG:
            qCode = tf.Print(qCode, (qCode, ), summarize=20, message="test qCode")
            aCode = tf.Print(aCode, (aCode[31, :], ), summarize=20, message="test aCode")
            self.scoreBias = tf.Print(self.scoreBias, (self.scoreBias, ), message="test scoreBias")

        self.evalScore = tf.matmul(qCode, aCode, transpose_b=True) + self.scoreBias

        # DEBUG
        if DEBUG:
            self.evalScore = tf.Print(self.evalScore, (self.evalScore, ), message="eval scores")


    def GetNextTrainBatch(self):
        batchData = ContextRnnAgent.GetNextTrainBatch(self)
        if self.floatType == tf.float32:
            floatType = np.float32
        else:
            floatType = np.float64
        batchData[self.invQSeqLen] = 1 / np.tile(batchData[self.qSeqLen], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invPaSeqLen] = 1 / np.tile(batchData[self.paSeqLen], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invPcSeqLen] = 1 / np.tile(batchData[self.pcSeqLen], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invNaSeqLen] = 1 / np.tile(batchData[self.naSeqLen], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invNcSeqLen] = 1 / np.tile(batchData[self.ncSeqLen], (self.nEmbedDim, 1) ).T.astype(floatType)
        return batchData


    def GetPredictBatch(self, batchData):
        if self.floatType == tf.float32:
            floatType = np.float32
        else:
            floatType = np.float64
        batchData[self.invQSeqLenEval] = 1 / np.tile(batchData[self.qSeqLenEval], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invASeqLenEval] = 1 / np.tile(batchData[self.aSeqLenEval], (self.nEmbedDim, 1) ).T.astype(floatType)
        batchData[self.invCSeqLenEval] = 1 / np.tile(batchData[self.cSeqLenEval], (self.nEmbedDim, 1) ).T.astype(floatType)
        return batchData


if __name__ == "__main__":
    flags=tf.flags

    trainCandidateFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-candidatesal.proto"
    trainOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-annotated.proto"
    evalCandidateFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-dev-candidatesal.proto"
    evalOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-dev-annotated.proto"
    vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/1460521688980_new_dict"
    
    # trainCandidateFile = "./qa-1460521688980_new-train-candidatesal.proto"
    # trainOrigFile = "./qa-1460521688980_new-train-annotated.proto"
    # evalCandidateFile = "./qa-1460521688980_new-dev-candidatesal.proto"
    # evalOrigFile = "./qa-1460521688980_new-dev-annotated.proto"
    # vocabPath = "./_1460521688980_new_dict"

    # summary folder for tensorboard visualization
    summaryPath = "./summary1"
    if os.path.exists(summaryPath):
        shutil.rmtree(summaryPath)
    os.makedirs(summaryPath)
    # check point folder for recovery
    # ckptPath = "./ckpt"
    ckptPath = "/Users/Jian/Downloads"
    if not os.path.exists(ckptPath):
        os.makedirs(ckptPath)
    # save prediction and evaluations
    predPath = "./pred"
    if not os.path.exists(predPath):
        os.makedirs(predPath)
    flags.DEFINE_string("summaryPath", summaryPath, "path for tensorboard summary")
    flags.DEFINE_string("ckptPath", ckptPath, "path for checkpoint files")
    flags.DEFINE_string("predPath", predPath, "path for predicion files")
    flags.DEFINE_integer("summaryFlushInterval", 100, "intervals of flushing summary for tensorboard")
    flags.DEFINE_integer("evalInterval", 1000, "intervals to evluate performance on training and testing data")
    flags.DEFINE_integer("ckptInterval", 500, "intervals of save model check points")
    flags.DEFINE_bool("doRestore", True, "True if loading from check points")
    if flags.FLAGS.doRestore:
        ckptFile = flags.FLAGS.ckptPath + "/model.ckpt-17001"

    # construct qa agent
    nEmbedDim = 100
    nHidden = 100
    nLayer = 1 
    cellType = "lstm" 
    keepProb = 1
    floatType = tf.float32
    idType = tf.int32 
    lossType = "max-margin"
    # batchSize = 64
    
    # DEBUG
    batchSize = 1
    doDebug = False

    negSampleSize = 100000
    initStdDev = 0.1
    articleLevel = False
    l2Regularizer = 0
    agent = BoWContextAgent(nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev, 
        negSampleSize, articleLevel)
    agent.LoadTrainData(trainCandidateFile, trainOrigFile, doDebug)
    agent.LoadEvalData(evalCandidateFile, evalOrigFile, doDebug)
    agent.LoadVocab(vocabPath)
    agent.ConstructInputNode(batchSize)
    agent.ConstructEvalInputNode()
    agent.ConstructGraph(l2Regularizer)
    agent.ConstructEvalGraph()
    # preparing training / evaluation data
    agent.PrepareData(doTrain=True)
    # agent.ShuffleData(seed=0)
    agent.PrepareData(doTrain=False)
    # prepare for evaluation on both training and 
    agent.PrepareEvalInput(onTrain=True)
    agent.PrepareEvalInput(onTrain=False)

    # construct qa evaluator
    evaluator = QaEvaluator(metrics=("exact-match", "in-sentence-rate") )

    FLAGS = flags.FLAGS
    # exponentially decreasing every 300 iterations
    # best for l2_regularizer part
    # lrFunc = lambda iterNo: 3e-3 * 0.975**(iterNo/200) #if iterNo < 650 else 5e-3 * 0.975**(iterNo/200)
   
    lrFunc = lambda iterNo: 0

    # best for no regularizer cases
    # lrFunc = lambda iterNo: 5e-3 * 0.975**(iterNo/200) if iterNo < 650 else 2.5*1e-3 * 0.975**(iterNo/200)
    maxIter = 20000
    trainer = TfTraining(model=agent, evaluator=evaluator, lrFunc=lrFunc, 
        optAlgo="Adam", maxGradNorm=10, maxIter=maxIter, FLAGS=FLAGS)

    tf.scalar_summary("training-loss", agent.loss)
    for iGrad, grad in enumerate(trainer.grads):
        tf.histogram_summary("grad hist " + str(iGrad), grad)
        tf.scalar_summary("grad norm " + str(iGrad), tf.global_norm( (grad, ) ) )
    summary = tf.merge_all_summaries()
    summaryWriter = tf.train.SummaryWriter(summaryPath)
    summarizer = (summary, summaryWriter)


    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            if FLAGS.doRestore:
                trainer.saver.restore(sess, ckptFile)
            else:
                sess.run(tf.initialize_all_variables() )
            trainer.Run(sess, summarizer)
















