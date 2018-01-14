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
from utils.loss_func import NCELoss, MaxMarginLoss, CrossEntLoss
from utils.tf_dl_utils import TfRNNCell, TfTraining, XavierInit
from learning_baseline.context_rnn import ContextRnnAgent
from utils.evaluator import QaEvaluator


DEBUG = False

class BoWContextAgent(ContextRnnAgent):
    def __init__(self, nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev, 
        negSampleSize, articleLevel=True, predTopK=1, flags=None):
        ContextRnnAgent.__init__(self, nEmbedDim, nHidden, nLayer, cellType, 
            keepProb, floatType, idType, lossType, initStdDev, 
            negSampleSize, articleLevel=articleLevel, predTopK=predTopK)
        self.FLAGS = flags


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
        if self.FLAGS.usePreTrain:
            # note we have to make sure the padding row is always zero and not trainable
            self.embedding = tf.concat(0, (tf.zeros( (1, self.nEmbedDim), dtype=floatType), 
                tf.Variable(tf.constant(self.initEmbedding[1:, :] ), 
                dtype=floatType) ), name="embedding")
        else:
            self.embedding = tf.concat(0, (tf.zeros( (1, self.nEmbedDim), dtype=floatType), 
                tf.Variable(tf.random_normal([self.nVocab, self.nEmbedDim], 
                stddev=self.initStdDev), dtype=floatType) ), name="embedding")
  
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
        # self.projection = tf.Variable(tf.random_normal([2 * self.nEmbedDim, self.nEmbedDim], 
        #     stddev=self.initStdDev), name="projection", dtype=floatType)
        self.projection = tf.Variable(XavierInit(2 * self.nEmbedDim, self.nEmbedDim, 
            [2 * self.nEmbedDim, self.nEmbedDim], floatType=floatType), name="projection", dtype=floatType)

        pCode = tf.concat(concat_dim=1, values=[paCode, pcCode] )
        nCode = tf.concat(concat_dim=1, values=[naCode, ncCode] )
        naCodeComb = tf.matmul(nCode, self.projection)
        paCodeComb = tf.matmul(pCode, self.projection)

        paScores, naScores, _ = self.GetPaAndNaScores(qCode, paCodeComb, 
            naCodeComb, self.naGroupPos, floatType, idType, batchSize)

        # construct loss layer
        if self.lossType == "max-margin":
            loss, paScores, naScores, assertions = \
                MaxMarginLoss(paScores, naScores, self.naGroupPos, floatType, idType, batchSize)
        elif self.lossType == "nce":
            loss, paScores, naScores = \
                NCELoss(paScores, naScores, self.naGroupPos, floatType, idType, batchSize)
            assertions = []
        elif self.lossType == "cross-entropy":
            loss, paScores, naScores = \
                CrossEntLoss(paScores, naScores, self.naGroupPos, floatType, idType, batchSize)
            assertions = []
        else:
            raise Exception("Loss type is not implemented")

        if self.l2Regularizer == 0:
            self.loss = loss
        else:
            self.loss = loss + self.l2Regularizer * tf.nn.l2_loss(self.projection)

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

        aCodeConcat = tf.concat(concat_dim=1, values=[aCode, cCode] )
        aCodeComb = tf.matmul(aCodeConcat, self.projection)

        self.evalScore = tf.matmul(qCode, aCodeComb, transpose_b=True) + self.scoreBias


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
    flags = tf.flags
    # summary folder for tensorboard visualization
    suffix = "-pretrain-100d-1e1-fake-unk-0.1-sgd-xent-200-phase"
    if not os.path.exists("./output"):
        os.makedirs("./output")
    summaryPath = "./output/summary" + suffix
    if os.path.exists(summaryPath):
        shutil.rmtree(summaryPath)
    os.makedirs(summaryPath)
    # check point folder for recovery
    ckptPath = "./output/ckpt" + suffix
    # ckptPath = "/Users/Jian/Downloads"
    if not os.path.exists(ckptPath):
        os.makedirs(ckptPath)
    # save prediction and evaluations
    predPath = "./output/pred" + suffix
    if not os.path.exists(predPath):
        os.makedirs(predPath)
    flags.DEFINE_string("summaryPath", summaryPath, "path for tensorboard summary")
    flags.DEFINE_string("ckptPath", ckptPath, "path for checkpoint files")
    flags.DEFINE_string("predPath", predPath, "path for predicion files")
    flags.DEFINE_integer("summaryFlushInterval", 100, "intervals of flushing summary for tensorboard")
    flags.DEFINE_integer("evalInterval", 2000, "intervals to evluate performance on training and testing data")
    flags.DEFINE_integer("ckptInterval", 2000, "intervals of save model check points")
    flags.DEFINE_bool("doRestore", False, "True if loading from check points")
    flags.DEFINE_bool("usePreTrain", True, "True if use pretrained glove vectors")
    flags.DEFINE_bool("useServer", False, "True if using codalab server")
    
    flags.DEFINE_float("fakeTrainUnkRate", 0.1, "artificially set 10% tokens in trainning data to <unk>")
    FLAGS = flags.FLAGS
    if FLAGS.doRestore:
        ckptFile = flags.FLAGS.ckptPath + "/model.ckpt-17001"
        # ckptFile = "/Users/Jian/Downloads/model.ckpt-30000"


    # construct qa agent
    nEmbedDim = 100
    nHidden = 100
    nLayer = 1 
    cellType = "lstm" 
    keepProb = 0.8
    floatType = tf.float32
    idType = tf.int32 
    lossType = "cross-entropy"
    batchSize = 64
    # set the path of files
    if FLAGS.useServer:
        trainCandidateFile = "./train-candidatesal.proto"
        trainOrigFile = "./train-annotated.proto"
        evalCandidateFile = "./dev-candidatesal.proto"
        evalOrigFile = "./dev-annotated.proto"
        vocabPath = "./vocab_dict/proto/vocab_dict"
        preTrainVecFile = "./glove/glove.6B." + str(nEmbedDim) + "d.init.npy"
    else:
        trainCandidateFile = "/Users/Jian/Data/research/squad/dataset/proto/train-candidatesal.proto"
        trainOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
        evalCandidateFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-candidatesal.proto"
        evalOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
        vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/vocab_dict"
        preTrainVecFile = "/Users/Jian/Data/research/squad/dataset/glove/glove.6B." + str(nEmbedDim) + "d.init.npy"

    # DEBUG
    doDebug = False
    # END of DEBUG

    negSampleSize = 100000
    initStdDev = 0.1
    articleLevel = False
    l2Regularizer = 0
    predTopK = 10
    agent = BoWContextAgent(nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev, 
        negSampleSize, articleLevel, predTopK, FLAGS)
    agent.LoadTrainData(trainCandidateFile, trainOrigFile, doDebug)
    agent.LoadEvalData(evalCandidateFile, evalOrigFile, doDebug)
    agent.LoadVocab(vocabPath) 


    agent.PrepareData(doTrain=True)
    # agent.ShuffleData(seed=0)
    agent.PrepareData(doTrain=False)
    raw_input("done")


    if FLAGS.usePreTrain:     
        agent.LoadInitWordVec(preTrainVecFile)

    agent.ConstructInputNode(batchSize)
    agent.ConstructEvalInputNode()
    agent.ConstructGraph(l2Regularizer)
    agent.ConstructEvalGraph()
    # preparing training / evaluation data
    agent.PrepareData(doTrain=True)
    agent.ShuffleData(seed=0)
    agent.PrepareData(doTrain=False)
    # prepare for evaluation on both training and 
    agent.PrepareEvalInput(onTrain=True)
    agent.PrepareEvalInput(onTrain=False)

    # agent.GetUnkRate(doTrain=True)
    # agent.GetUnkRate(doTrain=False)

    # construct qa evaluator
    evaluator = QaEvaluator(wordToId=agent.wordToId, idToWord=agent.idToWord,
        metrics=("exact-match-top-1", "exact-match-top-3", "exact-match-top-5", 
        "in-sentence-rate-top-1", "in-sentence-rate-top-3", "in-sentence-rate-top-5",
        "overlap-match-rate-1.0-top-1", "overlap-match-rate-1.0-top-3", "overlap-match-rate-1.0-top-5") )

    # exponentially decreasing every 200 iterations
    # lrFunc = lambda iterNo: 0
    lrFunc = lambda iterNo: 1e1 * 0.975**(iterNo/200) #if iterNo < 60 else 0.0 #if iterNo < 650 else 2.5*1e-3 * 0.975**(iterNo/200)
    maxIter = 30000
    trainer = TfTraining(model=agent, evaluator=evaluator, lrFunc=lrFunc, 
        optAlgo="SGD", maxGradNorm=10, maxIter=maxIter, FLAGS=FLAGS)

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


# cl rm ^1-3
# rm src.zip vocab_dict.zip
# zip -r src.zip src
# zip -r vocab_dict.zip dataset/proto/vocab_dict
# cl upload src.zip
# cl upload vocab_dict.zip
# cl run train-annotated.proto:0x981923/train-annotated.proto train-candidatesal.proto:0x03fc46/train-candidatesal.proto dev-annotated.proto:0x753738/dev-annotated.proto dev-candidatesal.proto:0xdfa81b/dev-candidatesal.proto vocab_dict:vocab_dict src:src glove:glove bow_context_rnn.py:src/learning_baseline/bow_context_rnn.py "python bow_context_rnn.py" -n bow_dim_100_1e1_fake_unk_0.1_sgd_x_ent_200_phase --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3
    















