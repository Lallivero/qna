import numpy as np
import tensorflow as tf
import os
import sys
import json
import random
import time

sys.path.append("./src/")
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.data_processor import DataProcessor
from utils.squad_utils import LoadJsonData, ParseJsonData
from utils.squad_utils import ReconstructStrFromSpan
from utils.loss_func import NCELoss, MaxMarginLoss
from utils.tf_dl_utils import TfRNNCell, TfTraining
from learning_baseline.context_rnn import ContextRnnAgent
from learning_baseline.bow_context_rnn import BoWContextAgent


def GetContextRnnAgent():
    '''
    it test the correctness of data derived for training.
    The functionality of word id conversion is also tested.
    '''
    nEmbedDim = 10
    nHidden = 10 
    nLayer = 1 
    cellType = "lstm" 
    keepProb = 1
    floatType = tf.float64
    idType = tf.int32 
    lossType = "max-margin"

    trainCandFile = "/Users/Jian/Data/research/squad/dataset/proto/old/qa-1460521688980_new-train-candidatesal.proto"
    trainOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/old/qa-1460521688980_new-train-annotated.proto"
    # evalCandFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-candidatesal.proto"
    # evalOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-annotated.proto"

    evalCandFile = "/Users/Jian/Data/research/squad/dataset/proto/old/qa-1460521688980_new-dev-candidatesal.proto"
    evalOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/old/qa-1460521688980_new-dev-annotated.proto"
    vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/old/1460521688980_new_dict"

    # agent = ContextRnnAgent(nEmbedDim, nHidden, nLayer, cellType, 
    #     keepProb, floatType, idType, lossType, negSampleSize=50)
    agent = BoWContextAgent(nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev=0.1, negSampleSize=50, predTopK=10)
    # agent = BoWContextAgent(nEmbedDim, nHidden, nLayer, cellType, 
    #     keepProb, floatType, idType, lossType, negSampleSize=50, predTopK=10)

    # agent.LoadTrainData(trainCandFile, trainOrigFile, doDebug=True)
    # agent.LoadEvalData(evalCandFile, evalOrigFile, doDebug=True)
    # agent.LoadVocab(vocabPath)
    # # prepare data for both training and evaluation
    # agent.PrepareData(doTrain=True)
    # agent.PrepareData(doTrain=False)
    return agent


def TestComputeGraphForward():
    agent = GetContextRnnAgent()
    agent.ShuffleData()
    random.seed(time.time() )
    sampleIter = random.randint(0, len(agent.trainSamples) - 1)
    agent.sampleIter = sampleIter

    batchSize = 100
    negSampleSize = 50
    agent.ConstructInputNode(batchSize=batchSize)
    assertions = agent.ConstructBoWGraph()
    batchData = agent.GetNextTrainBatch(negSampleSize)

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.initialize_all_variables() )
            results = sess.run([agent.loss, ] + assertions, feed_dict=batchData)
    print results


def TestGetNextTrainBatch():
    agent = GetContextRnnAgent()
    agent.ShuffleData()
    random.seed(time.time() )
    sampleIter = random.randint(0, len(agent.trainSamples) - 1)
    agent.sampleIter = sampleIter
    # batchSize = random.randint(len(agent.trainSamples) / 2, len(agent.trainSamples) - 1)
    batchSize = 512
    agent.ConstructInputNode(batchSize=batchSize)
    # agent.ConstructBoWGraph()
    batchData = agent.GetNextTrainBatch()

    batchData[agent.qRnnInput] = batchData[agent.qRnnInput].T
    batchData[agent.paRnnInput] = batchData[agent.paRnnInput].T
    batchData[agent.naRnnInput] = batchData[agent.naRnnInput].T
    batchData[agent.pcRnnInput] = batchData[agent.pcRnnInput].T
    batchData[agent.ncRnnInput] = batchData[agent.ncRnnInput].T

    # assert the maximal length of each data and padding correctness
    idx = np.argmax(batchData[agent.qRnnInput] == 0, axis=0)
    # in case of full length sequence
    idx[np.where(idx == 0) ] = agent.qMaxLen
    assert np.all(idx == batchData[agent.qSeqLen] )
    assert np.all(np.sum(batchData[agent.qRnnInput] != 0, axis=0) == batchData[agent.qSeqLen])
    assert np.max(idx) == agent.qMaxLen
    idx = np.argmax(batchData[agent.paRnnInput] == 0, axis=0)
    idx[np.where(idx == 0) ] = agent.paMaxLen
    assert np.all(idx == batchData[agent.paSeqLen] )
    assert np.all(np.sum(batchData[agent.paRnnInput] != 0, axis=0) == batchData[agent.paSeqLen])
    assert np.max(idx) == agent.paMaxLen
    idx = np.argmax(batchData[agent.naRnnInput] == 0, axis=0)
    idx[np.where(idx == 0) ] = agent.naMaxLen
    assert np.all(idx == batchData[agent.naSeqLen] )  
    assert np.all(np.sum(batchData[agent.naRnnInput] != 0, axis=0) == batchData[agent.naSeqLen]) 
    assert np.max(idx) == agent.naMaxLen
    idx = np.argmax(batchData[agent.pcRnnInput] == 0, axis=0)
    idx[np.where(idx == 0) ] = agent.pcMaxLen
    assert np.all(idx == batchData[agent.pcSeqLen] )  
    assert np.all(np.sum(batchData[agent.pcRnnInput] != 0, axis=0) == batchData[agent.pcSeqLen]) 
    assert np.max(idx) == agent.pcMaxLen
    idx = np.argmax(batchData[agent.ncRnnInput] == 0, axis=0)
    idx[np.where(idx == 0) ] = agent.ncMaxLen
    assert np.all(idx == batchData[agent.ncSeqLen] )  
    assert np.all(np.sum(batchData[agent.ncRnnInput] != 0, axis=0) == batchData[agent.ncSeqLen]) 
    assert np.max(idx) == agent.ncMaxLen

    # assert pa na has no overlapping (important)
    cumPos = batchData[agent.naGroupPos]
    for i in range(0, batchSize):
        paInputLen = batchData[agent.paSeqLen][i]
        paInput = batchData[agent.paRnnInput][:paInputLen, i]
        for j in range(cumPos[i], cumPos[i + 1] ):
            naInputLen = batchData[agent.naSeqLen][j]
            naInput = batchData[agent.naRnnInput][:naInputLen, j]
            assert paInput.tolist() != naInput.tolist()

    # assert context sentene span are all included
    cumPos = batchData[agent.naGroupPos]
    for i in range(0, batchSize):
        sample = agent.trainSamples[sampleIter + i]
        spanList = agent.trainCandidates[sample.title][sample.pAnsParaId][sample.pAnsSenId]
        # be careful about correct span is not in the first part of the negative example list
        for j in range(cumPos[i], min(cumPos[i + 1], cumPos[i + 1] + len(spanList) - 1 ), cumPos[i + 1] ):
            naInputLen = batchData[agent.naSeqLen][j]
            naInput = batchData[agent.naRnnInput][:naInputLen, j]
            if naInput.tolist() not in spanList:
                intersect = False
                for span in spanList:
                    if len(set(naInput.tolist() ) & set(span) ) != 0:
                        intersect = True
                assert intersect == True

    # assert the correct span is in the context sentence
    for i in range(0, batchSize):
        sample = agent.trainSamples[sampleIter + i]        
        paInputLen = batchData[agent.paSeqLen][i]
        paInput = batchData[agent.paRnnInput][:paInputLen, i]
        cInputLen = batchData[agent.pcSeqLen][i]
        cInput = batchData[agent.pcRnnInput][:cInputLen, i]

        contextStr = "".join(agent.IdToWord(cInput.tolist() ) )
        ansStr = "".join(agent.IdToWord(paInput.tolist() ) )
        # there are cases the answer contains partial words
        assert ansStr in contextStr
        # if len(set(cInput.tolist() ) & set(paInput.tolist() ) ) != len(set(paInput.tolist() ) ):
        #     print "test ", sample.title, sample.id, sample.pAnsParaId, sample.pAnsSenId
        #     print "test2 ", agent.IdToWord(cInput.tolist() ), agent.IdToWord(paInput.tolist() )
        # assert len(set(cInput.tolist() ) & set(paInput.tolist() ) ) == len(set(paInput.tolist() ) )

    # assert negative answer and negative context are associated
    assert batchData[agent.naSeqLen].size == batchData[agent.ncSeqLen].size
    for i in range(len(batchData[agent.naSeqLen] ) ):
        ansLen = batchData[agent.naSeqLen][i]
        ans = batchData[agent.naRnnInput][:ansLen, i]
        contextLen = batchData[agent.naSeqLen][i]
        context = batchData[agent.naRnnInput][:contextLen, i]
        ansStr = "".join(agent.IdToWord(ans.tolist() ) )
        contextStr = "".join(agent.IdToWord(context.tolist() ) )
        assert ansStr in contextStr

    print "context rnn GetNextTrainBatch test passed!"


def TestPrepareEvalInput(articleLevel=True):
    def TestEvalInput(evalCandPadded, evalContextPadded, evalCandLen, evalContextLen, withUnk=False):
        if withUnk == False:
            idx = np.argmax(evalCandPadded == 0, axis=1)
            # in case of full length sequence
            idx[np.where(idx == 0) ] = evalCandPadded.shape[1]
            assert np.all(idx == evalCandLen)
            assert np.all(np.sum(evalCandPadded != 0, axis=1) == evalCandLen)

            idx = np.argmax(evalContextPadded == 0, axis=1)
            # in case of full length sequence
            idx[np.where(idx == 0) ] = evalContextPadded.shape[1]
            assert np.all(idx == evalContextLen)
            assert np.all(np.sum(evalContextPadded != 0, axis=1) == evalContextLen)
        else:
            idx = np.argmax(np.fliplr(evalCandPadded) != 0, axis=1)
            idx = evalCandPadded.shape[1] - idx
            assert np.all(idx <= evalCandLen)
            idx = np.argmax(np.fliplr(evalContextPadded) != 0, axis=1)
            idx = evalContextPadded.shape[1] - idx
            assert np.all(idx <= evalContextLen)

        for i in range(evalCandPadded.shape[0] ):
            cand = [str(idx) for idx in evalCandPadded[i, :evalCandLen[i] ].tolist() ]
            context = [str(idx) for idx in evalContextPadded[i, :evalContextLen[i] ].tolist() ]
            assert "".join(cand) in "".join(context)


    agent = GetContextRnnAgent()
    agent.articleLevel = articleLevel
    # prepare for evaluation on both training and 
    agent.PrepareEvalInput(onTrain=True)
    agent.PrepareEvalInput(onTrain=False)

    # test train evaluation input data
    data = agent.trainCandInput
    evalCandPaddedAll, evalContextPaddedAll, evalCandLenAll, evalContextLenAll = data
    for title in evalCandPaddedAll.keys():
        evalCandPadded = evalCandPaddedAll[title]
        evalContextPadded = evalContextPaddedAll[title]
        evalCandLen = evalCandLenAll[title]
        evalContextLen = evalContextLenAll[title]
        if agent.articleLevel:
            TestEvalInput(evalCandPadded, evalContextPadded, evalCandLen, evalContextLen)
        else:
            for cand, context, candLen, contextLen in zip(evalCandPadded, evalContextPadded, evalCandLen, evalContextLen):
                TestEvalInput(cand, context, candLen, contextLen) 

    data = agent.evalCandInput
    evalCandPaddedAll, evalContextPaddedAll, evalCandLenAll, evalContextLenAll = data
    for title in evalCandPaddedAll.keys():
        evalCandPadded = evalCandPaddedAll[title]
        evalContextPadded = evalContextPaddedAll[title]
        evalCandLen = evalCandLenAll[title]
        evalContextLen = evalContextLenAll[title]
        if agent.articleLevel:
            TestEvalInput(evalCandPadded, evalContextPadded, evalCandLen, evalContextLen, withUnk=True)
        else:
            for cand, context, candLen, contextLen in zip(evalCandPadded, evalContextPadded, evalCandLen, evalContextLen):
                TestEvalInput(cand, context, candLen, contextLen, withUnk=True) 

    print "Prepare evaluation input test passed!"


def TestPredict(articleLevel=True):
    # pass
    agent = GetContextRnnAgent()
    agent.ConstructInputNode(batchSize=512)
    agent.ConstructEvalInputNode()
    agent.ConstructGraph()
    agent.ConstructEvalGraph()
    agent.articleLevel = articleLevel
    # prepare for evaluation on both training and 
    agent.PrepareEvalInput(onTrain=True)
    agent.PrepareEvalInput(onTrain=False)
    # assert the prediction is in the correct scope
    candInput = agent.trainCandInput
    candGlobalId = agent.trainCandGlobalId
    candidates = agent.trainCandidates
    origData = agent.trainOrigData

    sess = tf.Session()
    agent.session = sess
    agent.session.run(tf.initialize_all_variables() )
    predictionTrain = agent.PredictTrainSamples()
    candInput = agent.evalCandInput
    candGlobalId = agent.evalCandGlobalId
    candidates = agent.evalCandidates
    origData = agent.evalOrigData
    predictionEval = agent.PredictEvalSamples()

    print "start eval predictions"

    # for samples, origData, prediction in zip([agent.trainSamples, agent.evalSamples], [agent.trainOrigData, agent.evalOrigData], [predictionTrain, predictionEval] ):
    for samples, origData, prediction in zip([agent.evalSamples], [agent.evalOrigData], [predictionEval] ):
        # samples = agent.trainSamples
        # origData = agent.trainOrigData
        for sample in samples:
            idx = sample.id
            title = sample.title
            pred = prediction[idx]
            found = False
            if articleLevel:
                for para in origData[title].paragraphs:
                    for sen in para.context.sentence:
                        contextStr = ReconstructStrFromSpan(sen.token, (0, len(sen.token) ) )
                        if pred.ansStr in contextStr:
                            found = True
                            break
                    if found:
                        break
                assert found
            else:
                paraId = sample.pAnsParaId
                for sen in origData[title].paragraphs[paraId].context.sentence:
                    contextStr = ReconstructStrFromSpan(sen.token, (0, len(sen.token) ) )
                    if pred.ansStr in contextStr:
                        found = True
                        break
                assert found

    print "ContextRnnPrediction test passed!"




if __name__ == "__main__":
    for i in range(1):
        # with tf.variable_scope(str(i) ):
        #     # TestGetNextTrainBatch()
        #     # TestComputeGraphForward()
        #     # TestPrepareEvalInput(articleLevel=True)
        #     # TestPrepareEvalInput(articleLevel=False)
        #     TestPredict(articleLevel=True)

        with tf.variable_scope(str(i) + "plus"):
            TestPredict(articleLevel=True)


