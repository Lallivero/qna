import numpy as np
import tensorflow as tf
import os
import sys
import json
import random
import time

sys.path.append("./src/")
from utils.tf_dl_utils import TfRNNCell, TfTraining
from proto import io
from proto import CoreNLP_pb2
from proto import dataset_pb2
from proto import training_dataset_pb2
from agent import Agent


def GetAgent():
    '''
    it test the correctness of data derived for training.
    The functionality of word id conversion is also tested.
    '''
    nEmbedDim = 10
    nHidden = 10 
    nLayer = 1 
    cellType = "lstm" 
    keepProb = 1
    floatType = tf.float32
    idType = tf.int32 
    lossType = "max_margin"

    trainDataFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-candidatesal.proto"
    origDataFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-1460521688980_new-train-annotated.proto"
    vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/1460521688980_new_dict"

    agent = Agent(floatType, idType, lossType, articleLevel=True)
    # agent.LoadTrainingData(trainDataFile)
    # agent.LoadOriginalData(origDataFile)
    agent.LoadTrainData(trainDataFile, origDataFile, doDebug=True)
    agent.LoadVocab(vocabPath)
    agent.PrepareData()
    return agent


def TestPrepareData():
    agent = GetAgent()
    # cntWrong = 0
    for sample in agent.samples:
        title = sample.title
        qaId = sample.id
        query = sample.query
        paraId = sample.pAnsParaId
        senId = sample.pAnsSenId
        pAnsId = sample.pAnsId
        qaOrig = None

        for qa in agent.trainOrigData[title].paragraphs[paraId].qas:
            if qa.id == qaId:
                qaOrig = qa
                break
        assert qaOrig != None

        # test if the queries are the same
        qWord = np.array(agent.IdToWord(query) )
        qWordOrig = np.array( [token.word.lower() for token in qaOrig.question.sentence[0].token] )
        if not np.all(qWord == qWordOrig):
            print "qWord error ", qWord, qWordOrig
        assert np.all(qWord == qWordOrig)

        # test if answers are the same
        ansWord = np.array(agent.IdToWord(sample.ans) )
        ansWordOrig = np.array( [token.word.lower() for token in qaOrig.answer.sentence[0].token] )
        # the period is removed if it is the last token of answer
        if (ansWord.size == ansWordOrig.size and not np.all(ansWord == ansWordOrig) ) \
            or (ansWord.size + 1 == ansWordOrig.size and not np.all(ansWord == ansWordOrig[:-1] ) ):
            print "ansWord error ", paraId, senId, pAnsId, qaId
        if ansWord.size == ansWordOrig.size:
            assert np.all(ansWord == ansWordOrig)
        else:
            assert np.all(ansWord == ansWordOrig[:-1] )

        # test if the answer is contained in the context
        ansWord = ansWord.tolist()
        contextWord = agent.IdToWord(sample.context)
        if "".join(ansWord) not in "".join(contextWord):
            print "\ncontext error ", ansWord, contextWord
            print "".join(ansWord), "".join(contextWord)
            # cntWrong += 1
        assert "".join(ansWord) in "".join(contextWord)

        # test correct answer in the globalId list
        paraId = sample.pAnsParaId
        senId = sample.pAnsSenId
        assert sample.pAnsId in agent.trainCandidateGlobalId[title][paraId][senId]

    print "Data preparation test passed!"


def TestNegativeSampling():
    agent = GetAgent()
    agent.ShuffleData()
    random.seed(time.time() )
    sampleIter = random.randint(0, len(agent.trainSamples) - 1)
    agent.sampleIter = sampleIter
    # batchSize = random.randint(len(agent.samples) / 2, len(agent.samples) - 1)
    batchSize = 10
    negSampleSize = 100
    agent.negSampleSize = negSampleSize

    print "start"
    for sample in agent.trainSamples:
        negSample, _, _, _, _, negSampleParaId, negSampleSenId, negSampleSpanId \
            = agent.NegativeSampling(sample, negSampleSize)
        for nSamp, pId, sId, spId in zip(negSample, negSampleParaId, negSampleSenId, negSampleSpanId):
            assert nSamp in agent.trainCandidates[sample.title][pId][sId]
            assert spId in agent.trainCandGlobalId[sample.title][pId][sId]
            assert not (pId == sample.pAnsParaId and sId == sample.pAnsSenId and nSamp == sample.ans)
            # if agent.articleLevel == False:
            #     assert pId == sample.pAnsParaId

    print "Negative sampling test passed!"


if __name__ == "__main__":
    # TestPrepareData()
    TestNegativeSampling()
