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
from rnn_agent import RnnAgent


def GetRnnAgent():
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

    trainDataFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-annotated-train-candidates-1460521688980_new.proto"
    origDataFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-annotated-full-1460521688980_new.proto"
    vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/1460521688980_new_dict"

    agent = RnnAgent(nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType)
    agent.LoadTrainingData(trainDataFile)
    agent.LoadOriginalData(origDataFile)
    agent.LoadVocab(vocabPath)
    agent.PrepareData()
    return agent


def TestPrepareData():
    agent = GetRnnAgent()
    # cntWrong = 0
    for sample in agent.samples:
        title = sample.title
        qaId = sample.id
        query = sample.query
        paraId = sample.pAnsParaId
        senId = sample.pAnsSenId
        pAnsId = sample.pAnsId
        qaOrig = None

        for qa in agent.origData[title].paragraphs[paraId].qas:
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
    print "Data preparation test passed!"


if __name__ == "__main__":
    TestPrepareData()
