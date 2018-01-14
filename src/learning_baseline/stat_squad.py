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
import matplotlib.pyplot as plt


def LoadCandidateData(dataFile):
    dataIn = io.ReadArticles(dataFile, cls=training_dataset_pb2.TrainingArticle)
    dataDict = dict()
    for data in dataIn:
        dataDict[data.title] = data
    return dataDict


def LoadOrigData(self, dataFile):
    dataIn = io.ReadArticles(dataFile, cls=dataset_pb2.Article)
    dataDict = dict()
    for data in dataIn:
        dataDict[data.title] = data
    return dataDict


def TraverseSentence(func, data):
    '''
    the data is original dataset protobuf file.
    It can be loaded with LoadOrigData
    append results to a result list
    '''
    results = []
    for title in data.keys():
        for paragraph in data[title]:
            for sentence in paragraph.context.sentence:
                results += func(sentence)


def TraverseParagraph(func):
    pass


def TraverseQuestions(func):
    pass


def TraverseAnswers(func):
    pass


if __name__ == "__main__":
    # Get the histogram for sentence length
    FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    data = LoadOrigData(File)
    GetSenLength = lambda x: len(sentence.token)
    results = TraverseSentence(GetSenLength, data)
    plt.hist(results, normed=1, facecolor="b")





