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
from matplotlib.backends.backend_pdf import PdfPages



def save_pdf_from_fig(fig, file_name):
    page = PdfPages(file_name)
    plt.axis("on")
    plt.savefig(page, format='pdf', bbox_inches='tight')
    page.close()


def LoadCandidateData(dataFile):
    dataIn = io.ReadArticles(dataFile, cls=training_dataset_pb2.TrainingArticle)
    dataDict = dict()
    for data in dataIn:
        dataDict[data.title] = data
    return dataDict


def LoadOrigData(dataFile):
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
        for paragraph in data[title].paragraphs:
            for sentence in paragraph.context.sentence:
                results += func(sentence)
    return results


def TraverseParagraph(func, data):
    results = []
    for title in data.keys():
        for paragraph in data[title].paragraphs:
            results += func(paragraph)
    return results


def TraverseArticle(func, data):
    results = []
    for title in data.keys():
        results += func(data[title] )
    return results


def TraverseQuestionAnswer(func, data):
    results = []
    for title in data.keys():
        for paragraph in data[title].paragraphs:
            for qa in paragraph.qas:
                results += func(qa)
    return results


def TraverseQAandContext(func, data):
    results = []
    for title in data.keys():
        for paragraph in data[title].paragraphs:
            context = paragraph.context
            for qa in paragraph.qas:
                results += func(qa, context)
    return results   


def TraverseQAandTargetSentence(func, data, candSenId):
    results = []
    qaKeySet = set(candSenId.keys() )
    for title in data.keys():
        for paragraph in data[title].paragraphs:
            context = paragraph.context
            for qa in paragraph.qas:
                if qa.id in qaKeySet:
                    results += func(qa, context.sentence[candSenId[qa.id] ] )
    return results


def GetNConstituentByPara(candidateArticle):
    nSpan = dict()
    for cand in candidateArticle.candidateAnswers:
        paraId = cand.paragraphIndex
        if paraId not in nSpan.keys():
            nSpan[paraId] = 1
        else:
            nSpan[paraId] += 1
    return [nSpan[key] for key in nSpan.keys() ]


def GetNConstituentByArticle(candidateArticle):
    return [len(candidateArticle.candidateAnswers), ]


def GetNQaPairByArticle(data):
    nQa = 0
    for paragraph in data.paragraphs:
        nQa += len(paragraph.qas)
    return [nQa, ]


def GetOverlapQandContext(qa, context, stopWords):
    overlap = set()
    qToken = set([token.word.lower() for token in qa.question.sentence[0].token] )
    for s in context.sentence:
        sToken = set( [token.word.lower() for token in s.token] )
        overlap = overlap.union(sToken.intersection(qToken) )
    return [len(overlap.difference(stopWords) ), ]


def GetOverlapQandTargetSentence(qa, sentence, stopWords):
    qToken = set( [token.word.lower() for token in qa.question.sentence[0].token] )
    sToken = set( [token.word.lower() for token in sentence.token] )
    overlap = sToken.intersection(qToken)
    return [len(overlap.difference(stopWords) ), ]


def GetAnsSenId(candData):
    ansSenId = dict()
    for title in candData.keys():
        for qa in candData[title].questions:
            pAns = candData[title].candidateAnswers[qa.correctAnswerIndex]
            senId = pAns.sentenceIndex
            ansSenId[qa.id] = senId
    return ansSenId


def GetUnkRateQa():
    FileName = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    evalOrigData = LoadOrigData(FileName)
    vocabPath = "./dataset/proto/vocab_dict"
    with open(os.path.join(vocabPath, "word2id_train.json") ) as fp:
        wordToIdTrain = json.load(fp)
    trainVocabSet = set(wordToIdTrain.keys() )
    cntQa = 0
    cntUnkQ = 0
    cntUnkA = 0
    for title in evalOrigData.keys():
        for paragraph in evalOrigData[title].paragraphs:
            for qa in paragraph.qas:
                cntQa += 1
                for token in qa.question.sentence[0].token:
                    if token.word.lower() not in trainVocabSet:
                        cntUnkQ += 1
                        break
                if len(qa.answer.sentence) > 0:
                    for token in qa.answer.sentence[0].token:
                        if token.word.lower() not in trainVocabSet:
                            cntUnkA += 1
                            break
    print "unk in Q rate ", cntUnkQ / float(cntQa)
    print "unk in A rate ", cntUnkA / float(cntQa)


def averageLength():
    fig = plt.figure(figsize=(10, 5) )
    # # Get the histogram for question length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetQueryLength = lambda qa: [len(qa.question.sentence[0].token), ]
    # results = TraverseQuestionAnswer(GetQueryLength, data)
    # print "mean length question ", np.mean(np.array(results) ), len(results)
    # print "std length question ", np.std(np.array(results) )

    # histRange = (0, 50)
    # bins = 50
    # y, edge = np.histogram(results, bins=bins, range=histRange, normed=1)
    # edge = edge[:-1] #+ histRange[1] / float(bins)
    # plt.plot(edge[4:], y[4:] * 100.0, "ro-", linewidth=2, label="Ave. query len.")

    # average answer length
    FileName = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    data = LoadOrigData(FileName)
    GetAnsLength = lambda qa: [len(qa.answers[0].sentence[0].token) if len(qa.answers[0].sentence) != 0 else 0, ]
    results = TraverseQuestionAnswer(GetAnsLength, data)
    print "mean length answer ", np.mean(np.array(results) ), len(results)
    print "std length answer ", np.std(np.array(results) )
    histRange = (0, 50)
    bins = 50
    y, edge = np.histogram(results, bins=bins, range=histRange, normed=1)
    edge = edge[:-1] #+ histRange[1] / float(bins)
    plt.plot(edge[1:], y[1:] * 100.0, "go-", linewidth=2, label="Average answer length")

    # Get question overlapping containing sentence
    FileName = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    data = LoadOrigData(FileName)
    candFileName = "/Users/Jian/Data/research/squad/dataset/proto/dev-candidatesal.proto"
    candData = LoadCandidateData(candFileName)
    with open("./src/utils/stop_word.txt", "r") as fp:
        stopWords = set( [word.lower().strip() for word in fp.read().splitlines() ] )
    ansSenId = GetAnsSenId(candData)
    GetOverLapQandTargetSentenceWithStopWords = \
        lambda qa, sentence: GetOverlapQandTargetSentence(qa, sentence, stopWords)
    results = TraverseQAandTargetSentence(GetOverLapQandTargetSentenceWithStopWords, data, ansSenId)
    print "mean overlapping q and target s ", np.mean(np.array(results) ), len(results)
    print "std length overlapping q and target s ", np.std(np.array(results) )
    histRange = (0, 20)
    bins = 20
    y, edge = np.histogram(results, bins=bins, range=histRange, normed=1)
    edge = edge[:-1] # + histRange[1] / float(bins)
    plt.plot(edge, y * 100.0, "bo-", linewidth=2, label="Question / answer sentence overlap")
    # plt.hist(results, bins=20, range=(0, 20), normed=1, facecolor="g")
    plt.legend(loc="upper right", fontsize=20)
    plt.xlim( (0, 20) )
    plt.xlabel("# words", fontsize=20)
    plt.ylabel("Percentage", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    fileName = "/Users/Jian/Data/research/squad/paper/figure/ave_length.pdf"
    save_pdf_from_fig(fig, fileName)
    plt.show()





if __name__ == "__main__":
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # print "number of articles ", len(data.keys() )

    # # number of questions
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # results = TraverseArticle(GetNQaPairByArticle, data)
    # print "total qa number ", np.sum(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")

    # # Get the histogram for sentence length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetSenLength = lambda sentence: [len(sentence.token), ]
    # results = TraverseSentence(GetSenLength, data)
    # print "mean length context sentence ", np.mean(np.array(results) ), len(results)
    # histRange = (0, 150)
    # bins = 50
    # y, edge = np.histogram(results, bins=bins, range=histRange)
    # edge = edge[:-1] + histRange[1] / float(bins)
    # plt.semilogx(edge, y, "ro-")
    # # plt.hist(results, bins=50, range=(0, 150), normed=1, facecolor="g")
    # # plt.show()

    # # Get the histogram for paragraph length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetParaLength = lambda para: [sum( [len(sentence.token) for sentence in para.context.sentence] ), ]
    # results = TraverseParagraph(GetParaLength, data)
    # print "mean length context paragraph ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    # # plt.show()

    # # Get the histogram for article length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetParaLength = lambda para: [sum( [len(sentence.token) for sentence in para.context.sentence] ), ]
    # GetArticleLength = lambda article: [sum( [GetParaLength(para)[0] for para in article.paragraphs] ), ]
    # results = TraverseArticle(GetArticleLength, data)
    # print "mean length context article ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    # # plt.show()

    # # # Get question for question length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetQueryLength = lambda qa: [len(qa.question.sentence[0].token), ]
    # results = TraverseQuestionAnswer(GetQueryLength, data)
    # print "mean length question ", np.mean(np.array(results) ), len(results)
    # plt.figure()
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    # # # plt.show()

    # # Get answer length
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetAnsLength = lambda qa: [len(qa.answer.sentence[0].token) if hasattr(qa, "answer") and len(qa.answer.sentence) != 0 else 0, ]
    # results = TraverseQuestionAnswer(GetAnsLength, data)
    # print "mean length answer ", np.mean(np.array(results) ), len(results)
    # histRange = (0, 50)
    # bins = 50
    # y, edge = np.histogram(results, bins=bins, range=histRange)
    # edge = edge[:-1] + histRange[1] / float(bins)
    # plt.semilogx(edge, y, "go-")
    # plt.figure()
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    # plt.show()

    # # Get the histogram for article level # of constitunet
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-candidatesal.proto"
    # data = LoadCandidateData(FileName)
    # results = TraverseArticle(GetNConstituentByArticle, data)
    # print "mean # constituent per article ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    
    # # Get the histogram for paragraph level # of constituent
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-candidatesal.proto"
    # data = LoadCandidateData(FileName)
    # results = TraverseArticle(GetNConstituentByPara, data)
    # print "mean # constituent per paragraph ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")

    # # number of paragraph per article
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # GetNParagraph = lambda article: [len(article.paragraphs), ]
    # results = TraverseArticle(GetNParagraph, data)
    # print "mean length context article ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=50, normed=1, facecolor="g")
    # plt.show()

    # # get question overlapping
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # with open("./src/utils/stop_word.txt", "r") as fp:
    #     stopWords = set( [word.lower().strip() for word in fp.read().splitlines() ] )
    # GetOverLapQandContextWithStopWords = \
    #     lambda qa, context: GetOverlapQandContext(qa, context, stopWords)
    # results = TraverseQAandContext(GetOverLapQandContextWithStopWords, data)
    # print "mean overlapping q and c ", np.mean(np.array(results) ), len(results)
    # plt.hist(results, bins=20, range=(0, 20), normed=1, facecolor="g")
    # plt.show()

    # # Get question overlapping containing sentence
    # FileName = "/Users/Jian/Data/research/squad/dataset/proto/train-annotated.proto"
    # data = LoadOrigData(FileName)
    # candFileName = "/Users/Jian/Data/research/squad/dataset/proto/train-candidatesal.proto"
    # candData = LoadCandidateData(candFileName)
    # with open("./src/utils/stop_word.txt", "r") as fp:
    #     stopWords = set( [word.lower().strip() for word in fp.read().splitlines() ] )
    # ansSenId = GetAnsSenId(candData)
    # GetOverLapQandTargetSentenceWithStopWords = \
    #     lambda qa, sentence: GetOverlapQandTargetSentence(qa, sentence, stopWords)
    # results = TraverseQAandTargetSentence(GetOverLapQandTargetSentenceWithStopWords, data, ansSenId)
    # print "mean overlapping q and target s ", np.mean(np.array(results) ), len(results)
    # histRange = (0, 20)
    # bins = 20
    # y, edge = np.histogram(results, bins=bins, range=histRange)
    # edge = edge[:-1] + histRange[1] / float(bins)
    # plt.semilogx(edge, y, "ro-")
    # # plt.hist(results, bins=20, range=(0, 20), normed=1, facecolor="g")
    # plt.show()

    # # Get unknown vocabulary size
    # vocabPath = "./dataset/proto/vocab_dict"
    # with open(os.path.join(vocabPath, "word2id_train.json") ) as fp:
    #     wordToIdTrain = json.load(fp)
    # with open(os.path.join(vocabPath, "word2id_dev.json") ) as fp:
    #     wordToIdDev = json.load(fp)
    # print "train vocab ", len(wordToIdTrain.keys() )
    # print "dev vocab ", len(wordToIdDev.keys() )
    # print "unknown ", len(set(wordToIdDev.keys() ).difference(set(wordToIdTrain.keys() ) ) )

    # GetUnkRateQa()
    averageLength()








