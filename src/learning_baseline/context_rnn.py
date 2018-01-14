import numpy as np
import tensorflow as tf
import os
import sys
import json
import random

sys.path.append("./src/")
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.data_processor import DataProcessor
from utils.squad_utils import ReconstructStrFromSpan, ObjDict, UnkrizeData
from utils.loss_func import NCELoss, MaxMarginLoss, CrossEntLoss
from utils.tf_dl_utils import TfRNNCell, TfTraining
from learning_baseline.rnn_agent import RnnAgent
from learning_baseline.agent import QaPrediction, QaData


class ContextRnnAgent(RnnAgent):
    '''
    The model encode answer-related context sentence and answer seperately
    It combines the encoding of context sentence and answer together via
    matrix multiplication. Then the question encoding and the combined encoding
    if feed into a max-margin layer. 
    '''
    def __init__(self, nEmbedDim, nHidden, nLayer, cellType, 
        keepProb, floatType, idType, lossType, initStdDev, 
        negSampleSize, articleLevel=True, predTopK=1):
        RnnAgent.__init__(self, nEmbedDim, nHidden, nLayer, cellType, 
            keepProb, floatType, idType, lossType, initStdDev)
        self.negSampleSize = negSampleSize
        self.articleLevel = articleLevel
        self.predTopK = predTopK
        # self.scores is for prediction use
        self.scores = None


    def ConstructInputNode(self, batchSize):
        '''
        The required input placeholder for class derived from ContextRnnAgent
        Additional input can be addressed in derived class ConstructInputNode
        function. Additional input node for derived model are added in 
        ConstructGraph functions.
        '''
        # placeholders setup
        floatType = self.floatType
        idType = self.idType
        self.batchSize = batchSize
        self.qRnnInput = \
            tf.placeholder(idType, [batchSize, None], name="qRnnInput")
        self.qSeqLen = tf.placeholder(idType, [batchSize], name="qSeqLen") 
        self.paRnnInput = \
            tf.placeholder(idType, [batchSize, None], name="paRnnInput")
        self.paSeqLen = tf.placeholder(idType, [batchSize], name="paSeqLen")
        self.naRnnInput = \
            tf.placeholder(idType, [None, None], name="naRnnInput")
        self.naSeqLen = tf.placeholder(idType, [None], name="naSeqLen")
        self.naGroupPos = tf.placeholder(idType, [batchSize + 1], name="naGroupPos")
        self.pcRnnInput = tf.placeholder(idType, [batchSize, None], name="pcRnnInput")
        self.pcSeqLen = tf.placeholder(idType, [batchSize], name="pcSeqLen")
        self.ncRnnInput = tf.placeholder(idType, [None, None], name="ncRnnInput")
        self.ncSeqLen = tf.placeholder(idType, [None], name="ncSeqLen")


    def ConstructEvalInputNode(self):
        idType = self.idType
        floatType = self.floatType
        # the shape of rnninput is [batchSize, maxStep]
        self.qRnnInputEval = \
            tf.placeholder(idType, [None, None], name="qRnnInputEval")
        self.aRnnInputEval = \
            tf.placeholder(idType, [None, None], name="aRnnInputEval")
        self.cRnnInputEval = \
            tf.placeholder(idType, [None, None], name="cRnnInputEval")
        self.qSeqLenEval = tf.placeholder(idType, [None], name="qSeqLenEval")
        self.aSeqLenEval = tf.placeholder(idType, [None], name="aSeqLenEval")
        self.cSeqLenEval = tf.placeholder(idType, [None], name="cSeqLenEval")


    def GetNextTrainBatch(self):
        # get input sequence length for q pa and context
        # print "Preparing data for the next batch!"
        sampleIter = self.sampleIter
        negSampleSize = self.negSampleSize
        batchSize = self.batchSize
        sampleIter = self.sampleIter
        sampleList = self.trainSamples
        nSample = len(sampleList)
        qSeqLen = [len(sampleList[i % nSample].query) for i in range(sampleIter, sampleIter + batchSize) ]
        paSeqLen = [len(sampleList[i % nSample].ans) for i in range(sampleIter, sampleIter + batchSize) ]
        cSeqLen = [len(sampleList[i % nSample].context) for i in range(sampleIter, sampleIter + batchSize)]
        self.qMaxLen = max(qSeqLen)
        self.paMaxLen = max(paSeqLen)
        self.pcMaxLen = max(cSeqLen)

        # get input sequence for q pa and context
        qSeq = np.zeros( (batchSize, self.qMaxLen) )
        paSeq = np.zeros( (batchSize, self.paMaxLen) )
        cSeq = np.zeros( (batchSize, self.pcMaxLen) )
        for i in range(sampleIter, (sampleIter + batchSize) ):
            idx = i % nSample
            sample = sampleList[idx]
            idx = i - sampleIter
            qSeq[idx, :qSeqLen[idx] ] = sample.query
            paSeq[idx, :paSeqLen[idx] ] = sample.ans
            cSeq[idx, :cSeqLen[idx] ] = sample.context

        # get input sequence for na
        naList = list()
        naSeqLen = list()
        ncList = list()
        ncSeqLen = list()
        naGroupSize = list()
        for i in range(sampleIter, (sampleIter + batchSize) ):
            sample = sampleList[i % nSample]
            nAns, nAnsLen, nAnsGroupSize, nAnsContext, nAnsContextLen, _, _, _ = \
                self.NegativeSampling(sample, negSampleSize)
            naList += nAns
            naSeqLen += nAnsLen
            ncList += nAnsContext
            ncSeqLen += nAnsContextLen
            naGroupSize.append(nAnsGroupSize)
        naGroupSize = [0, ] + naGroupSize
        self.naMaxLen = max(naSeqLen)
        self.ncMaxLen = max(ncSeqLen)
        naSeq = np.zeros( (sum(naGroupSize), self.naMaxLen) )
        ncSeq = np.zeros( (sum(naGroupSize), self.ncMaxLen) )
        for i, (nAns, nAnsContext) in enumerate(zip(naList, ncList) ):
            naSeq[i, :naSeqLen[i] ] = nAns
            ncSeq[i, :ncSeqLen[i] ] = nAnsContext

        if self.idType == tf.int32:
            npIdType = np.int32
        else:
            npIdType = np.int64
        batchData = {self.qRnnInput : qSeq.astype(npIdType), 
            self.qSeqLen : np.array(qSeqLen).astype(npIdType),
            self.paRnnInput : paSeq.astype(npIdType), 
            self.paSeqLen : np.array(paSeqLen).astype(npIdType),
            self.naRnnInput : naSeq.astype(npIdType),
            self.naSeqLen : np.array(naSeqLen).astype(npIdType),
            self.naGroupPos : np.cumsum(np.array(naGroupSize) ).astype(npIdType),
            self.pcRnnInput : cSeq.astype(npIdType),
            self.pcSeqLen: np.array(cSeqLen).astype(npIdType),
            self.ncRnnInput : ncSeq.astype(npIdType),
            self.ncSeqLen : np.array(ncSeqLen).astype(npIdType) }

        if self.FLAGS.fakeTrainUnkRate > 0.0:
            padId = self.wordToId["<pad>"]
            unkId = self.wordToId["<unk>"]
            rate = self.FLAGS.fakeTrainUnkRate
            batchData[self.qRnnInput] = UnkrizeData(batchData[self.qRnnInput], rate / 2.0, padId, unkId)
            batchData[self.paRnnInput] = UnkrizeData(batchData[self.paRnnInput], rate, padId, unkId)
            batchData[self.pcRnnInput] = UnkrizeData(batchData[self.pcRnnInput], rate, padId, unkId)
            batchData[self.naRnnInput] = UnkrizeData(batchData[self.naRnnInput], rate, padId, unkId)
            batchData[self.ncRnnInput] = UnkrizeData(batchData[self.ncRnnInput], rate, padId, unkId)

        # # DEBUG
        # print "\ntrain data pa"
        # print self.IdToWord(batchData[self.qRnnInput][0, :].tolist() )
        # print self.IdToWord(batchData[self.paRnnInput][0, :].tolist() )
        # print self.IdToWord(batchData[self.pcRnnInput][0, :].tolist() )
        # for i in range(batchData[self.naRnnInput].shape[0] ):
        #     print "\n train data na ", i
        #     print self.IdToWord(batchData[self.naRnnInput][i, :].tolist() )
        #     print self.IdToWord(batchData[self.ncRnnInput][i, :].tolist() )
        # # END of DEBUG

        nSample = len(sampleList)
        self.sampleIter += self.batchSize
        self.sampleIter = self.sampleIter % nSample
        return batchData


    def GetPaAndNaScores(self, qCode, paCode, naCode, naGroupPos, floatType, idType, batchSize):
        scoreBias = tf.Variable(tf.zeros([1], dtype=floatType), name="score-bias")
        # scoreBias = tf.get_variable("score_bias", [1], dtype=floatType)        
        self.scoreBias = scoreBias
        paScores = [tf.matmul(qCode[i:(i + 1), :], paCode[i:(i + 1), :], 
            transpose_b=True) + scoreBias for i in xrange(batchSize) ]
        paScores = tf.concat(0, paScores)

        naScoreList = []
        assertions = []
        for i in xrange(batchSize):
            # TODO better way to slice the tensor
            beginPos = tf.concat(0, [naGroupPos[i:(i+1) ], tf.constant([0,], dtype=idType) ] )
            nNaSample = naGroupPos[ (i+1):(i+2) ] - naGroupPos[i]
            # Assert the sample has non-zero negative answers
            assertions.append(tf.Assert(tf.greater(naGroupPos[i+1] - naGroupPos[i], 0), \
                ["Zero negative answers in max margin loss!"] ) )
            sliceSize = tf.concat(0, [nNaSample, tf.constant( [-1, ], dtype=idType) ] )
            # TODO the current version seems to require beginPos and sliceSize to be int32
            # however other places required int64. we keep the global setting to be int64
            # while the others 
            naCodeSlice = tf.slice(naCode, tf.cast(beginPos, tf.int32), tf.cast(sliceSize, tf.int32) )
            scores = tf.matmul(qCode[i:(i + 1), :], naCodeSlice, transpose_b=True) + scoreBias
            naScoreList.append(scores)
        naScores = tf.transpose(tf.concat(1, naScoreList) )
        return paScores, naScores, scoreBias


    def PrepareEvalInput(self, onTrain=False):
        '''
        @param onTrain: True if evaluate over training set.
        False if evaluate over evaluation set (dev now and test in the future)
        @return evalCandPadded, evalContextPadded, evalCandLen, evalContextLen:
        indexed first by article title. Later on, if in articleLevel,
        they will be plain list, otherwise it is organized as list of list describing 
        each paragraph
        '''
        # print "Preparing evaluation input!"
        if onTrain:
            evalCandidates = self.trainCandidates
            evalOrig = self.trainOrigData
        else:
            evalCandidates = self.evalCandidates
            evalOrig = self.evalOrigData
        evalCandPadded = dict()
        evalContextPadded = dict()
        evalCandLen = dict()
        evalContextLen = dict()
        if self.articleLevel:
            for title in evalCandidates.keys():
                article = evalOrig[title]
                candidates = evalCandidates[title]
                nCand = sum( [len(candSen) for candPara in candidates for candSen in candPara] )
                maxCandLen = max( [max( [max( [len(cand) \
                    for cand in candSen if len(candSen) != 0] ) for candSen in candPara] ) for candPara in candidates] )
                maxContextLen = max( [ max( [len(sen.token) for sen in para.context.sentence] ) for para in article.paragraphs] )
                candMat = np.zeros( (nCand, maxCandLen) )
                contextMat = np.zeros( (nCand, maxContextLen) )
                spanId = 0
                candLen = []
                contextLen = [] 
                for iPara, candPara in enumerate(candidates):
                    for iSen, candSen in enumerate(candPara):
                        context = self.TokenToId(article.paragraphs[iPara].context.sentence[iSen].token)
                        for cand in candSen:
                            candMat[spanId, :len(cand) ] = cand
                            contextMat[spanId, :len(context) ] = context
                            candLen.append(len(cand) )
                            contextLen.append(len(context) )
                            spanId += 1
                evalCandPadded[title] = candMat.astype(np.int32)
                evalContextPadded[title] = contextMat.astype(np.int32)
                evalCandLen[title] = np.array(candLen).astype(np.int32)
                evalContextLen[title] = np.array(contextLen).astype(np.int32)
                if onTrain:
                    assert evalCandPadded[title].shape[0] == len(self.trainCandData[title].candidateAnswers)
                else:
                    assert evalCandPadded[title].shape[0] == len(self.evalCandData[title].candidateAnswers)
        else:
            for title in evalCandidates.keys():
                article = evalOrig[title]
                candidates = evalCandidates[title]
                evalCandPadded[title] = list()
                evalContextPadded[title] = list()
                evalCandLen[title] = list()
                evalContextLen[title] = list()
                for iPara, candPara in enumerate(candidates):
                    nCand = sum( [len(candSen) for candSen in candPara] )
                    maxCandLen = max( [max( [len(cand) for cand in candSen] ) for candSen in candPara if len(candSen) != 0] )
                    maxContextLen = max( [len(sen.token) for sen in article.paragraphs[iPara].context.sentence] )
                    candMat = np.zeros( (nCand, maxCandLen) )
                    contextMat = np.zeros( (nCand, maxContextLen) )
                    spanId = 0
                    candLen = []
                    contextLen = []
                    for iSen, candSen in enumerate(candPara):
                        context = self.TokenToId(article.paragraphs[iPara].context.sentence[iSen].token)
                        for cand in candSen:
                            candMat[spanId, :len(cand) ] = cand
                            contextMat[spanId, :len(context) ] = context
                            candLen.append(len(cand) )
                            contextLen.append(len(context) )
                            spanId += 1

                    evalCandPadded[title].append(candMat.astype(np.int32) )
                    evalContextPadded[title].append(contextMat.astype(np.int32) )
                    evalCandLen[title].append(np.array(candLen).astype(np.int32) )
                    evalContextLen[title].append(np.array(contextLen).astype(np.int32) )
                # as we trim the and . from candidates, the following assertions may not apply
                # if onTrain:
                #     assert sum( [mat.shape[0] for mat in evalCandPadded[title] ] ) == len(self.trainCandData[title].candidateAnswers)
                # else:
                #     assert sum( [mat.shape[0] for mat in evalCandPadded[title] ] ) == len(self.evalCandData[title].candidateAnswers)
        if onTrain == True:
            self.trainCandInput = (evalCandPadded, evalContextPadded, evalCandLen, evalContextLen)
        else:
            self.evalCandInput = (evalCandPadded, evalContextPadded, evalCandLen, evalContextLen)


    # Count unk rate
    def GetUnkRate(self, doTrain=False):
        if doTrain:
            samples = self.trainSamples
            candInput = self.trainCandInput
            candGlobalId = self.trainCandGlobalId
            candData = self.trainCandData
            origData = self.trainOrigData
        else:
            samples = self.evalSamples
            candInput = self.evalCandInput
            candGlobalId = self.evalCandGlobalId
            candData = self.evalCandData
            origData = self.evalOrigData

        candPadded, contextPadded, candLen, contextLen = candInput
        cntUnkQ = 0
        cntUnkA = 0
        cntUnkC = 0
        cntQ = 0
        cntA = 0
        cntC = 0
        for iSample, sample in enumerate(samples):
            title = sample.title
            qaId = sample.id
            if self.articleLevel:
                nCand = len(candPadded[title] )
                qRnnInput = np.array(sample.query).reshape( (1, len(sample.query) ) )
                qSeqLen = np.array( (len(sample.query), ) )
                paRnnInput = candPadded[title]
                paSeqLen = candLen[title]
                pcRnnInput = contextPadded[title]
                pcSeqLen = contextLen[title]
            else:
                paraId = sample.pAnsParaId
                nCand = len(candPadded[title][paraId] )
                qRnnInput = np.array(sample.query).reshape( (1, len(sample.query) ) )
                qSeqLen = np.array( (len(sample.query), ) )
                paRnnInput = candPadded[title][paraId]
                paSeqLen = candLen[title][paraId]
                pcRnnInput = contextPadded[title][paraId]
                pcSeqLen = contextLen[title][paraId]

            
            cntUnkQ += np.sum(qRnnInput == self.wordToId["<unk>"] )
            cntUnkA += np.sum(paRnnInput == self.wordToId["<unk>"] )
            cntUnkC += np.sum(pcRnnInput == self.wordToId["<unk>"] )
            cntQ += np.sum(qSeqLen)
            cntA += np.sum(paSeqLen)
            cntC += np.sum(pcSeqLen)

        print "unk rate in query ", cntUnkQ , " / ", cntQ, " : ", cntUnkQ / float(cntQ)
        print "unk rate in candidates ", cntUnkA, " / ", cntA, " : ", cntUnkA / float(cntA)
        print "unk rate in context ", cntUnkC, " / ", cntC, " : ", cntUnkC / float(cntC)


    # TODO recover back to the stable version
    def Predict(self, samples, candInput, candGlobalId, candData, 
        origData, session):
        '''
        predict each sample in samples with the related data in candInput.
        @param candInput: it can be either self.trainCandInput or self.evalCandInput
        candInput is produced via self.PrepareEvalInput
        we reuse the interface of paRnnInput and pcRnnInput to get the scores
        @param candGlobalId: either self.trainCandGlobalId or evalCandGlobalId
        @param candData: either self.traincandData or self.evalcandData
        @param origData: either self.trainOrigData or self.evalOrigData
        '''
        candPadded, contextPadded, candLen, contextLen = candInput
        prediction = dict()
        topK = self.predTopK

        for iSample, sample in enumerate(samples):
            title = sample.title
            qaId = sample.id
            if self.articleLevel:
                nCand = len(candPadded[title] )
                qRnnInput = np.array(sample.query).reshape( (1, len(sample.query) ) )
                qSeqLen = np.array( (len(sample.query), ) )
                paRnnInput = candPadded[title]
                paSeqLen = candLen[title]
                pcRnnInput = contextPadded[title]
                pcSeqLen = contextLen[title]
            else:
                paraId = sample.pAnsParaId
                nCand = len(candPadded[title][paraId] )
                qRnnInput = np.array(sample.query).reshape( (1, len(sample.query) ) )
                qSeqLen = np.array( (len(sample.query), ) )
                paRnnInput = candPadded[title][paraId]
                paSeqLen = candLen[title][paraId]
                pcRnnInput = contextPadded[title][paraId]
                pcSeqLen = contextLen[title][paraId]

            batchData = {self.qRnnInputEval : qRnnInput,
                self.aRnnInputEval : paRnnInput, 
                self.cRnnInputEval : pcRnnInput,
                self.qSeqLenEval : qSeqLen,
                self.aSeqLenEval : paSeqLen,
                self.cSeqLenEval : pcSeqLen}

            # # # DEBUG
            # # print qaId, 
            # # print "pred data sum", np.sum(batchData[self.aRnnInputEval] ), \
            # #     np.sum(batchData[self.cRnnInputEval] )
            # print "test input ", paRnnInput[123, :], paSeqLen[123]
            # print "test input 2 ", pcRnnInput[123, :], pcSeqLen[123]
            # # # END of DEBUG



            batchData = self.GetPredictBatch(batchData)
            scores = session.run(self.evalScore, feed_dict=batchData)


            # # # DEBUG
            # print "\n\n\n\n test pa score ", np.argmax(scores), np.max(scores)
            # for i in range(batchData[self.aRnnInputEval].shape[0] ):
            #     print "\n test data na ", i, scores[0, i]
            #     print self.IdToWord(batchData[self.aRnnInputEval][i, :].tolist() )
            #     print self.IdToWord(batchData[self.cRnnInputEval][i, :].tolist() )
            # # raw_input("done")

            # print self.IdToWord(paRnnInput[np.argmax(scores), :].tolist() )
            # print self.IdToWord(pcRnnInput[np.argmax(scores), :].tolist() )
            # print self.IdToWord(qRnnInput[0, :].tolist() )

            # predict a topK list
            predIdSort = np.argsort(-scores[0, :] )
            prediction[qaId] = list()
            for i in range(min(topK, scores.size) ):
                predId = predIdSort[i]
                if self.articleLevel == False:
                    # from paragraph level span id to article level span id
                    globalId = [idx for idSen in candGlobalId[title][sample.pAnsParaId] for idx in idSen]
                    predId = globalId[predId]
                predInfo = candData[title].candidateAnswers[predId]
                predParaId = predInfo.paragraphIndex
                predSenId = predInfo.sentenceIndex
                predSpanStart = predInfo.spanBeginIndex
                predSpanEnd = predInfo.spanBeginIndex + predInfo.spanLength
                tokens = origData[title].paragraphs[predParaId].context.sentence[predSenId].token[predSpanStart:predSpanEnd]
                predStr = ReconstructStrFromSpan(tokens, (0, len(tokens) ) )
                prediction[qaId].append(QaPrediction(title, qaId, predStr, predParaId, predSenId, ansToken=tokens) )
        return prediction


    def PredictTrainSamples(self, session):
        samples = self.trainSamples
        candInput = self.trainCandInput
        candGlobalId = self.trainCandGlobalId
        candData = self.trainCandData
        origData = self.trainOrigData
        prediction = self.Predict(samples, candInput, candGlobalId, candData, origData, session)
        return prediction


    def PredictEvalSamples(self, session):
        samples = self.evalSamples
        candInput = self.evalCandInput
        candGlobalId = self.evalCandGlobalId
        candData = self.evalCandData
        origData = self.evalOrigData
        prediction = self.Predict(samples, candInput, candGlobalId, candData, origData, session)
        return prediction
