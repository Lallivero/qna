import numpy as np
import sys
import json
from math import floor
from collections import OrderedDict
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils.squad_utils import LoadCandidateData, LoadOrigData
from dataset_stat.stat_squad import save_pdf_from_fig
from evaluation.evaluator import Evaluator

def GetQuestionWordId(sentence):
    tokens = sentence.token
    questionWordId = None
    for i, token in enumerate(tokens):  
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            questionWordId = i
            break
    if questionWordId is None:
        return -1
    else:
        return questionWordId

def GetShortestPathDepGraph(graph, isTargetFunc, root=None):
    edgeList = list()
    for edge in graph.edge:
        # originally, it is 1 based representation,
        # we use 0 based in our implementation
        src = edge.source - 1 
        tar = edge.target - 1
        dep = edge.dep
        # if dep == "root":
        #     print src, tar
        #     continue
        # if dep in ['punct', 'cop', 'root']:
        #     continue
        if max(src, tar) > len(edgeList) - 1:
            edgeList += [list() for i in range(max(src, tar) - len(edgeList) + 1) ]
        edgeList[src].append( (tar, "-> " + dep) )
        edgeList[tar].append( (src, "<- " + dep) ) 

    assert len(graph.root) == 1
    # look for target using shortest path
    idIter = 0
    visited = [False] * len(edgeList)
    if root is None:
        roots = [root - 1 for root in graph.root]
    else:
        roots = [root, ]
    for i in range(len(roots) ):
        queue = [ (roots[i], "-> anchor", -1), ]
        while idIter < len(edgeList):
            nodeId, depType, _ = queue[idIter]
            visited[nodeId] = True
            if isTargetFunc(nodeId):
                break
            for connectId, dep in edgeList[nodeId]:
                if visited[connectId]:
                    continue
                queue.append( (connectId, dep, idIter) )
            idIter += 1

    # back trace
    nodePath = []
    depPath = []
    if idIter < len(queue):
        # the case where the target is found
        while idIter != -1:
            nodeId, dep, idIter = queue[idIter]
            depPath.append(dep)
            nodePath.append(nodeId)
        # the path is reversed so that it starts from root
        nodePath.reverse()
        depPath.reverse()
    return nodePath, depPath


def DisplaySentence(sentence):
    words = [token.word for token in sentence.token]
    print " ".join(words)
    print sentence.basicDependencies.root[0], words[sentence.basicDependencies.root[0] - 1]


def GetQPath(sentence, root=None):
    qWordId = GetQuestionWordId(sentence)
    if qWordId is None:        
        path = []
    else:
        isTarget = lambda id: id == qWordId
        path = GetShortestPathDepGraph(sentence.basicDependencies, isTarget, root)
    return path


def GetAPath(span, sentence, root=None):
    '''
    span from candidate protobuf
    '''
    isTarget = lambda idx: idx >= span.spanBeginIndex and idx < span.spanBeginIndex + span.spanLength
    path = GetShortestPathDepGraph(sentence.basicDependencies, isTarget, root)
    return path


#  to get path starting anchor
def GetCorrespondence(qSen, aSen, stopWords):
    '''
    make sure stopWords is a set for fast checking
    here we remove duplicated lemma
    '''
    q = qSen.token
    a = aSen.token
    lemmaCorrList = []
    coList = []
    for i in range(len(q) ):
        for j in range(len(a) ):
            if q[i].lemma.lower() == a[j].lemma.lower() \
                and q[i].lemma.lower() not in stopWords \
                and q[i].lemma.lower() not in lemmaCorrList:
                    coList.append( (i, j) )
                    lemmaCorrList.append(q[i].lemma.lower() )
    # coList = [ (i, j) for i in range(len(q) ) for j in range(len(a) ) if q[i].lemma.lower() == a[j].lemma.lower() and q[i].lemma.lower() not in stopWords]
    return coList


def GetRoot(qSen, aSen):
    # switch to 0 based node indexing
    return [ (qSen.basicDependencies.root[0] - 1, aSen.basicDependencies.root[0] - 1), ]


def GetQAPathPairs(origData, candData, anchorFunc=None):
    # extract question path
    qSenDict = dict()
    ansSenDict = dict()
    for title in origData.keys():
        article = origData[title]
        for iPara, paragraph in enumerate(article.paragraphs):
            for qa in paragraph.qas:
                qSenDict[qa.id] = qa.question.sentence[0]
                ansSenDict[qa.id] = qa.answers[0].sentence[0]

    # extract answer path
    aSenDict = dict()
    aSpanDict = dict()
    for title in candData.keys():
        candidates = candData[title]
        candAnsList = candidates.candidateAnswers
        article = origData[title]
        for qa in candidates.questions:
            span = candAnsList[qa.correctAnswerIndex]
            paraId = span.paragraphIndex
            senId = span.sentenceIndex
            sentence = article.paragraphs[paraId].context.sentence[senId]
            aSenDict[qa.id] = sentence
            aSpanDict[qa.id] = span

    # eliminate answers that is not exact constituents
    nQaBeforeRm = len(qSenDict.keys() )
    aSpanDictKey = set(aSpanDict.keys() )
    aSenDictKey = set(aSenDict.keys() )
    for qaId in ansSenDict.keys():
        if qaId not in aSpanDictKey \
            or qaId not in aSenDictKey:
            del qSenDict[qaId]
            del ansSenDict[qaId]
            continue
        exactAns = [token.word.lower() for token in ansSenDict[qaId].token]
        start = aSpanDict[qaId].spanBeginIndex
        end = start + aSpanDict[qaId].spanLength
        spanAns = [token.word.lower() for token in aSenDict[qaId].token[start:end] ]
        if exactAns != spanAns:
            del qSenDict[qaId]
            del ansSenDict[qaId]
            del aSenDict[qaId]
            del aSpanDict[qaId]
    nQaAfterRm = len(qSenDict.keys() )
    print "Got ", nQaAfterRm / float(nQaBeforeRm), " exactly covered pair!"

    # # DEBUG
    # fp = open("./output/dep_analysis/multi_comparison.txt", "w")



    aSenDictKeys = set(aSenDict.keys() )
    cntOverlapping = 0
    qPathDict = dict()
    aPathDict = dict()
    for qaId in qSenDict.keys():
        if qaId not in aSenDictKeys:
            continue
        # assert qaId in aSenDictKeys
        qSen = qSenDict[qaId]
        aSen = aSenDict[qaId]
        coList = anchorFunc(qSen, aSen)
        anchorFound = False
        if len(coList) != 0:
            # get path in query
            qPathDict[qaId] = list()
            aPathDict[qaId] = list()
            for anchorQ, anchorA in coList:

                # # TODO recover to single version
                # for iWord in range(aSpanDict[qaId].spanLength):
                sentence = qSenDict[qaId]
                qNodePath, qDepPath = GetQPath(sentence, anchorQ) 

                # get path in answer
                span = aSpanDict[qaId]

                # # TODO recover the single version
                # span.spanBeginIndex = span.spanBeginIndex + iWord
                # span.spanLength = 1

                sentence = aSenDict[qaId]
                aNodePath, aDepPath = GetAPath(span, sentence, anchorA) 
                
                # if len(qDepPath) <= 4 and len(aDepPath) <= 4:
                if len(aNodePath) > 0:
                    qPathDict[qaId].append(dict() )
                    qPathDict[qaId][-1]["depPath"] = qDepPath
                    qPathDict[qaId][-1]["nodePath"] = qNodePath
                    aPathDict[qaId].append(dict() )
                    aPathDict[qaId][-1]["nodePath"] = aNodePath
                    aPathDict[qaId][-1]["depPath"] = aDepPath
                    anchorFound = True


                #     # TODO recover the single version
                #     qPathVisual, qSenFull = GetPathFullVisual(qSen.token, qDepPath, qNodePath)
                #     aPathVisual, aSenFull = GetPathFullVisual(aSen.token, aDepPath, aNodePath)
                #     # fp.write("q: " + qSenFull)
                #     # fp.write("q: " + qSenFull)
                #     fp.write("q: " + qPathVisual.encode("utf8") + "\n")
                #     fp.write("a: " + aSenFull.encode("utf8") + "\n")
                #     fp.write("a: " + aPathVisual.encode("utf8") + "\n")
                #     fp.write("\n")

                # fp.write("\n\n")  


        if anchorFound:
            cntOverlapping += 1
        # else:
        #     print "Correspondence can not be found!"
    # print "overlap rate ", cntOverlapping / float(len(qSenDict.keys() ) )
    # ansSenDict contains the answer constituent span (may not be the exact correct answer)
    return qPathDict, aPathDict, qSenDict, aSenDict, ansSenDict


def GetPathFullVisual(tokens, depPath, nodePath):
    visual = [" (" + depPath[i] + ") " + tokens[nodePath[i] ].word for i in range(len(nodePath) ) ]
    visual = " ".join(visual)
    senFull = " ".join( [token.word for token in tokens] )
    return visual, senFull

def CompressPath(pathDict):
    '''
    help to merge some of the path by removing or replace components
    '''
    for key in pathDict.keys():
        for i in range(len(pathDict[key] ) ):
            pathDict[key][i]["depPathFull"] = pathDict[key][i]["depPath"]
            pathDict[key][i]["nodePathFull"] = pathDict[key][i]["nodePath"]
            # # Get away of compound
            # pathDict[key][i]["depPath"] = [edge for node, edge \
            #     in zip(pathDict[key][i]["nodePath"], pathDict[key][i]["depPath"] ) if "compound" not in edge]
            # pathDict[key][i]["nodePath"] = [node for node, edge \
            #     in zip(pathDict[key][i]["nodePath"], pathDict[key][i]["depPath"] ) if "compound" not in edge]
            # # merge nmod and nummod
            # pathDict[key][i]["depPath"] = [edge if "nummod" not in edge else edge.replace("nummod", "nmod") for node, edge \
            #     in zip(pathDict[key][i]["nodePath"], pathDict[key][i]["depPath"] ) ]
            # merge terms with subcategory represented by :
            pathDict[key][i]["depPath"] = [edge if ":" not in edge else edge.replace(edge.split(" ")[1], edge.split(" ")[1].split(":")[0] ) for node, edge \
                in zip(pathDict[key][i]["nodePath"], pathDict[key][i]["depPath"] ) ]
            
            # # compress to last 1/2 components
            # pathDict[key][i]["depPath"] = pathDict[key][i]["depPath"][-2:]
            # pathDict[key][i]["nodePath"] = pathDict[key][i]["nodePath"][-2:]

    return pathDict


def GetSortedPathType(pathDict):
    '''
    every entry in the pathDict is a list of paths
    '''
    allPath = [" ".join(pathDict[qaId][i]["depPath"] ) for qaId in pathDict.keys() for i in range(len(pathDict[qaId] ) ) ]
    pathTypeCnt = [ (pathType, allPath.count(pathType) ) for pathType in set(allPath) ]
    pathTypeCnt.sort(key=lambda pair: pair[1], reverse=True)
    pathType = [ele[0] for ele in pathTypeCnt]
    return pathType, pathTypeCnt


def IntersectQAPath(qPathDict, aPathDict, qPathType, aPathType):
    interCnt = dict()
    sampleIdDict = dict()

    qPathType = set(qPathType)
    aPathType = set(aPathType)
    aPathDictKey = set(aPathDict.keys() )
    for key in qPathDict.keys():
        assert key in aPathDictKey
        for i in range(len(qPathDict[key] ) ):
            qPath = " ".join(qPathDict[key][i]["depPath"] )
            aPath = " ".join(aPathDict[key][i]["depPath"] )
            if qPath in qPathType:
                qT = qPath
            else:
                qT = "other"
            if aPath in aPathType:
                aT = aPath
            else:
                aT = "other"

            if (qT, aT) not in interCnt:
                interCnt[ (qT, aT) ] = 1
                sampleIdDict[ (qT, aT) ] = [ (key, i), ]
            else:
                interCnt[ (qT, aT) ] += 1
                sampleIdDict[ (qT, aT) ].append( (key, i) )
    return interCnt, sampleIdDict


def PrintIntersectTypeWithExample(interCnt, sampleIdDict, 
    qSampleByQaId, aSampleByQaId, nSampleToDisplay, fileName, 
    ansSenDict=None, topKQ=10, topKA=5):
    nSample = 0
    qCnt = dict()
    for keyQ, keyA in interCnt.keys():
        if keyQ not in qCnt.keys():
            qCnt[keyQ] = interCnt[ (keyQ, keyA) ]
        else:
            qCnt[keyQ] += interCnt[ (keyQ, keyA) ]
        nSample += interCnt[ (keyQ, keyA) ]
    cntByQ = [ (keyQ, qCnt[keyQ] ) for keyQ in qCnt.keys() ]
    cntByQ.sort(key=lambda x:x[1], reverse=True)
    testCnt = 0
    with open(fileName, "w") as fp:
        cntByQA = dict()
        for keyQ, _ in cntByQ:
            cntByQA[keyQ] = list()
        for keyQ, keyA in interCnt.keys():
            cntByQA[keyQ].append( (keyA, interCnt[ (keyQ, keyA) ] ) )
        for iQ, (keyQ, cnt) in enumerate(cntByQ):
            if iQ >= topKQ:
                break
            cntByQA[keyQ].sort(key=lambda x:x[1], reverse=True)
            fp.write("\n\n\n" + keyQ + " " + str(cnt / float(nSample) * 100.0) + "\n")
            for i in range(min(len(cntByQA[keyQ] ), topKA) ):
                keyA, cnt = cntByQA[keyQ][i]
                fp.write("\t" + keyA + " " + str(cnt / float(nSample) * 100.0) + "\n")

                nDisplaySample = min(nSampleToDisplay, len(sampleIdDict[ (keyQ, keyA) ] ) )
                idx = np.random.choice(np.array(range(len(sampleIdDict[ (keyQ, keyA) ] ) ) ), 
                    nDisplaySample, replace=False).tolist()
                sampleId = [sampleIdDict[ (keyQ, keyA) ][corrId] for corrId in idx]

                for idx in sampleId:
                    qaId = idx[0]
                    corrId = idx[1]
                    fp.write("\t\t" + qSampleByQaId[qaId][corrId]["sentence"].encode("utf8") + "\n")
                    fp.write("\t\t" + qSampleByQaId[qaId][corrId]["visual"].encode("utf8") + "\n")
                    fp.write("\t\t" + aSampleByQaId[qaId][corrId]["sentence"].encode("utf8") + "\n")
                    fp.write("\t\t" + aSampleByQaId[qaId][corrId]["visual"].encode("utf8") + "\n")
                    tokens = [token.word for token in ansSenDict[qaId].token]
                    tokens = " ".join(tokens)
                    fp.write("\t\t" + tokens.encode("utf8") + "\n\n")


def GetSamplesForPathTypes(pathTypeList, pathDict, senDict):
    '''
    the pathTypeList should contains all the path types in pathDict
    '''
    sampleDict = dict()
    sampleDictByQaId = dict()
    for pathType in pathTypeList:
        sampleDict[pathType] = list()
    senDictKey = set(senDict.keys() )
    for qaId in pathDict.keys():
        assert qaId in senDictKey
        tokens = senDict[qaId].token
        tokens = [token.word for token in tokens]
        for i in range(len(pathDict[qaId] ) ):
            nodePath = pathDict[qaId][i]["nodePath"]
            nodePathFull = pathDict[qaId][i]["nodePathFull"]
            depPath = pathDict[qaId][i]["depPath"]
            depPathFull = pathDict[qaId][i]["depPathFull"]
            assert len(nodePath) == len(depPath)
            pathType = " ".join(depPath)
            sentence = " ".join(tokens)
            visualPath = [" (" + depPathFull[i] + ") " + tokens[nodePathFull[i] ] for i in range(len(nodePathFull) ) ]
            visualPath = " ".join(visualPath)
            sample = {"id": qaId, "pathType": pathType, "sentence": sentence, "visual": visualPath}
            sampleDict[pathType].append(sample)
            if qaId not in sampleDictByQaId.keys():
                sampleDictByQaId[qaId] = [sample, ]
            else:
                sampleDictByQaId[qaId].append(sample)
    return sampleDict, sampleDictByQaId


def GetEditDistance(pathA, pathB, cost=None):
    '''
    0 for exact matching
    1 for deleting from B to match A
    2 for inserting to B to match A
    3 for substituting to match A
    '''
    m = len(pathA)
    n = len(pathB)
    distance = np.zeros( (m + 1, n + 1) )
    operation = np.zeros( (m + 1, n + 1) )
    distance[0, :] = np.array(range(n + 1) )
    distance[:, 0] = np.array(range(m + 1) )
    nodeDeleteOp = list()
    nodeInsertOp = list()
    nodeSubOp = list()
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cost == None:
                costInsert = distance[i - 1, j] + 1
                costDelete = distance[i, j - 1] + 1
                if pathA[i - 1] == pathB[j - 1]:
                    costSub = distance[i - 1, j - 1]
                else:
                    costSub = distance[i - 1, j - 1] + 2
            else:
                costInsert = distance[i - 1, j] + cost["insert"].setdefault(pathA[i - 1], sys.float_info.max / 1000.0)
                costDelete = distance[i, j - 1] + cost["delete"].setdefault(pathB[j - 1], sys.float_info.max / 1000.0)
                if pathA[i - 1] == pathB[j - 1]:
                    costSub = distance[i - 1, j - 1]
                else:
                    costSub = distance[i - 1, j - 1] + cost["replace"].setdefault( (pathA[i - 1], pathB[j - 1] ), sys.float_info.max / 1000.0)

                # costInsert = distance[i - 1, j] + cost["insert"][pathA[i - 1] ]
                # costDelete = distance[i, j - 1] + cost["delete"][pathB[j - 1] ]
                # if pathA[i - 1] == pathB[j - 1]:
                #     costSub = distance[i - 1, j - 1]
                # else:
                #     costSub = distance[i - 1, j - 1] + cost["replace"][ (pathA[i - 1], pathB[j - 1] ) ]
            minCost = min(costInsert, costDelete, costSub)
            distance[i, j] = minCost

            if costSub == minCost and pathA[i - 1] != pathB[j - 1]:
                operation[i, j] = 3
            elif costInsert == minCost:
                operation[i, j] = 2
            elif costDelete == minCost:
                operation[i, j] = 1

            # if costDelete == minCost:
            #     operation[i, j] = 1
            # elif costInsert == minCost:
            #     operation[i, j] = 2
            # elif pathA[i - 1] != pathB[j - 1]:
            #     operation[i, j] = 3
            # else:
            #     operation[i, j] = 0
    # backtrace
    curI = m
    curJ = n
    while (curI > 0 and curJ > 0):
        if operation[curI, curJ] == 1:
            nodeDeleteOp.append(pathB[curJ - 1] )
            curJ = curJ - 1
        elif operation[curI, curJ] == 2:
            nodeInsertOp.append(pathA[curI - 1] )
            curI = curI - 1
        else:
            if pathA[curI - 1] != pathB[curJ - 1]:
                nodeSubOp.append( (pathA[curI - 1], pathB[curJ - 1] ) )
            curI -= 1
            curJ -= 1
    # return distance[m, n], nodeDeleteOp, nodeInsertOp, nodeSubOp
    return min(distance[m, n], 8), nodeDeleteOp, nodeInsertOp, nodeSubOp



def PrintRandomSample(pathType, sampleDict, nSampleToDisplay, resultFile, ansSenDict=None, qSenDict=None, topK=1):
    '''
    pathType is assumed to be orderd where the first type covers the largest portion
    '''
    if ansSenDict is not None:
        ansSenDictKey = set(ansSenDict.keys() )
    if qSenDict is not None:
        qSenDictKey = set(qSenDict.keys() )

    nSample = 0
    for key in sampleDict.keys():
        nSample += len(sampleDict[key] )

    with open(resultFile, "w") as fp:
        for iType, pType in enumerate(pathType):
            if iType >= topK:
                break
            samples = sampleDict[pType]
            fp.write(pType + "  " + str(len(samples) / float(nSample) * 100.0) + "\n")
            nDisplaySample = min(nSampleToDisplay, len(samples) )
            sampleId = np.random.choice(np.array(range(len(samples) ) ), 
                nDisplaySample, replace=False).tolist()
            # cntSample += len(samples)
            for idx in sampleId:
                if qSenDict is not None:
                    if samples[idx]["id"] not in qSenDictKey:
                        print "ans not found for the id!"
                        continue
                    qaId = samples[idx]["id"]
                    tokens = [token.word for token in qSenDict[qaId].token]
                    tokens = " ".join(tokens)
                    fp.write("\t" + tokens.encode("utf8") + "\n\n")
                fp.write("\t" + samples[idx]["sentence"].encode("utf8") + "\n")
                fp.write("\t" + samples[idx]["visual"].encode("utf8") + "\n")
                if ansSenDict is not None:
                    if samples[idx]["id"] not in ansSenDictKey:
                        print "ans not found for the id!"
                        continue
                    qaId = samples[idx]["id"]
                    tokens = [token.word for token in ansSenDict[qaId].token]
                    tokens = " ".join(tokens)
                    fp.write("\t" + tokens.encode("utf8") + "\n\n")


def VisualConfusionMat(qPathType, aPathType, cntDict, topK=200):
    cntDictKeys = set(cntDict.keys() )
    confMat = np.zeros( (len(qPathType), len(aPathType) ) )
    for iQT, qT in enumerate(qPathType):
        for iAT, aT in enumerate(aPathType):
            if (qT, aT) not in cntDictKeys:
                continue
            confMat[iQT][iAT] = cntDict[ (qT, aT) ]
    # confMat = confMat / np.sum(confMat, axis=1)

    heatmap = plt.pcolor(confMat[:topK, :topK] )
    plt.show()


def PrintSampleToJson(qPathDict, aPathDict, qSenDict, aSenDict, aSenFullDict, cntDict, fileName, thresh=30):
    '''
    thresh: filter away those path type pairs whose occurrance is smaller than thresh
    '''
    samples = []
    for qaId in qPathDict.keys():
        for i in range(len(qPathDict[qaId] ) ):
            qPath = " ".join(qPathDict[qaId][i]["depPath"] )
            aPath = " ".join(aPathDict[qaId][i]["depPath"] )
            if cntDict[ (qPath, aPath) ] < thresh:
                continue
            qSen = " ".join( [token.word for token in qSenDict[qaId].token] )
            aSen = " ".join( [token.word for token in aSenDict[qaId].token] )
            ansSen = " ".join( [token.word for token in ansSenDict[qaId].token] )
            qDepPathFull = qPathDict[qaId][i]["depPathFull"]
            qNodePathFull = qPathDict[qaId][i]["nodePathFull"]
            aDepPathFull = aPathDict[qaId][i]["depPathFull"]
            aNodePathFull = aPathDict[qaId][i]["nodePathFull"]
            tokens = qSenDict[qaId].token
            qPathFullVisual = " ".join( [" (" + qDepPathFull[i] + ") " + tokens[qNodePathFull[i] ].word for i in range(len(qNodePathFull) ) ] )
            tokens = aSenDict[qaId].token
            aPathFullVisual = " ".join( [" (" + aDepPathFull[i] + ") " + tokens[aNodePathFull[i] ].word for i in range(len(aNodePathFull) ) ] )
            # sample = {"ID": qaId, "QUERY": qSen, "QPATH": qPath, "QPATHVISUAL": qPathFullVisual,\
            #     "ANSSENTENCE": aSen, "ANS": ansSen, "APATH": aPath, "APATHVISUAL": aPathFullVisual}
            sample = OrderedDict( [ ("ID", qaId), ("QUERY", qSen), ("QPATH", qPath), ("QPATHVISUAL", qPathFullVisual),\
                ("ANSSENTENCE", aSen), ("ANS", ansSen), ("APATH", aPath), ("APATHVISUAL", aPathFullVisual) ] )
            samples.append(sample)
    print "get ", len(samples)
    with open(fileName, "w") as fp: 
        print fp
        json.dump(samples, fp)


def TestEditDistance():
    pathA = "KART"
    pathB = "KARMA"
    # pathA = "INTENTION"
    dist, deleteOp, insertOp, subOp = GetEditDistance(pathA, pathB)
    print dist
    print deleteOp
    print insertOp
    print subOp


def RefineCost(deleteRate, insertRate, subRate):
    # pass
    cost = dict()
    cost["insert"] = dict()
    cost["delete"] = dict()
    cost["replace"] = dict()
    for pType in insertRate.keys():
        if pType in insertRate.keys() and insertRate[pType] != 0:
            cost["insert"][pType] = 1 / insertRate[pType]
        else:
            cost["insert"][pType] = sys.float_info.max / 1000.0
        # cost["insert"][pType] = 1 / insertRate[pType]
    for pType in deleteRate.keys():
        if pType in deleteRate.keys() and deleteRate[pType] != 0:
            cost["delete"][pType] = 1 / deleteRate[pType]
        else:
            cost["delete"][pType] = sys.float_info.max / 1000.0
    for qType, aType in subRate.keys():
        if (qType, aType) in subRate.keys() and subRate[ (qType, aType) ] != 0:
            cost["replace"][ (qType, aType) ] = 1 / subRate[ (qType, aType) ] 
        else:
            cost["replace"][ (qType, aType) ] = sys.float_info.max / 1000.0
        # cost["replace"][pType] 
    return cost


def BucktizeCost(distDict, nBucket, minVal=0, maxVal=20):
    dist = [distDict[key] for key in distDict.keys() ]
    # minVal = min(dist)
    # maxVal = max(dist)
    interval = (maxVal - minVal) / float(nBucket)

    distDictBucket = dict()
    for qaId, dist in distDict.iteritems():
        bucketId = floor(dist / interval)
        if bucketId >= nBucket:
            bucketId = nBucket - 1
        distDictBucket[qaId] = (0.5 + bucketId) * interval
    return distDictBucket


def StatEditOpOnDep(qPath, aPath, cost=None):
    deleteOpList = list()
    insertOpList = list()
    subOpList = list()
    allDepList = list()
    distList = list()
    # get qa specific edit distance (max / ave etc.)
    distDict = dict()
    exSelection = list()
    for qaId in qPath.keys():
        distByQaId = list()
        for i in range(len(qPath[qaId] ) ):
            qDepPath = qPath[qaId][i]["depPath"]
            aDepPath = aPath[qaId][i]["depPath"]
            # allDepList += qDepPath
            allDepList += aDepPath
            if len(aDepPath) == 0 or len(qDepPath) == 0:
                continue
            dist, deleteOp, insertOp, subOp = GetEditDistance(qDepPath, aDepPath, cost)
            

            # if len(deleteOp) == 1 and len(insertOp) == 1 and len(subOp) == 1:
            exSelection.append( (qaId, i, deleteOp, insertOp, subOp, dist) )
            #     print qaId, i
            #     print qDepPath
            #     print qPath[qaId][i]["nodePath"]
            #     print aDepPath
            #     print aPath[qaId][i]["nodePath"]
            #     print deleteOp, insertOp, subOp
            #     print 
            #     print 

            # print qDepPath, aDepPath, deleteOp, insertOp, subOp
            deleteOpList += deleteOp
            insertOpList += insertOp
            # subOpList += [dep for tup in subOp for dep in tup]
            subOpList += subOp
            distList.append(dist)
            distByQaId.append(dist)
        if len(distByQaId) > 0:
            distDict[qaId] = min(distByQaId)
    depType = list(set(allDepList) )
    depCnt = dict()
    deleteRate = dict()
    insertRate = dict()
    subRate = dict()
    for dType in depType:
        depCnt[dType] = allDepList.count(dType)
        deleteRate[dType] = deleteOpList.count(dType) / float(depCnt[dType] )
        insertRate[dType] = insertOpList.count(dType) / float(depCnt[dType] )
        # assert depCnt[dType] != 0
        # assert deleteRate[dType] != 0
        # assert insertRate[dType] != 0
        # subRate[dType] = subOpList.count(dType) / 
    for (qType, aType) in list(set(subOpList) ):
        subRate[ (qType, aType) ] = subOpList.count( (qType, aType) ) / float(depCnt[aType] )
        # assert subRate[ (qType, aType) ] != 0


    # # print sorted(deleteCnt.items(), key=lambda x: x[1], reverse=True)
    # # print 
    # # print 
    # # print sorted(insertCnt.items(), key=lambda x: x[1], reverse=True)
    # # print 
    # # print 
    # # print sorted(subCnt.items(), key=lambda x: x[1], reverse=True)

    # plt.hist(np.array(distList), bins=10, range=(1, 10) )

    # with open("./output/dep_analysis/dep_deletion.txt", "w") as fp:
    #     # json.dump(sorted(deleteCnt.items(), key=lambda x: x[1], reverse=True), fp)
    #     for item in sorted(deleteCnt.items(), key=lambda x: x[1], reverse=True):
    #         fp.write(str(item[0] ) + " : " + str(item[1] ) + "\n")


    # plt.show()
    return distDict, deleteRate, insertRate, subRate, exSelection


def PrintOpExamples(qSenDict, aSenDict, ansSenDict, exSelection, qPathDict, aPathDict, distDict):
    with open("./edit-dist-example.txt", "w") as fp:
        for qaId, corrId, deleteOp, insertOp, subOp, dist in exSelection:

            # if distDict[qaId] != 0:
            #     continue
            if len(deleteOp) == 1 and len(insertOp) == 1 and len(subOp) == 1:
            # if dist == 0:
                fp.write(qaId + "\n")
                tokens = qSenDict[qaId].token
                depPath = qPathDict[qaId][corrId]["depPath"]
                nodePath = qPathDict[qaId][corrId]["nodePath"]
                qPathVisual, qSenVisual = GetPathFullVisual(tokens, depPath, nodePath)

                tokens = aSenDict[qaId].token
                depPath = aPathDict[qaId][corrId]["depPath"]
                nodePath = aPathDict[qaId][corrId]["nodePath"]
                aPathVisual, aSenVisual = GetPathFullVisual(tokens, depPath, nodePath)

                fp.write(qSenVisual.encode("utf8") + "\n")
                fp.write(qPathVisual.encode("utf8") + "\n\n")
                fp.write(aSenVisual.encode("utf8") + "\n")
                fp.write(aPathVisual.encode("utf8") + "\n\n")

                fp.write(deleteOp[0] + "\n")
                fp.write(insertOp[0] + "\n")
                fp.write(subOp[0][0] + "\t" + subOp[0][1] + "\n\n\n")




if __name__ == "__main__":
    candFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-candidatesal.proto"
    origFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    stopWordFile = "/Users/Jian/Data/research/squad/src/utils/stop_word.txt"

    # candFile = "./dev-candidatesal.proto"
    # origFile = "./dev-annotated.proto"
    # stopWordFile = "./src/utils/stop_word.txt"

    anchorType = "corr"
    if "corr" in anchorType:
        with open(stopWordFile, "r") as fp:
            stopWords = set( [word.lower() for word in fp.read().splitlines() ] )
        anchorFunc = lambda qSen, aSen: GetCorrespondence(qSen, aSen, stopWords) 
    else:
        anchorFunc = GetRoot
    origData = LoadOrigData(origFile)
    candData = LoadCandidateData(candFile)
    qPathDict, aPathDict, qSenDict, aSenDict, ansSenDict = GetQAPathPairs(origData, candData, anchorFunc)


    # print aSenDict[aSenDict.keys()[0] ].parseTree
    # raw_input("test")


    qPathDict = CompressPath(qPathDict)
    aPathDict = CompressPath(aPathDict)

    editDist, deleteStat, insertStat, subStat, exSelection = \
        StatEditOpOnDep(qPathDict, aPathDict, cost=None)


    # # re-compute using refined edit distance
    # costDict = RefineCost(deleteStat, insertStat, subStat)
    # editDist, deleteStat, insertStat, subStat = StatEditOpOnDep(qPathDict, aPathDict, costDict)
    # editDist = BucktizeCost(editDist, nBucket=12, minVal=0.0, maxVal=36)



    # PrintOpExamples(qSenDict, aSenDict, ansSenDict, exSelection, qPathDict, aPathDict, editDist)

    # raw_input("done")


    editDistGroup = dict()
    for dist, qaIdByDist in groupby(sorted(editDist.iteritems(), key=lambda x: x[1] ), key=lambda x: x[1] ):
        editDistGroup[dist] = list(qaIdByDist)

    predFile = "./output/dev-predictions-it3.json"
    jsonDataFile = "./dataset/json/dev.json"
    # predFile = "./dev-predictions-it3.json"
    # jsonDataFile = "./dev.json"
    with open(predFile, "r") as fp:
        predDict = json.load(fp)

    evaluator = Evaluator(jsonDataFile)
    exactMatchRateList = list()
    F1List = list()
    for dist in sorted(editDistGroup.keys() ):
        predSubDict = dict()
        for qaId, _ in editDistGroup[dist]:
            predSubDict[qaId] = predDict[qaId]
        exactMatchRate = evaluator.ExactMatch(predSubDict)
        F1 = evaluator.F1(predSubDict)
        exactMatchRateList.append(exactMatchRate)
        F1List.append(F1)
        print "edit dist ", dist
        print "number of sample ", len(editDistGroup[dist] ) 
        print "exact match ", exactMatchRate
        print "F1 ", F1
        print 


    allDist = [dist for _, dist in editDist.iteritems() ]
    # print np.mean(np.array(allDist) ), np.std(np.array(allDist) )
    # print np.max(np.array(allDist) )

    fileName = "/Users/Jian/Data/research/squad/paper/figure/edit-dist-hist.pdf"
    fig = plt.figure(figsize=(10, 5) )
    plt.hist(np.array(allDist), bins=20, range=(-0.5, 19.5), normed=True, facecolor=[0.0, 0.5, 1.0], alpha=0.75)
    plt.xlabel("Syntactic divergence", fontsize=20)
    plt.ylabel("Percentage", fontsize=20)
    plt.xlim( (-0.5, 8.5) )
    plt.xticks(range(0, 9) )
    def toPercent(y, position):
        s = str(100.0 * y)
        return s
    formatter = FuncFormatter(toPercent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tick_params(axis="both", which="major", labelsize=18)
    save_pdf_from_fig(fig, fileName)
    plt.show()


    # fileName = "/Users/Jian/Data/research/squad/paper/figure/edit-dist-stratification-logistic.pdf"
    # fig = plt.figure(figsize=(10, 6) )
    # plt.plot(np.array(sorted(editDistGroup.keys() ) ), np.array(exactMatchRateList), \
    #     linewidth=2, label="Exact match")
    # plt.plot(np.array(sorted(editDistGroup.keys() ) ), np.array(F1List), \
    #     linewidth=2, label="F1")
    # plt.xlim( (0, 10) )
    # plt.ylim( (20, 80) )
    # plt.xlabel("Edit distance", fontsize=20)
    # plt.ylabel("Preformance (%)", fontsize=20)
    # plt.legend(loc="upper right", fontsize=20)
    # plt.tick_params(axis="both", which="major", labelsize=18)
    # # save_pdf_from_fig(fig, fileName)
    # plt.show()



    # # specify how many types to take
    # topK = 10000
    # topKQ = topK
    # topKA = 5
    # nDisplaySample = 3
    # resultFile = "./output/dep_analysis/qPath-" + anchorType + "-" + str(nDisplaySample) + "-Sample.txt"
    # qPathType, qPathTypeCnt = GetSortedPathType(qPathDict)
    # nSample = len(qPathDict.keys() )
    # qSampleDict, qSampleDictByQaId = GetSamplesForPathTypes(qPathType, qPathDict, qSenDict)
    # PrintRandomSample(qPathType, qSampleDict, nDisplaySample, 
    #     resultFile, ansSenDict, qSenDict, topK)
    

    # resultFile = "./output/dep_analysis/aPath-" + anchorType + "-" + str(nDisplaySample) + "-Sample.txt"
    # aPathType, aPathTypeCnt = GetSortedPathType(aPathDict)
    # nSample = len(aPathDict.keys() )
    # aSampleDict, aSampleDictByQaId = GetSamplesForPathTypes(aPathType, aPathDict, aSenDict)
    # PrintRandomSample(aPathType, aSampleDict, nDisplaySample, 
    #     resultFile, ansSenDict, qSenDict, topK)
    

    # # qPathType = qPathType[:topK]
    # # aPathType = aPathType[:topK]
    # fileName = "./output/dep_analysis/mixPath-" + anchorType + "-" + str(nDisplaySample) + "-Sample.txt"
    # interCnt, sampleIdDict = IntersectQAPath(qPathDict, aPathDict, qPathType, aPathType)
    # PrintIntersectTypeWithExample(interCnt, sampleIdDict, 
    #     qSampleDictByQaId, aSampleDictByQaId, nDisplaySample, fileName, ansSenDict, topKQ, topKA)

    # topK = 200
    # VisualConfusionMat(qPathType, aPathType, interCnt, topK)
    # pathJsonFile = "./output/dep_analysis/anchor-pair-path-all.json"
    # PrintSampleToJson(qPathDict, aPathDict, qSenDict, aSenDict, aSenDict, interCnt, pathJsonFile, thresh=1)


# a few directions to compress
# 
















