import numpy as np
from utils.squad_utils import LoadJsonData
from utils.evaluator import QaEvaluator
from learning_baseline.agent import QaData, QaPrediction

def GetQaSamplesFromJson(data):
    samples = list()
    for article in data:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                qaId = qas["id"]
                ansStr = qas["answers"][0]["text"].lower().strip()
                ansStr = CleanAnswer(ansStr)
                # tokens = ansStr.split(" ")
                # if tokens[0] == "the":
                #     tokens = tokens[1:]
                # if tokens[-1] == ".":
                #     tokens = tokens[:-1]
                # ansStr = " ".join(tokens).strip()
                queryStr = qas["question"].lower().strip()
                sample = QaData(title, qaId)
                sample.ansStr = ansStr
                sample.queryStr = queryStr
                samples.append(sample)
    return samples


def GetQaPredFromJson(data, humanId):
    '''
    currently human id ranges from 1 to 2
    0 th answer is the orginal ground truth
    '''
    predictions = dict()
    for article in data:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                qaId = qas["id"]

                print len(qas["answers"] )


                ansStr = qas["answers"][humanId]["text"].lower().strip()
                ansStr = CleanAnswer(ansStr)
                # tokens = ansStr.split(" ")
                # if tokens[0] == "the":
                #     tokens = tokens[1:]
                # if tokens[-1] == ".":
                #     tokens = tokens[:-1]
                # ansStr = " ".join(tokens).strip()
                queryStr = qas["question"].lower().strip()
                pred = QaPrediction(title, qaId, ansStr, queryStr = queryStr)
                pred.queryStr = queryStr
                predictions[qaId] = (pred, )
    return predictions


IGNORED_WORDS = set(['the', 'a', 'an', 'in', 'to', 'over', 'by', 'between', 'at', 'after', 'from', 'as', 'for', 'around', 'about', 'on', 'since', 'through', 'with', 'within', 'if', 'of', 'before', 'during', 'near', 'under', 'although', 'because', 'out', 'above', 'into', 'towards', 'that', 'atop', 'besides', 'via', 'until', 'without'])

def CleanAnswer(answer):
    answer = answer.lower().strip()
    # answer = answer.replace('\xc2\xa0', ' ')
    while len(answer) > 1 and answer[0] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
        answer = answer[1:]
    while len(answer) > 1 and answer[-1] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
        answer = answer[:-1]

    answer_tokens = answer.split(' ')
    while len(answer_tokens) and answer_tokens[0] in IGNORED_WORDS:
        answer_tokens = answer_tokens[1:]
    
    return ' '.join(answer_tokens)




if __name__ == "__main__":
    fileName = "/Users/Jian/Data/research/squad/dataset/json/train.json"
    humanId = 1
    data = LoadJsonData(fileName)["data"]
    qaSamples = GetQaSamplesFromJson(data)

    print len(qaSamples)

    fileName = "/Users/Jian/Data/research/squad/dataset/json/dev.json"
    humanId = 1
    data = LoadJsonData(fileName)["data"]
    qaSamples = GetQaSamplesFromJson(data)

    print len(qaSamples)

    fileName = "/Users/Jian/Data/research/squad/dataset/json/test.json"
    humanId = 1
    data = LoadJsonData(fileName)["data"]
    qaSamples = GetQaSamplesFromJson(data)

    print len(qaSamples)


    qaPredictions = GetQaPredFromJson(data, humanId)

    metrics = ["exact-match-top-1", ]
    evaluator = QaEvaluator(metrics=metrics)
    evaluator.EvaluatePrediction(qaSamples, qaPredictions)
