import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import defaultdict
import json
import random

from proto.io import ReadArticles
from utils.squad_utils import LoadCandidateData, LoadOrigData

MONTHS = set(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
def ClassifyDate(answer, qa):
    if len([token for sentence in answer.sentence for token in sentence.token]) >= 5:
        return False
    
    def IsYear(token):
        try:
            year_str = token.word
            if year_str.endswith('s'):
                year_str = year_str[:-1]
            year = int(year_str)
            if year >= 1000 and year <= 2100:
                return True
        except ValueError:
            pass
        return False

    def IsMonth(token):
        return token.lemma in MONTHS
    
    return any([IsYear(token) or IsMonth(token) or token.ner == 'DATE' for sentence in answer.sentence for token in sentence.token])

def ClassifyOtherNumeric(answer, qa):
    if len([token for sentence in answer.sentence for token in sentence.token]) >= 5:
        return False

    return any([token.pos == 'CD' for sentence in answer.sentence for token in sentence.token])
    

VBP = set( ["VBG", "VBD", "VBN", "VBP", "VBZ", "VP"] )
ADJP = set( ["JJ", "JJR", "JJS", "ADJP"] )
ADVP = set( ["RB", "RBR", "RBS", "ADVP"] )
# for common noun
NN = set( ["NN", "NNS", "NP", "NP-TMP", "PP"] )
NNP = set( ["NP", "NP-TMP", "NNP", "NNPS", "PP"] )
CLAUSE = set( ["S", "SBAR", "SBARQ", "SINV", "SQ"] )


def ClassifyNoun(answer, qa, tag):
    return all(['NN' in token.pos and 'NNP' not in token.pos for sentence in answer.sentence for token in sentence.token])

NNPS = set(['NNP', 'NNPS'])
NNP_POS = set(['NNP', 'NNPS', 'CC', '-LRB-', '-RRB-', ',', 'DT', 'IN', '.', 'LS', 'JJ', 'PRP', 'POS'])
NNP_NERS = set(['LOCATION', 'PERSON', 'ORGANIZATION'])
def ClassifyProperNoun(answer, qa, tag):
    if all([token.ner in NNP_NERS for sentence in answer.sentence for token in sentence.token]):
        return True
    if not any([token.pos in NNPS for sentence in answer.sentence for token in sentence.token]):
        return False
    if all([token.pos in NNP_POS or token.ner in NNP_NERS for sentence in answer.sentence for token in sentence.token]):
        return True
    if tag in NNP and not all( [token.word.islower() for token in answer.sentence[0].token] ):
        return True
    return False

def ClassifyPerson(answer):
    if all([token.ner == 'PERSON' or token.pos == 'CC' for sentence in answer.sentence for token in sentence.token]):
        return True
    num_person = sum([token.ner == 'PERSON' for sentence in answer.sentence for token in sentence.token])
    num_tokens = len([token for sentence in answer.sentence for token in sentence.token])
    return 1.0 * num_person / num_tokens >= 0.5

def ClassifyLocation(answer):
    num_location = sum([token.ner == 'LOCATION' for sentence in answer.sentence for token in sentence.token])
    num_tokens = len([token for sentence in answer.sentence for token in sentence.token])
    return 1.0 * num_location / num_tokens >= 0.5


def Classify(answer, qa, tag):
    if ClassifyDate(answer, qa):
        return 'Date'
      
    if ClassifyOtherNumeric(answer, qa):
        return 'Other Numeric'
   
    if ClassifyProperNoun(answer, qa, tag):
        if ClassifyPerson(answer):
            return 'Person'
        if ClassifyLocation(answer):
            return 'Location'
        if tag in NNP:
            return 'Other Entity'

    if tag in NN:
        return 'Common Noun Phrase'

    if tag in VBP:
        return 'Verb Phrase'

    if tag in ADJP:
        return 'Adjective Phrase'

    if tag in CLAUSE:
        return "Clause"

    return 'Others'


def GetConstituentType(parseTree, targetSpan, beginPos):
    spanBeginList = list()
    spanEndList = list()
    tag = None
    wordList = list()
    spanBegin = beginPos
    spanEnd = beginPos
    if len(parseTree.child) == 0:
        # is safe to use tag = None for leaf as the tag are never leaf      
        return spanBegin, spanBegin + 1, tag, [parseTree.value, ]

    for child in parseTree.child:
        spanBegin, spanEnd, tagTmp, word = GetConstituentType(child, targetSpan, spanEnd)
        spanBeginList.append(spanBegin)
        spanEndList.append(spanEnd)
        wordList += word

        # preserve the highest level of tag
        if tagTmp != None:
            tag = tagTmp

    spanBegin = min(spanBeginList)
    spanEnd = max(spanEndList)
    if spanBegin == targetSpan.spanBeginIndex \
        and spanEnd == targetSpan.spanBeginIndex + targetSpan.spanLength:
        tag = parseTree.value

    return min(spanBeginList), max(spanEndList), tag, wordList



if __name__ == '__main__':
    # new version
    candFile = "./dataset/proto/dev-candidatesal.proto"
    origFile = "./dataset/proto/dev-annotated.proto"

    origData = LoadOrigData(origFile)
    candData = LoadCandidateData(candFile)

    # extract answer path
    PosDict = defaultdict(str)
    for title in candData.keys():
        candidates = candData[title]
        candAnsList = candidates.candidateAnswers
        article = origData[title]
        for qa in candidates.questions:
            span = candAnsList[qa.correctAnswerIndex]
            paraId = span.paragraphIndex
            senId = span.sentenceIndex
            sentence = article.paragraphs[paraId].context.sentence[senId]
            spanBegin, spanEnd, tag, word = GetConstituentType(sentence.parseTree, span, 0)
            
            assert spanBegin == 0
            assert spanEnd == len(sentence.token)
            assert tag.isupper()
            PosDict[qa.id] = tag

    # test = sorted( [PosDict[key] for key in PosDict.keys() ] )
    # print list(set(test) )
    # raw_input("done")



    articles = ReadArticles(origFile)
     
    # Shuffle for more randomness of examples.
    questions = []
    for article in articles:
        for paragraph in article.paragraphs:
            for qa in paragraph.qas:
                if qa.id in PosDict.keys():
                    questions.append((article, paragraph, qa, PosDict[qa.id] ) )
    random.shuffle(questions)

    tag_counts = defaultdict(int)
    tag_examples = defaultdict(list)
    tags = {}
    for article, paragraph, qa, tag in questions:
        tag = Classify(qa.answers[0], qa, tag)
        tag_counts[tag] += 1
        tag_examples[tag].append(qa)
        tags[qa.id] = tag
    #     if tag == "Others":
    #         print PosDict[qa.id]
    #         print [token.word for token in qa.answers[0].sentence[0].token]
    # raw_input("next")


    total_count = sum(tag_counts.values())    
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print tag, round(100.0 * count / total_count, 1)
        examples = tag_examples[tag]
        # for i in xrange(min(100, len(examples))):
        #     print ' ', examples[i].answers[0].text, '  ', ' '.join([token.ner for token in examples[i].answers[0].sentence[0].token]), '  ', ' '.join([token.pos for token in examples[i].answers[0].sentence[0].token]), examples[i].question.text
        # print tag



    with open('./dev-answertypetags.json', 'w') as fileobj:
        fileobj.write(json.dumps(tags))
