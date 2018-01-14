import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import defaultdict
import random

from proto.io import ReadArticles
from utils.squad_utils import ReconstructStrFromSpan

articles = ReadArticles('dataset/dev-annotated.proto')

questions = []


for article in articles:
    for paragraph in article.paragraphs:
        for qa in paragraph.qas:
            questions.append((article, paragraph, qa))

random.shuffle(questions)
            
pos_counts = defaultdict(int)
for article, paragraph, qa in questions:
    pos = ''
    for sentence in qa.answer.sentence:
        for token in sentence.token:
            pos += token.pos + ' '
    pos_counts[pos] += 1
    if pos not in [
'CD ',
'NNP NNP ',
'NNP ',
'NN ',
'NNS ',
'JJ ',
'JJ NN ',
'NNP NNP NNP ',
'JJ NNS ',
'NN NN ',
'DT NN ',
'CD NN ',
'CD NNS ']:
        print pos
        print qa.question.text
        print qa.answer.text
        print
        print



total_count = 0
for pos, count in sorted(pos_counts.items(), key=lambda val: val[1], reverse=True):
    total_count += count
    print pos, int(100.0 * count / len(questions)), int(100.0 *  total_count / len(questions))




# 'did', 'was', 'is', 'has', 'do', 'does', 'can', 'are', 'have'


