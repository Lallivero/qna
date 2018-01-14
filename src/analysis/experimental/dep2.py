import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import defaultdict
import random

from proto.io import ReadArticles
from utils.squad_utils import ReconstructStrFromSpan


def DisplayDepTree(sentence, root=None, spaces=0):
    if root is None:
        root = sentence.basicDependencies.root[0] - 1
    sys.stdout.write(' ' * spaces)
    print sentence.token[root].word, sentence.token[root].pos
    for edge in sentence.basicDependencies.edge:
        if edge.source - 1 == root:
            DisplayDepTree(sentence, edge.target - 1, spaces + 2)

def DisplayParseTree(tree, spaces=0):
    sys.stdout.write(' ' * spaces)
    print tree.value
    for child in tree.child:
        DisplayParseTree(child, spaces + 2)

articles = ReadArticles('dataset/dev-annotated.proto')

questions = []


for article in articles:
    for paragraph in article.paragraphs:
        for qa in paragraph.qas:
            questions.append((article, paragraph, qa))

random.shuffle(questions)
            
wh_counts = defaultdict(int)
for article, paragraph, qa in questions:
    whToken = None    
    question = qa.question.sentence[0]
    
    for token in question.token:
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            whToken = token
            break
    
    if whToken is None:
        questionRoot = question.basicDependencies.root[0] - 1

    else:
        print whToken.lemma, whToken.pos
        print ReconstructStrFromSpan(question.token)
        DisplayDepTree(question)
        print
        print ReconstructStrFromSpan(qa.answer.sentence[0].token)
        for answer_token in qa.answer.sentence[0].token:
            sys.stdout.write(answer_token.pos + ' ')
        print
        last_sentence = None
        for sentence in paragraph.context.sentence:
            if sentence.characterOffsetBegin <= qa.answerOffset:
                last_sentence = sentence
        if last_sentence is not None:
            print ReconstructStrFromSpan(last_sentence.token)
            DisplayDepTree(last_sentence)
            DisplayParseTree(last_sentence.parseTree)
        print
        print
        wh_counts[(whToken.lemma, whToken.pos)] += 1  


total_count = 0
for lemma_and_pos, count in sorted(wh_counts.items(), key=lambda val: val[1], reverse=True):
    total_count += count
    print lemma_and_pos[0], lemma_and_pos[1], count, 1.0 *  total_count / len(questions)




# 'did', 'was', 'is', 'has', 'do', 'does', 'can', 'are', 'have'


