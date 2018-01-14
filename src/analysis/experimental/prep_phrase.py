import sys
reload(sys)
sys.setdefaultencoding('utf-8')
 
from collections import defaultdict
import random
 
from proto.io import ReadArticles

if __name__ == '__main__':
    articles = ReadArticles('dataset/dev-annotated.proto')
 
    # Shuffle for more randomness during development.
    questions = []
    for article in articles:
        for paragraph in article.paragraphs:
            for qa in paragraph.qas:
                questions.append((article, paragraph, qa))
    random.shuffle(questions)
    
    print len(questions)
    lemmas = defaultdict(int)
    total = 0
    for article, paragraph, qa in questions:
        if qa.answer.sentence[0].token[0].pos in ['IN', 'TO']:
            lemmas[qa.answer.sentence[0].token[0].lemma] += 1
            total += 1
    print total
    for lemma in sorted(lemmas.items(), key=lambda x: -x[1]):
        print lemma[0], lemma[1]