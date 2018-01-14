import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import Counter
import json

from evaluation.evaluator import Evaluator

def ReadTags():
    text_to_id = {}
    with open('./dataset/json/dev.json') as fileobj:
        for article in json.loads(fileobj.read())['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    text_to_id[qa['question']] = qa['id']
    
    tags = {}
    with open('./dataset/tags-May-23-2016') as fileobj:
        for question in json.loads(fileobj.read()):
            tags[text_to_id[question['text']]] = [tag['text'] for tag in question['tags']]
    return tags

if __name__ == '__main__':
    tags = ReadTags()
    
    id_to_qa = {}
    with open('./dataset/json/dev.json') as fileobj:
        for article in json.loads(fileobj.read())['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    id_to_qa[qa['id']] = (qa, paragraph)
    
    
    print len(tags), 'tagged questions'
    print
    
    counts = Counter([tag for qtags in tags.values() for tag in qtags])
    
    for tag, _ in Counter([tag for qtags in tags.values() for tag in qtags]).most_common():
        if tag == 'lex':
            continue

        questions = filter(lambda x: tag in x[1], tags.items())

        num_total = len(questions)
        print str(round(100.0 * num_total / len(tags), 1)) + '%', tag, 'questions:'
        print
        only_questions = filter(lambda x: len(x[1]) == 1, questions)
        for i in xrange(min(10, len(only_questions))):
            qa, paragraph = id_to_qa[only_questions[i][0]]
            print qa['question']
            context = paragraph['context']
            context = context[0:qa['answers'][0]['answer_start']] + '>>>>>' + qa['answers'][0]['text'] + '<<<<<' + context[qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text']):]
            print context
            print
        print
        print
        raw_input("next")
        
