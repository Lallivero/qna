import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import Counter
import json

from evaluation.evaluator import Evaluator

if __name__ == '__main__':
    with open('dataset/dev-predictions-it3.json') as fileobj:
        predictions = json.loads(fileobj.read())

    with open('dataset/dev-answertypetags.json') as fileobj:
        tags = json.loads(fileobj.read())

    evaluator = Evaluator('dataset/dev.json')

    print len(tags), 'tagged questions'
    for tag, _ in Counter(tags.values()).most_common():
        num_correct = 0
        total_f1 = 0
        num_total = 0
        for question_id, _ in filter(lambda x: x[1] == tag, tags.items()):
            print tag, evaluator.GetAnswerText(question_id)
            num_total += 1
            predicted_answer = predictions.get(question_id, None)
            if predicted_answer is not None:
                if evaluator.ExactMatchSingle(question_id, predicted_answer):
                    num_correct += 1
                total_f1 += evaluator.F1Single(question_id, predicted_answer)

        print str(round(100.0 * num_total / len(tags), 1)) + '%', tag, 'questions, exact match', str(round(100.0 * num_correct / num_total, 1)) + '%', ', F1', round(100.0 * total_f1 / num_total, 1)
