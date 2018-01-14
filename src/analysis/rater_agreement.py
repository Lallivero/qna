from collections import Counter
import json

from evaluation.evaluator import Evaluator

if __name__ == '__main__':
    with open('dataset/dev.json') as fileobj:
        articles = json.loads(fileobj.read())['data']

    

    predictions = {}
    num_same_counts = Counter()
    for article in articles:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if len(qa['answers']) >= 3:
                    num_same_counts[3 - len(set([Evaluator.CleanAnswer(answer['text']) for answer in qa['answers'][0:3]])) + 1] += 1

                if len(qa['answers']) > 1:
                    predictions[qa['id']] = qa['answers'].pop(1)['text']

    evaluator = Evaluator(articles=articles)
    print 'Exact match:', round(evaluator.ExactMatch(predictions), 1)
    print 'F1:', round(evaluator.F1(predictions), 1)
    total_num_same_count = sum(num_same_counts.values())
    for num_same, count in sorted(num_same_counts.items()):
        print num_same, 'same:', round(100.0 * count / total_num_same_count, 1)

    with open('dataset/dev-answertypetags.json') as fileobj:
        tags = json.loads(fileobj.read())

    print len(tags), 'tagged questions'
    for tag, _ in Counter(tags.values()).most_common():
        num_correct = 0
        total_f1 = 0
        num_total = 0
        for question_id, _ in filter(lambda x: x[1] == tag, tags.items()):
            num_total += 1
            predicted_answer = predictions.get(question_id, None)
            if predicted_answer is not None:
                if evaluator.ExactMatchSingle(question_id, predicted_answer):
                    num_correct += 1
                total_f1 += evaluator.F1Single(question_id, predicted_answer)

        print str(round(100.0 * num_total / len(tags), 1)) + '%', tag, 'questions, exact match', str(round(100.0 * num_correct / num_total, 1)) + '%', ', F1', round(100.0 * total_f1 / num_total, 1)
