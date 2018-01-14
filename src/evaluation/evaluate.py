import json

from evaluation.evaluator import Evaluator

if __name__ == '__main__':
    with open('dataset/dev-predictions-final-it4.json', 'r') as f:
        bad_format_predictions = json.loads(f.read())
        predictions = {}
        for question_id, predictions_list in bad_format_predictions.iteritems():
            predictions[question_id] = predictions_list[0]
    
    evaluator = Evaluator('dataset/dev.json')
    print evaluator.ExactMatch(predictions)
    print evaluator.F1(predictions)
