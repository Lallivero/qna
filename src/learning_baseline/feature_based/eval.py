import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import random
import tensorflow as tf
from tensorflow import flags

from evaluation.evaluator import Evaluator
from learning_baseline.feature_based.compute_metrics import ComputeAndDisplayMetrics
from learning_baseline.feature_based.input import FeatureCounter, ReadExamples, GetInputPlaceholders, ReadQuestionAnnotations
from learning_baseline.feature_based.graph import GetLogits, GetVariables

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'dataset/test.json', '')
flags.DEFINE_string('input-articles', 'dataset/test-annotatedpartial.proto', '')
flags.DEFINE_string('input-features', 'dataset/test-featuresbucketized.proto', '')
flags.DEFINE_integer('num-features', 186194776, '')
flags.DEFINE_string('input-model', 'dataset/model', '')
flags.DEFINE_integer('min-articles', None, '')

if __name__ == '__main__':
    feature_counter = FeatureCounter(num_features=FLAGS.num_features)

    titles = set()
    examples = ReadExamples(FLAGS.input_features, feature_counter, FLAGS.min_articles, titles)
    random.shuffle(examples)
    question_annotations = ReadQuestionAnnotations(FLAGS.input_articles)
    evaluator = Evaluator(path=FLAGS.input, restrict_to_titles=titles)

    inputs = GetInputPlaceholders()
    variables = GetVariables(feature_counter)
    logits = GetLogits(inputs, variables)    
    _, predict_op = tf.nn.top_k(logits, 1)
                    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.input_model)

        ComputeAndDisplayMetrics(
            sess, inputs, None, predict_op, examples, question_annotations,
            evaluator, '')

        # if FLAGS.print_errors:
        #     for example in examples:
        #         prediction, weights = sess.run([predict_op, variables.W], feed_dict=GetFeedDict(inputs, example))
        #         predictions = BuildPredictions(question_annotations, example, np.squeeze(prediction, 1))
        # 
        #         for question_index, question_id in enumerate(example.question_ids):
        #             predicted_answer = predictions[question_id][0]
        #             if not evaluator.ExactMatchSingle(question_id, predicted_answer):
        #                 annotations = question_annotations[question_id]
        #                 print 'Answers for:', annotations.article.title
        #                 print 'Question:', annotations.qa.question.text
        #         
        #                 if example.label[question_index] == prediction[question_index][0]:
        #                     print '  Correct Text (not a span):', evaluator._answers[question_id][0]
        #                     print '  Predicted Span:', predicted_answer
        #                 else:
        #                     correct_features = set()
        #                     predicted_features = set()
        #                     
        #                     for i in xrange(example.input_indices.shape[0]):
        #                         if example.input_indices[i][0] == question_index:
        #                             if example.input_indices[i][1] == example.label[question_index]:
        #                                 correct_features.add(example.input_indices[i][2])
        #                             elif example.input_indices[i][1] == prediction[question_index][0]:
        #                                 predicted_features.add(example.input_indices[i][2])
        #                     
        #                     same_features = correct_features & predicted_features
        #                     correct_features -= same_features
        #                     predicted_features -= same_features
        #                     
        #                     def PrintAnswer(candidate_index, features, prefix):
        #                         span = example.candidate_answers[candidate_index]
        #                         sentence_tokens = annotations.article.paragraphs[span.paragraphIndex].context.sentence[span.sentenceIndex].token
        #                         print '  ' + prefix + ' Sentence:',  ReconstructStrFromSpan(sentence_tokens)
        #                         print '  ' + prefix + ' Span:', BuildPrediction(annotations, span)
        #         
        #                         total_weight = 0
        #                         sorted_weights = []
        #                         for feature_index in features:
        #                             total_weight += weights[feature_index]
        #                             sorted_weights.append((weights[feature_index], feature_counter.GetName(feature_index)))
        #         
        #                         print '  ' + prefix + ' Score:', total_weight
        #                         print '  ' + prefix + ' Features:'
        #                         sorted_weights.sort(reverse=True)
        #                         for weight, name in sorted_weights:
        #                             print '    ' + str(weight), name
        #         
        #                     
        #                     PrintAnswer(example.label[question_index], correct_features, 'Correct')
        #                     print
        #                     PrintAnswer(prediction[question_index][0], predicted_features, 'Predicted')
        #                     print
        #         
        #                 correct_span = example.candidate_answers[example.label[question_index]]
        #                 predicted_span = example.candidate_answers[prediction[question_index][0]]
        #                 if correct_span.paragraphIndex == predicted_span.paragraphIndex and correct_span.sentenceIndex == predicted_span.sentenceIndex:
        #                     print 'Correct Sentence!'
        #                 else:
        #                     print 'Wrong Sentence!'
        #         
        #                 print
        #                 print
        #                 print
