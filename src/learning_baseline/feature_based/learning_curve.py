import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import logging
import random

import tensorflow as tf

from evaluation.evaluator import Evaluator
from learning_baseline.feature_based.compute_metrics import ComputeAndDisplayMetrics
from learning_baseline.feature_based.input import ReadExamples, ReadQuestionAnnotations, GetInputPlaceholders, GetFeedDict, FeatureCounter
from learning_baseline.feature_based.graph import GetLogits, GetVariables


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input-train', '', '')
flags.DEFINE_string('input-train-articles', '', '')
flags.DEFINE_string('input-train-features', '', '')
flags.DEFINE_string('input-dev', '', '')
flags.DEFINE_string('input-dev-articles', '', '')
flags.DEFINE_string('input-dev-features', '', '')

flags.DEFINE_string('tsv-output', '', '')

flags.DEFINE_integer('max-train-articles', None, '')
flags.DEFINE_integer('max-dev-articles', None, '')

flags.DEFINE_integer('num-iterations', 10, '')
flags.DEFINE_float('learning-rate', 0.1, '')
flags.DEFINE_float('l2', 0.1, '')

flags.DEFINE_integer('num-points', 17, '')


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    random.seed(123)

    # Used to compute the number of weights to use.
    feature_counter = FeatureCounter()

    training_titles = set()
    training_examples = ReadExamples(FLAGS.input_train_features, feature_counter, FLAGS.max_train_articles, training_titles)
    random.shuffle(training_examples) 

    dev_titles = set()
    dev_examples = ReadExamples(FLAGS.input_dev_features, feature_counter, FLAGS.max_dev_articles, dev_titles)
    dev_question_annotations = ReadQuestionAnnotations(FLAGS.input_dev_articles)
    dev_evaluator = Evaluator(path=FLAGS.input_dev, restrict_to_titles=dev_titles)

    # Use a small set of articles for computing the metrics on the training set.
    training_metric_titles = set(random.sample(training_titles, len(dev_titles))) if len(training_titles) > len(dev_titles) else training_titles
    training_metric_examples = [example for example in training_examples if example.article_title in training_metric_titles]
    training_question_annotations = ReadQuestionAnnotations(FLAGS.input_train_articles)
    training_evaluator = Evaluator(path=FLAGS.input_train, restrict_to_titles=training_metric_titles)

    logger.info('Using %d features.', feature_counter.NumFeatures())
    logger.info('Using %d training paragraphs and %s dev paragraphs', len(training_examples), len(dev_examples))

    with open(FLAGS.tsv_output, 'w') as tsv_out:
        tsv_out.write('NumQuestions\tTrainingExactMatch\tTrainingF1\tDevExactMatch\tDevF1\n')

        inputs = GetInputPlaceholders()
        variables = GetVariables(feature_counter)
        logits = GetLogits(inputs, variables)
        
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inputs.label))
        train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_op)
        _, predict_op = tf.nn.top_k(logits, 1)
        
        scale_weights_op = variables.W.assign(variables.W * inputs.weight_scaling_constant)

        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)

            for point in xrange(1, FLAGS.num_points + 1):
                num_training_questions = 0
                for i, example in enumerate(training_examples):
                    if i % FLAGS.num_points < point:
                        num_training_questions += example.num_questions
                
                maximum_dev_exact_match = 0
                best_training_metrics = None
                best_dev_metrics = None
                for it in xrange(1, FLAGS.num_iterations + 1):
                    # Train.
                    weight_scaling_constant = 1.0
                    for i, example in enumerate(training_examples):
                        if i != 0 and i % 1000 == 0:
                            logger.info('Training example %d', i)

                        if i % FLAGS.num_points < point:
                            sess.run([train_op], feed_dict=GetFeedDict(inputs, example, weight_scaling_constant))
                            weight_scaling_constant *= 1 - FLAGS.l2 / (1.0 * len(training_examples) * point / FLAGS.num_points)
                    sess.run([scale_weights_op], feed_dict={inputs.weight_scaling_constant: weight_scaling_constant})
        
                    # Compute metrics.
                    training_metrics, _ = ComputeAndDisplayMetrics(
                        sess, inputs, loss_op, predict_op, training_metric_examples,
                        training_question_annotations, training_evaluator, 'Training')
                    dev_metrics, _ = ComputeAndDisplayMetrics(
                        sess, inputs, loss_op, predict_op, dev_examples,
                        dev_question_annotations, dev_evaluator, 'Dev')
                    if dev_metrics['DevExactMatch'] > maximum_dev_exact_match:
                        maximum_dev_exact_match = dev_metrics['DevExactMatch']
                        best_training_metrics = training_metrics
                        best_dev_metrics = dev_metrics

                tsv_out.write('%d\t%f\t%f\t%f\t%f\n' % (num_training_questions,
                                                      best_training_metrics['TrainingExactMatch'],
                                                      best_training_metrics['TrainingF1'],
                                                      best_dev_metrics['DevExactMatch'],
                                                      best_dev_metrics['DevF1']))
