from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf

from proto.io import ReadArticle, ReadArticles
from proto import training_dataset_pb2


logger = logging.getLogger(__name__)


InputExample = namedtuple('InputExample',
                          ['input_indices',
                           'input_mask',
                           'label',
                           'article_title',
                           'question_ids',
                           'num_questions',
                           'candidate_answers'])
InputPlaceholders = namedtuple('InputPlaceholders',
                               ['input_indices',
                                'input_mask',
                                'label',
                                'weight_scaling_constant'])
QuestionAnnotations = namedtuple('QuestionAnnotations',
                                 ['article',
                                  'paragraph',
                                  'qa'])

def LookupFeature(featuredict, index):
    pass

class FeatureCounter(object):
    def __init__(self, num_features=None):
        self._num_features_called = False
        self._num_features = 0
        if num_features is not None:
            self._num_features = num_features

    def NumFeatures(self):
        self._num_features_called = True
        return self._num_features

    def Count(self, index):
        assert not self._num_features_called
        self._num_features = max(self._num_features, index + 1)
        return index

def ReadExamples(path, feature_counter, min_articles=None, read_titles=None):
    examples = []
    with open(path, 'rb') as fileobj:
        num_articles = 0
        while min_articles is None or num_articles < min_articles:
            article = ReadArticle(fileobj, cls=training_dataset_pb2.TrainingArticle)
            if article is None:
                break
            num_articles += 1
            
            if num_articles % 10 == 0:
                logger.info('Read %d articles', num_articles)

            if read_titles is not None:
                read_titles.add(article.title)

            for paragraph in article.paragraphs:
                max_num_indices = 0
                for question_index, question in enumerate(paragraph.questions):
                    for candidateAnswerI, candidateAnswerFeatures in enumerate(question.candidateAnswerFeatures):
                        max_num_indices = max(max_num_indices, len(candidateAnswerFeatures.indices))

                input_indices = np.zeros([len(paragraph.questions), len(paragraph.candidateAnswers), max_num_indices], dtype=np.int32)
                input_mask = np.zeros([len(paragraph.questions), len(paragraph.candidateAnswers), max_num_indices], dtype=np.int8)
                label = np.zeros([len(paragraph.questions)], dtype=np.int64)
                for question_index, question in enumerate(paragraph.questions):
                    for candidateAnswerI, candidateAnswerFeatures in enumerate(question.candidateAnswerFeatures):
                        for i, index in enumerate(candidateAnswerFeatures.indices):
                            feature_counter.Count(index)
                            input_indices[question_index][candidateAnswerI][i] = index
                            input_mask[question_index][candidateAnswerI][i] = 1

                    label[question_index] = question.correctAnswerIndex

                question_ids = [question.id for question in paragraph.questions]                
                examples.append(InputExample(input_indices, input_mask, label, article.title, question_ids, len(question_ids), paragraph.candidateAnswers))
    return examples

def GetInputPlaceholders():
    # Input of size batch size x number of candidate answers x maximum number of features.
    input_indices = tf.placeholder(tf.int32, [None, None, None])
    input_mask = tf.placeholder(tf.int8, [None, None, None])

    # Correct label.
    label = tf.placeholder(tf.int64, [None])
    
    weight_scaling_constant = tf.placeholder(tf.float32)

    return InputPlaceholders(input_indices, input_mask, label, weight_scaling_constant)

def GetFeedDict(inputs, example, weight_scaling_constant=1.0):
    return {inputs.input_indices: example.input_indices,
            inputs.input_mask: example.input_mask,
            inputs.label: example.label,
            inputs.weight_scaling_constant: weight_scaling_constant}

def ReadQuestionAnnotations(path):
    return dict([(qa.id, QuestionAnnotations(article, paragraph, qa))
                 for article in ReadArticles(path)
                 for paragraph in article.paragraphs
                 for qa in paragraph.qas])
