import argparse
import sys
import random

from proto.io import ReadArticles
from utils.squad_utils import GetContextConstituentSpan, ReconstructStrFromSpan

from proto import training_dataset_pb2

def PrintParseTree(tree, spaces=0):
    for _ in xrange(spaces):
        sys.stdout.write(' ')
    sys.stdout.write(tree.value.encode('utf-8'))
    sys.stdout.write('\n')
    for child in tree.child:
        PrintParseTree(child, spaces + 2)
    

def ListNonEmpty(msg):
    for descriptor, value in msg.ListFields():
        print descriptor.name


def GetSentence(paragraph, text):
    for sentence in paragraph.context.sentence:
        sentence_text = ReconstructStrFromSpan(sentence.token, (0, len(sentence.token)))
        if text in sentence_text:
            return sentence
    return None

def HasNounPrefix(article, text_to_span, text_tokens):
    num_answer_tokens = len(text_tokens)
    text = ReconstructStrFromSpan(text_tokens, (0, num_answer_tokens))
    
    for span_text, span_tuple in text_to_span.iteritems():
        paragraph_i, sentence_i, span = span_tuple
        span_length = span[1] - span[0]
        if span_length > num_answer_tokens and text in span_text:
            span_tokens = article.paragraphs[paragraph_i].context.sentence[sentence_i].token[span[0]:span[1]]
            last_span_tokens = span_tokens[-len(text_tokens):]
            if ReconstructStrFromSpan(last_span_tokens, (0, len(last_span_tokens))) == text:
                # Note: PRP$ gives a few percent increase.
                if all([token.pos in ['PRP$', 'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] for token in span_tokens[:-num_answer_tokens]]):
                    return True
    
    return False


def HasFollowingPreposition(article, text_to_span, text_tokens):
    num_answer_tokens = len(text_tokens)
    text = ReconstructStrFromSpan(text_tokens, (0, num_answer_tokens))
    
    for span_text, span_tuple in text_to_span.iteritems():
        paragraph_i, sentence_i, span = span_tuple
        span_length = span[1] - span[0]
        if span_length > num_answer_tokens and text in span_text:
            span_tokens = article.paragraphs[paragraph_i].context.sentence[sentence_i].token[span[0]:span[1]]
            
            if ReconstructStrFromSpan(span_tokens[0:num_answer_tokens], (0, num_answer_tokens)) == text and span_tokens[num_answer_tokens].pos in ['IN', 'TO']:
                return True
    return False


def DebugIncomplete(sentence, text):
    PUNCT = [' ', '.', ',']
    
    sentence_text = ReconstructStrFromSpan(sentence.token, (0, len(sentence.token)))
    pos = sentence_text.find(text)
    pos_length = len(text)
    pos_changed = False
    while pos - 1 >= 0 and sentence_text[pos - 1] not in PUNCT:
        pos -= 1
        pos_length += 1
        pos_changed = True
    while pos + pos_length < len(sentence_text) and sentence_text[pos + pos_length] not in PUNCT:
        pos_length += 1
        pos_changed = True

    if pos_changed and sentence_text[pos:pos + pos_length] in text_to_span:
        print text, '---', sentence_text[pos:pos + pos_length]

if __name__ == '__main__':
    random.seed(123)


    articles = ReadArticles('dataset/dev-annotated.proto')    
    
    num_answers = 0
    num_answer_span_pairs = 0
    num_multiple_sentences = 0
    num_bad_answers = 0
    num_has_dot = 0
    num_has_space_begin = 0
    num_has_space_end = 0
    num_needs_the = 0
    num_needs_a = 0
    num_needs_dollar = 0
    num_missing_quotes = 0
    num_missing_s = 0
    num_missing_from_any_sentence = 0
    num_appears_in_long_sentence = 0
    num_has_noun_prefix = 0
    num_has_following_preposition = 0
    num_broken = 0
    for article in articles:
        text_to_span = {}
        
        spans = GetContextConstituentSpan(article)
        num_spans = 0
        for paragraph_i in xrange(len(spans)):
            paragraph = article.paragraphs[paragraph_i]
            for sentence_i in xrange(len(spans[paragraph_i])):
                tokens = paragraph.context.sentence[sentence_i].token
                for span in spans[paragraph_i][sentence_i]:
                    num_spans += 1
                    text = ReconstructStrFromSpan(tokens, span)
                    text_to_span[text] = (paragraph_i, sentence_i, span)

        bad = []
        for paragraph in article.paragraphs:
            for qa in paragraph.qas:
                num_answers += 1
                num_answer_span_pairs += num_spans
                if len(qa.answer.sentence) > 1:
                    num_multiple_sentences += 1
                    continue

                text_tokens = qa.answer.sentence[0].token
                num_answer_tokens = len(text_tokens)
                text = ReconstructStrFromSpan(text_tokens, (0, num_answer_tokens))

                if text.endswith('.'):
                    num_has_dot += 1
                    text = text[:-1]
                    
                if text.startswith(' '):
                    num_has_space_begin += 1
                    text = text[1:]
                if text.endswith(' '):
                    num_has_space_end += 1
                    text = text[:-1]
                
                if text in text_to_span:
                    continue
                
                if 'the ' + text in text_to_span or 'The ' + text in text_to_span:
                    num_needs_the += 1
                    continue
                
                if 'a ' + text in text_to_span:
                    num_needs_a += 1
                    continue
        
                if '$' + text in text_to_span:
                    num_needs_dollar += 1
                    continue
                
                if '``' + text + '\'\'' in text_to_span:
                    num_missing_quotes += 1
                    continue

                if text + '\'s' in text_to_span:
                    num_missing_s += 1
                    continue

                sentence = GetSentence(paragraph, text)
                if sentence is None:
                    num_missing_from_any_sentence += 1
                    continue

                if sentence.parseTree.value == 'X':
                    num_appears_in_long_sentence += 1
                    continue
                
                if HasNounPrefix(article, text_to_span, text_tokens):
                    num_has_noun_prefix += 1
                    continue
                
                if HasFollowingPreposition(article, text_to_span, text_tokens):
                    num_has_following_preposition += 1
                    continue
                                


                bad.append((text, paragraph, sentence.parseTree))
                num_bad_answers += 1

#         print article.title
#         random.shuffle(bad)
#         for answer, paragraph, parseTree in bad:
#             print answer.encode('utf-8')
#             PrintParseTree(parseTree)
#             print
  
    print
    print 'Number of answers:', num_answers
    print 'Number broken:', num_broken
    print 'Number of answer - span pairs:', num_answer_span_pairs
    print 'Number answers with multiple sentences:', 1.0 * num_multiple_sentences / num_answers
    print 'Number of bad answers:', 1.0 * num_bad_answers / num_answers
    print 'Number has dot:', 1.0 * num_has_dot / num_answers
    print 'Number has space begin:', 1.0 * num_has_space_begin / num_answers
    print 'Number has space end:', 1.0 * num_has_space_end / num_answers
    print 'Number needs the:', 1.0 * num_needs_the / num_answers
    print 'Number needs a:', 1.0 * num_needs_a / num_answers
    print 'Number needs $:', 1.0 * num_needs_dollar / num_answers
    print 'Number needs quotes:', 1.0 * num_missing_quotes / num_answers
    print 'Number needs \'s:', 1.0 * num_missing_s / num_answers
    print 'Number missing from any sentence:', 1.0 * num_missing_from_any_sentence / num_answers
    print 'Number appears in long sentence:', 1.0 * num_appears_in_long_sentence / num_answers
    print 'Number has noun prefix:', 1.0 * num_has_noun_prefix / num_answers
    print 'Number has following preposition:', 1.0 * num_has_following_preposition / num_answers
     

