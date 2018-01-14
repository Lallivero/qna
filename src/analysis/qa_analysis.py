import sys
reload(sys)
sys.setdefaultencoding('utf-8')
 
from collections import defaultdict
import random
 
from proto.io import ReadArticles


def DisplayDepTree(sentence, dep=None, root=None, spaces=0):
    if root is None:
        root = sentence.basicDependencies.root[0] - 1
    sys.stdout.write(' ' * spaces)
    if dep is not None:
        sys.stdout.write(dep + ' ')
    print sentence.token[root].word, sentence.token[root].lemma, sentence.token[root].pos
    for edge in sentence.basicDependencies.edge:
        if edge.source - 1 == root:
            DisplayDepTree(sentence, edge.dep, edge.target - 1, spaces + 2)


def ClassifyQuestion(question):
    tokens = question.sentence[0].token
    
    wh_token_index = None
    for i, token in enumerate(tokens):
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            wh_token_index = i
            break
    
    if wh_token_index is None:
        return 'Other'

    edge_types = set()
    edge_lemmas = set()
    for edge in question.sentence[0].basicDependencies.edge:
        if edge.dep in ['punct', 'cop', 'root']:
            continue
        
        if edge.source == wh_token_index + 1:
            if tokens[edge.target - 1].lemma == 'in':
                continue
            edge_types.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].lemma)
            
        if edge.target == wh_token_index + 1:
            edge_types.add(tokens[wh_token_index].lemma +  ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma +  ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].lemma)

    if 'what <- det - name' in edge_lemmas or 'what - nsubj -> name' in edge_lemmas or 'what <- dobj - call' in edge_lemmas or 'what <- nsubjpass - call' in edge_lemmas:
        return 'What name / is called?'

    if tokens[wh_token_index].lemma == 'when' or 'what <- det - year' in edge_lemmas or 'what - nsubj -> year' in edge_lemmas:
        return 'When / What year?'

    for wh_token_type in ['what', 'which']:
        for edge_type in [' <- det - ', ' - nsubj -> ', ' <- nsubj - ', ' <- dobj - ', ' - dep -> ', ' <- nmod - ']:
            for noun_type in ['NN', 'NNS', 'NNP']:
                if wh_token_type + edge_type + noun_type in edge_types:
                    return 'What / Which NN[*]?'
    
    for edge_type in [' <- dobj - ', ' <- nsubjpass - ', ' <- nsubj - ', ' - nsubj -> ', ' - dep -> ']:
        for verb_type in ['VB', 'VBN', 'VBZ', 'VBP', 'VBD', 'VBG']:
            if 'what' + edge_type + verb_type in edge_types:
                return 'What VB[*]?'

    if 'how <- advmod - many' in edge_lemmas or 'how <- advmod - much' in edge_lemmas:
        return 'How much / many?'

    if tokens[wh_token_index].lemma == 'how':
        return 'How?'
    
    if tokens[wh_token_index].lemma in ['who', 'whom', 'whose']:
        return 'Who?'
    
    if tokens[wh_token_index].lemma == 'where':
        return 'Where?'

    return 'Other'


def ClassifyAnswer(answer):
    num_tokens = sum([len(sentence.token) for sentence in answer.sentence])
    
    if num_tokens >= 10:
        return 'Long Answer'

    if num_tokens <= 4:
        for sentence in answer.sentence:
            for token in sentence.token:
                try:
                    num = int(token.word)
                    if 1200 <= num <= 2050:
                        return 'Date'
                except ValueError:
                    continue

    all_pos = [token.pos
               for token in sentence.token
               for sentence in answer.sentence]

    if num_tokens <= 4 and 'CD' in all_pos:
        return 'Other Numeric'

    if 1.0 * (all_pos.count('NNP') + all_pos.count('NNPS')) / len(all_pos) > 0.3:
        return 'Name'
 
    if ClassifyNounPhrase(all_pos):
        return 'Noun Phrase'

    if all_pos[0][0:2] == 'VB':
        return 'Verb Phrase'
    
    if ClassifyAdjectivePhrase(all_pos):
        return 'Adjective Phrase'
    
    return 'Other'


NOUN_POS = set(['NN', 'NNS'])
OTHER_POS_IN_NOUN_PHRASE = set(['NN', 'NNS', 'DT', 'JJ', 'POS', 'IN', 'CC', 'JJR', 'JJS', ',', 'RB', 'RBR', 'RBS', '``', '\'\'', 'FW', '-LRB-', '-RRB-'])
def ClassifyNounPhrase(all_pos):
    pos_set = set(all_pos)
    return bool(pos_set & NOUN_POS) and not bool(pos_set - OTHER_POS_IN_NOUN_PHRASE)


ADJECTIVE_POS = set(['JJ', 'JJR', 'JJS'])
OTHER_POS_IN_ADJECTIVE_PHRASE = set(['JJ', 'JJR', 'JJS', 'DT', 'POS', 'CC', 'RB', 'RBR', 'RBS', ',', '``', '\'\'', '-LRB-', '-RRB-'])
def ClassifyAdjectivePhrase(all_pos):
    pos_set = set(all_pos)
    return bool(pos_set & ADJECTIVE_POS) and not bool(pos_set - OTHER_POS_IN_ADJECTIVE_PHRASE)


if __name__ == '__main__':
    articles = ReadArticles('dataset/dev-annotated.proto')
 
    # Shuffle for more randomness of examples.
    questions = []
    for article in articles:
        for paragraph in article.paragraphs:
            for qa in paragraph.qas:
                questions.append((article, paragraph, qa))
    random.shuffle(questions)

    answer_counts = defaultdict(int)
    answer_question_counts = defaultdict(lambda: defaultdict(int))
    answer_question_examples = defaultdict(lambda: defaultdict(list))
    for article, paragraph, qa in questions:
        question_type = ClassifyQuestion(qa.question)
        answer_type = ClassifyAnswer(qa.answers[0])

        # Make some small corrections for broken date / numeric value detection.
        if question_type == 'When / What year?':
            answer_type = 'Date'

        if question_type == 'How much / many?':
            answer_type = 'Other Numeric'

        answer_counts[answer_type] += 1
        answer_question_counts[answer_type][question_type] += 1
        answer_question_examples[answer_type][question_type].append(qa)

    for answer_type, answer_count in sorted(answer_counts.iteritems(), key=lambda x: -x[1]):
        print answer_type + '  ' + str(round(100.0 * answer_count / len(questions), 1)) + '%'

        def DisplayStat(question_type, count, examples):
            print '  ' + question_type + '  ' + str(round(100.0 * count / len(questions), 1)) + '%'
            for i in xrange(0, 5):
                print '    ' + examples[i].question.text + '  ' + examples[i].answers[0].text
        
        num_displayed = 0
        other_sum = 0
        other_examples = []
        for question_type, count in sorted(answer_question_counts[answer_type].iteritems(), key=lambda x: -x[1]):
            examples = answer_question_examples[answer_type][question_type]
            if num_displayed < 4 and question_type != 'Other':
                DisplayStat(question_type, count, examples)
                num_displayed += 1
            else:
                other_sum += count
                other_examples.extend(examples)
        DisplayStat('Other', other_sum, other_examples)
        print
