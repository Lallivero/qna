import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import defaultdict, deque

from proto.io import ReadArticles

from proto.training_dataset_pb2 import TrainingArticle
from utils.squad_utils import ReconstructStrFromSpan

SUBJECTS = set(['nsubj', 'nsubjpass'])



articles = ReadArticles('dataset/dev-annotated.proto')
training_articles = ReadArticles('dataset/dev-candidatespl.proto', cls=TrainingArticle)

correctAnswers = {}
for article in training_articles:
    for paragraph in article.paragraphs:
        for qa in paragraph.questions:
            correctAnswers[qa.id] = paragraph.candidateAnswers[qa.correctAnswerIndex]

same_root = 0
contains_root = 0
total = 0
dep_types = defaultdict(int)
match_dep_paths = defaultdict(int)
match_dep_examples = defaultdict(list)
for article in articles:
    for paragraph in article.paragraphs:
        for qa in paragraph.qas:
            if qa.id not in correctAnswers:
                continue
            correctAnswer = correctAnswers[qa.id]
            sentence = paragraph.context.sentence[correctAnswer.sentenceIndex]
            question = qa.question.sentence[0]
            
#             Verify(sentence)
#             Verify(question)
            
            answer = qa.answer

            sentenceRoot = sentence.basicDependencies.root[0]
            questionRoot = question.basicDependencies.root[0]
            sentenceLemmas = set([token.lemma for token in sentence.token if token.pos not in ['IN', 'DT']])
            questionLemmas = dict([(question.token[i].lemma, i) for i in xrange(len(question.token)) if question.token[i].pos not in ['IN', 'DT']])
            if sentence.token[sentenceRoot - 1].lemma ==  question.token[questionRoot - 1].lemma:
                same_root += 1
            elif question.token[questionRoot - 1].lemma in sentenceLemmas:
                contains_root += 1
            else:
                pass
#                 print ReconstructStrFromSpan(sentence.token)
#                 print sentence.token[sentenceRoot - 1].lemma
#                 print ReconstructStrFromSpan(question.token)
#                 print question.token[questionRoot - 1].lemma
#                 print
#                 print    
            total += 1
            
            
            
        
                    
            
            def GetPath(root, spanBegin, spanEnd, sentence):
                if root >= spanBegin and root < spanEnd:
                    return None, None

                visited = set([root])
                q = deque([(root, '')])
                while q:
                    node, path = q.popleft()
                    if node  >= spanBegin and node < spanEnd:
                        return path, sentence.token[node]
                    for edge in sentence.basicDependencies.edge:
                        if edge.source - 1 == node and edge.target - 1 not in visited:
                            visited.add(edge.target - 1)
                            q.append((edge.target - 1, path + ' -> ' + edge.dep + ' ->'))
                        if edge.target - 1 == node and edge.source - 1 not in visited:
                            visited.add(edge.source - 1)
                            q.append((edge.source - 1, path + ' <- ' + edge.dep + ' <-'))
                return None, None


            for i in xrange(len(sentence.token)):
                if sentence.token[i].lemma in questionLemmas:
                    questionI = questionLemmas[sentence.token[i].lemma]
                    path, token = GetPath(i, correctAnswer.spanBeginIndex, correctAnswer.spanBeginIndex + correctAnswer.spanLength, sentence)
                    root_path, _ = GetPath(sentenceRoot - 1, i, i + 1, sentence)
                    question_root_path, _ = GetPath(questionRoot - 1, questionI, questionI + 1, question)
                    if path is not None:
                        output_paths = []
                        output_paths.append(sentence.token[i].pos + path + ' ' + token.pos)
                        if root_path is not None:
                            output_paths.append('S ' + sentence.token[sentenceRoot - 1].pos + root_path + ' ' + sentence.token[i].pos + path + ' ' + token.pos)
                        if question_root_path is not None:
                            output_paths.append('Q ' + question.token[questionRoot - 1].pos + question_root_path + ' ' + sentence.token[i].pos + path + ' ' + token.pos)
                        for output_path in output_paths:
                            match_dep_paths[output_path] += 1
                            match_dep_examples[output_path].append((sentence.token[sentenceRoot - 1].lemma, question.token[questionRoot - 1].lemma, sentence.token[i].lemma, token.lemma, ReconstructStrFromSpan(sentence.token), ReconstructStrFromSpan(question.token), ReconstructStrFromSpan(sentence.token, (correctAnswer.spanBeginIndex, correctAnswer.spanBeginIndex + correctAnswer.spanLength))))
                        
                
            

print same_root, contains_root, total

total_cnt = 0
for dep_path, cnt in sorted(match_dep_paths.items(), key=lambda x: x[1], reverse=True):
    total_cnt += cnt
    print dep_path, cnt, total_cnt
    examples = match_dep_examples[dep_path]
    for ex in xrange(0, min(5, len(examples))):
        sentence_root_lemma, question_root_lemma, sentence_lemma, answer_lemma, sentence, question, answer = examples[ex]
        if dep_path.startswith('S '):
            print sentence_root_lemma, '---', sentence_lemma, '---', answer_lemma, '---', sentence, '---', question, '---', answer
        elif dep_path.startswith('Q '):
            print question_root_lemma, '---', sentence_lemma, '---', answer_lemma, '---', sentence, '---', question, '---', answer
        else:
            print sentence_lemma, '---', answer_lemma, '---', sentence, '---', question, '---', answer
    print