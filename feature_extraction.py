import sklearn.linear_model
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("."))
import json
import util
import answer_extraction_new
import score_calc
import random
import operator

import pickle

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/proto"))
import src.proto.io

#Article objects
def get_dep_paths(context, qas):
	trues = list()
	q_id = qas.id
	ans_offset = qas.answerOffsets[0]
	ans = qas.answers[0].text
	ans_n_gram = ans.split(" ")


	ans_index_sent = 0
	ans_end_index = 0

	last_char = 0
	return_list = list()
	for sent in context.sentence:
		shared_words = get_shared_words_from_tokens(sent, qas.question.text)

		source_target_pairs = list()
		if ans_offset in range(last_char, sent.token[-1].endChar):
			for i in range(len(sent.token)):
				if sent.token[i].word in ans_n_gram:
					ans_index_sent = i
					ans_end_index = i + len(ans_n_gram)#+1

			#print("Q id:",q_id,ans_offset,"found")
			deps = sent.basicDependencies.edge
			for edge in deps:
				source = edge.source
				target = edge.target
				#print(source, target)
				source_target_pairs.append(tuple([source,target]))

			#contains dicts, word : pos graph to an answer word
			d_list = list()
			if len(shared_words) > 0:
				for w in shared_words:
					word_paths = ""
					const_paths = ""
					wf, word_paths  = word_path(w, sent, word_paths, sent.basicDependencies.root[0])
					cf,const_paths = const_path(sent, ans_index_sent, ans_end_index, const_paths, sent.basicDependencies.root[0])
					word_paths = word_paths +" " + const_paths
					d_list.append("dep_path = " + word_paths)
					trues.append(True)

			return_list.extend(d_list)

		last_char = sent.token[-1].endChar
#	print(d_dic_list)

	return dict(zip(return_list, trues))


def word_path(word, sentence, word_paths, propagator_index):
	edges = sentence.basicDependencies.edge
	propagator_edges = [i for i in edges if i.source is propagator_index]
	found = False
	for edge in propagator_edges:
		if (propagator_index-1) == word.tokenBeginIndex:
			found = True
			word_paths = word_paths + word.pos +" "+ edge.dep
			return found, word_paths
		else:
			found, word_paths = word_path(word, sentence, word_paths, edge.target)
			if found:
				word_paths= word_paths + " " + sentence.token[propagator_index-1].pos +" "+ edge.dep
	return found, word_paths
'''
	edges = sentence.basicDependencies.edge
	prop_edges = list()
	for edge in edges:
		if edge.source-1 == propagator_index:
			prop_edges.append(word_path(word, sentence, start_index, end_index, word_dict, edge.target-1))
		elif sentence.token[edge.source-1] == word:
			return [tuple([sentence.token[propagator_index].pos, sentence.token[propagator_index].deprel, word.pos])]
	return prop_edges
'''
def const_path(sentence, start_index, end_index, const_paths, propagator_index):
	edges = sentence.basicDependencies.edge
	propagator_edges = [i for i in edges if i.source is propagator_index]
	found = False
	for edge in propagator_edges:
		if propagator_index in range(start_index, end_index):
			found = True
			const_paths= const_paths + (sentence.token[propagator_index-1].pos + " " + edge.dep)
			return found, const_paths
		else:
			found, const_paths = const_path(sentence, start_index, end_index, const_paths, edge.target)
			if found:
				const_paths = const_paths + (" " +sentence.token[propagator_index-1].pos +" "+ edge.dep)
	return found, const_paths

def get_matching_word_frequencies(context, qas):
	q = qas.question.text
	flag = False
	ans_offset = qas.answerOffsets[0]
	ans = qas.answers[0].text
	ans_start, ans_end = 0, 0
	ans_sent_offset = 0
	ans_n_gram = ans.split(" ")
	relevant_sentence = 0
	last_char = 0

	for sent in context.sentence:
		if ans_offset in range(last_char, sent.token[-1].endChar):
			for i in range(len(sent.token)):
				if sent.token[i].word in ans_n_gram:
					ans_start = sent.token[i].beginChar
					ans_end = sent.token[i].beginChar + len(ans)

					try:
						relevant_sentence = score_calc.get_sentences(context.text)[sent.sentenceIndex]
					except:
						print(len(score_calc.get_sentences(context.text)), sent.sentenceIndex)
						print(context.text)
					ans_sent_offset = sent.token[0].beginChar
					flag = True
					break
			if flag:
				break

		last_char = sent.token[-1].endChar

	feature_names = ['left_span', 'span', 'right_span', 'sentence']
	tfidfs = list()
	tfidf_sums = list()
	if relevant_sentence is not 0:
		shared_left = get_shared_words(relevant_sentence[:ans_start], q)
		shared_span	= get_shared_words(relevant_sentence[ans_start:ans_end], q)
		shared_right = get_shared_words(relevant_sentence[ans_end:], q)
		shared_sent= get_shared_words(relevant_sentence, q)
#		print(relevant_sentence)
		for w in q.split(" "):
			if len(shared_left) is not 0:
				tfidfs.append(score_calc.get_tf_idf2(w, shared_left))
		tfidf_sums.append(sum(tfidfs))
		#print("left: ", tfidfs)
		tfidfs = list()
		for w in q.split(" "):
			if len(shared_span) is not 0:
				tfidfs.append(score_calc.get_tf_idf2(w, shared_span))
		tfidf_sums.append(sum(tfidfs))
		#print("span: ", tfidfs)
		tfidfs = list()
		for w in q.split(" "):
			if len(shared_right) is not 0:
				tfidfs.append(score_calc.get_tf_idf2(w, shared_right))
		tfidf_sums.append(sum(tfidfs))
		#print("right: ", tfidfs)
		tfidfs = list()
		for w in q.split(" "):
			if len(shared_sent) is not 0:
				tfidfs.append(score_calc.get_tf_idf2(w, shared_sent))
		tfidf_sums.append(sum(tfidfs))
		#print("sent: ", tfidfs)

	return dict(zip(feature_names, tfidf_sums)), ans



def get_shared_words(text1, text2):
	shared = list()
	for word in text1.split(" "):
		if word in text2.split(" "):
			shared.append(word)

	return shared

def get_shared_words_from_tokens(sent, text2):
	shared = list()
	for word in sent.token:
		if word.word in text2.split(" "):
			shared.append(word)

	return shared

def encode_classes(y_symbols):

    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))

    # We assign each name a number
    dict_classes = dict(enumerate(classes))

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes

def manual_extraction(feature_list, answer_list, corpus, training):
	fobj = open(corpus, "rb")
	article = src.proto.io.ReadArticle(fobj)
	all_the_features = feature_list
	all_the_answers = answer_list
	while article:
		print("Reading:", article.title)
		for paragraph in article.paragraphs:
			#print(paragraph.context.text[:40])
			context = paragraph.context
			qas_list = [paragraph.qas[i] for i in range(0,len(paragraph.qas))]
			for qas in qas_list:
				#features = dict()
				#features = get_lex_features(context, qas)
				features, ans = get_matching_word_frequencies(context, qas)
				features.update(get_dep_paths(context, qas))
				features['Q'] = qas.question.text
				#features['constituents'] = sentence_iterator(context)
				all_the_features.append(features)
				if training:
					all_the_answers.append(qas.answers[0].text)
		article = src.proto.io.ReadArticle(fobj)
	return all_the_features, all_the_answers

def manual_extraction_per_q(context, qas):
	features = dict()
	features = get_matching_word_frequencies(context, qas)[0]
	features.update(get_dep_paths(context, qas))
	#features['Q'] = qas.question.text
	constituents = list()
	for sent in context.sentence:
		constituents.extend(util.get_constituent(sent, n=30))
	return features, constituents

def sentence_iterator(context):
	the_bigger_list = list()
	for sentence in context.sentence:
		the_bigger_list.extend(util.get_constituent(sentence))
	return the_bigger_list


###############################################NYI########################################
# def get_lex_features(context, qas):
# 	q_lemmas = [i.lemma for i in qas.question.sentence[0].token]
# 	#print(qas.question.text)
# 	ans_offset = qas.answerOffsets[0]
# 	ans = qas.answers[0].text
# 	ans_start, ans_end = 0, 0
# 	ans_sent_offset = 0
# 	ans_n_gram = ans.split(" ")
# 	relevant_sentence = 0
# 	last_char = 0
# 	flag = False
#
# 	for sent in context.sentence:
# 		if ans_offset in range(last_char, sent.token[-1].endChar):
# 			for i in range(len(sent.token)):
# 				if sent.token[i].word in ans_n_gram:
# 					ans_start = i
# 					ans_end = i + len(ans_n_gram)
# 					relevant_sentence = sent
# 					ans_sent_offset = sent.token[0].beginChar
# 					flag = True
# 					break
# 			if flag:
# 				break
#
# 		last_char = sent.token[-1].endChar
#
# 	g_edges = set()
# 	try:
#
# 		dep_tree = relevant_sentence.basicDependencies
# 		h_edges = set()
#
# 		for edge in dep_tree.edge:
# 			if edge.source-1 in range(ans_start, ans_end) and edge.target-1 not in range(ans_start, ans_end):
# 				h_edges.add(tuple([edge.source-1, edge.target-1]))
# 		for t in h_edges:
# 			for edge in dep_tree.edge:
# 				if edge.source-1 == t[1]:
# 					g_edges.add(tuple([edge.source-1, edge.target-1]))
# 				if edge.target-1 == t[0]:
# 					g_edges.add(tuple([edge.source-1, edge.target-1]))
# 		g_edges.update(h_edges)
# 		#g_edges = sorted(g_edges, reverse = True)
#
# 	except:
# 		print("###ERROR###", sys.exc_info()[0])
#
# 	lemmas = set()
# 	for i in g_edges:
# 		lemmas.add(relevant_sentence.token[i[0]].lemma)
# 		lemmas.add(relevant_sentence.token[i[1]].lemma)
#
# 	data = dict()
# 	data['A'] = {'ans':ans}
#
# #	lemmas.update(q_lemmas)
# ###CARTESIAN PRODUCT#####
# 	all_lemmas = dict()
# 	for ql in q_lemmas:
# 		for l in lemmas:
# 			all_lemmas[tuple([ql,l])] = 1
#
# 		#for s in context.sentence:
# 			#for t in s.token:
# 				#temp = tuple([ql, t.lemma])
# 				#if temp not in all_lemmas:
# 					#all_lemmas[temp] = 0
#
# #####
# 	#print(all_lemmas)
# 	data['Q'] = all_lemmas
#
# 	return data

if __name__=='__main__':
	file_name = "train-anotated.proto/train-annotated.proto"
	#443 articles in train

	clf = sklearn.linear_model.LogisticRegression(penalty="l2", dual = True, solver="liblinear")

	all_the_features = list()
	all_the_answers = list()
#	all_the_features, all_the_answers = manual_extraction(all_the_features, all_the_answers, file_name, True)
#		print(all_the_features[0], all_the_answers[0])

	#PICKLING
#	pickle.dump(all_the_features, open('all_the_features.pkl', 'wb'))
#	pickle.dump(all_the_answers, open('all_the_answers.pkl', 'wb'))

	#UNPICKLING
	all_the_features = pickle.load(open('all_the_features.pkl', 'rb'))
	all_the_answers = pickle.load(open('all_the_answers.pkl', 'rb'))
	#reduce training set due to memory constraints
	fraction = 7
	some_of_the_features = all_the_features[:int(len(all_the_features)/fraction)]
	some_of_the_answers = all_the_answers[:int(len(all_the_answers)/fraction)]
#	print("feat vs ans", len(all_the_features), len(all_the_answers))

	v = DictVectorizer(sparse = True)
	#y, dict_classes, inv_dict_classes = encode_classes(all_the_answers)
	x = v.fit_transform(some_of_the_features)
#	y = v.fit_transform(all_the_answers)

	pickle.dump(v, open('dvec.pkl', 'wb'))

	#print("X vs Y", x.shape[0], y.shape[0])
	print("DONE")

#	making the model
	print("Fitting model...")

	model = clf.fit(x, some_of_the_answers)
	pickle.dump(model, open('model.pkl', 'wb'))
	print("Model saved")
	predictions = model.predict(x)
	for pred in predictions[:10]:
		print(v.inverse_transform(pred))
	#print("Test prediction:",dict_classes[model.predict(x)[0]])

#	Loading the model
#	v = pickle.load(open('dvec.pkl', 'rb'))
#	model = pickle.load(open('model.pkl', 'rb'))
#	ans = v.inverse_transform(model.predict(BLARGH))
#	print(ans)
