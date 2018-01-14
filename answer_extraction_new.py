import score_calc
import util
import random
import json
import math
import operator

import sys
import os
from src.utils import squad_utils
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/proto"))
import src.proto.io

def json_to_dict(file):
	return json.load(file)


def get_constituent(sent):
	constituents = squad_utils.GetConstituentSpanBySentence(sent.parseTree)

	ans = list()
	for start, end in constituents:
		l = [t.value for t in sent.token[start:end]]
		string = ""
		for word in l:
			string += word + " "
			
		ans.append(string[:-1])
		
	return ans
	
	
#in case of emergency, look below!
#def get_candidate_likelihood(question, sentence, candidate):
#	candidate += " "
#	candidateless_sent = ""
#	for word in sentence.split(" "):
		#if word not in candidate.split(" "):
#			candidateless_sent += word + " "
#	candidateless_sent = candidateless_sent[:-1] + "."
#
#	#res = cosine_sim.get_candidate_likelihood(question, candidateless_sent)[0][1]
#	return tuple([candidate, res])

def sliding_window(question, candidates, sentence):
	sw_list = list()
	
	for i in range(0, len(candidates)):
		sw_cand = list()

		S = score_calc.get_unigrams(question.text + candidates[i][0]).keys()
		P = sentence.split(" ")
		if P[0] == "":
			P = P[1:]
		p_len = len(P)
		for k in range(len(S)):
			P.append("EOS")
		for j in range(p_len):
			for w in range(len(S)):
				s = 0
				if P[j+w] in S:
					s += math.log10(1 + 1 / score_calc.get_occurences(P[j + w], P))
				sw_cand.append(s)
		sw_list.append(tuple([sum(sw_cand), candidates[i][0]]))
	return sw_list

def sliding_window_calc(question, context):
	sents = score_calc.get_sentences(context.text)
	sw_scores = list()

	for sent, text in zip(sents, context.sentence):
		candidates = candidate_ranking(question.text, text, sent)
		temp = sliding_window(question, candidates, sent)
		for t in temp:
			sw_scores.append(t)
	
	return sorted(sw_scores, reverse=True)[:10]

	#Article objects
def candidate_ranking(question, sentence, unformated_sent):
	candidate = dict()
#use constituents or named enteties
	#candidates = get_constituent(sentence)
	candidates = util.get_ne(sentence)
	new_can = list()
	for c in candidates:
		if len(c.split(" ")) <= 30:
			#print('removed: ' , c)
#			candidates.remove(c)
			new_can.append(c)
	for can in new_can:
		score = score_calc.get_uni_bi_similarity(question, unformated_sent, can)			
		candidate[score[0]] = score[1]
	res = sorted(candidate.items(), key = operator.itemgetter(1), reverse=True)
	sub_set = list()	
	top_score = res[0]
	for i in res:
		if i[1] == top_score[1]:
			sub_set.append(i)
		else:
			break
	return sub_set
	

if __name__=="__main__":
	file_name = "dev-anotated.proto/dev-annotated.proto"
	
	fobj = open(file_name, "rb")

	article = src.proto.io.ReadArticle(fobj)
	
	
	Q = article.paragraphs[4].qas[0].question
	A = article.paragraphs[4].context

		
	swl = sliding_window_calc(Q,A)
	#print(A.text)
	print("\n", Q.text)
	for i in sorted(swl, reverse=True):
		print(i)
	

	#swl = sliding_window_likelihood(cand_r, Q, r)
	#print("swl", swl)

	#candidate_ans = list()
	#for i in range(0,len(r)):
#		a = get_constituent(A, str(i+1))
#		for t in a:
#			candidate_ans.append(t)
#	rand = (int)(random.random()*len(candidate_ans))

#	rand_ans = candidate_ans[rand]
	#print(rand_ans)


