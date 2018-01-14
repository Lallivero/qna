import regex as re
import math
import operator

def get_candidate_likelihood(question, context):
	b = get_unigrams(question)
	cosines = dict()
	for sent in get_sentences(context.text):
		tfidf_array = dict()
		for i in get_unigrams(sent):
			val = get_tf_idf(i, context.text)
			tfidf_array[i] = val

		c = get_cosine_sim(b, tfidf_array)
		cosines[sent] = c

	cosines = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True)
	res = []
	for i in cosines:
		if i[1] > 0:
			res.append(i)

	return cosines

def get_uni_bi_similarity(question, sent, candidate):
	q_bi = get_bigrams(question.split(" "))
	q_uni = get_unigrams(question)

	
	candidateless_sent = remove_candidates(sent, candidate)

	sent_bi = get_bigrams(candidateless_sent.split(" "))
	sent_uni = get_unigrams(candidateless_sent)

	uni_score = get_overlap(q_uni, sent_uni)
	bi_score = get_overlap(q_bi, sent_bi)
	score = uni_score + bi_score

	return tuple([candidate, score])


def get_overlap(dict1, dict2):
	score = 0
	for u in dict1:
		if u in dict2:
			score += dict2[u]

	return score


def remove_candidates(sentence, candidate):
	candidate += " "
	candidateless_sent = ""
	for word in sentence.split(" "):
		if word not in candidate.split(" "):
			candidateless_sent += word + " "
	candidateless_sent = candidateless_sent[:-1] + "."
	return candidateless_sent

def get_unigrams(text):
	unigrams = re.findall(r"[\p{L}'0-9]+", text)
	freq = {}
	for uni in unigrams:
		if uni in freq:
			freq[uni] += 1
		else:
			freq[uni] = 1
	return freq

def get_bigrams(words):
	bigrams = [tuple(words[inx:inx + 2]) for inx in range(len(words) - 1)]
	frequencies = {}
	for bigram in bigrams:
		if bigram in frequencies:
			frequencies[bigram] += 1
		else:
			frequencies[bigram] = 1
	return frequencies

def get_sentences(corpus):
	if corpus[-1] not in ".:?!":
		corpus = corpus + "."
	r = re.sub("[.:?!]", "\n", corpus)
	r = re.sub("[,;(){\}[\]]", "", r)
	r = r.split("\n")
	return r[:-1]

def get_occurences(word, text):
	counter = 0
	for w in text:
		if w.lower() == word.lower():
			counter += 1
	return counter

def get_tf_idf(word, text):
	sent = get_sentences(text)
	formatted_text = re.findall(r"[\p{L}'0-9]+", text)
	oc = get_occurences(word, formatted_text)
	tf = oc / len(formatted_text)
	occ = get_occurences(word, formatted_text)
	if occ is 0:
		occ = 1
	idf = math.log10(len(formatted_text) / occ)
	return tf * idf

def get_tf_idf2(word, span):
	oc = get_occurences(word, span)
	tf = oc/len(span)
	occ = oc
	if occ is 0:
		occ = 1
	idf = math.log10(len(span)/occ)
	return tf * idf

def get_cosine_sim(q_dict, tf_idf):
	dot_sum = 0
	absA = 0
	absB = 0
	for i in q_dict.keys():
		p = 0
		if i in tf_idf:
			p = tf_idf[i]
		dot_sum += (1 / len(q_dict)) * p
		absA += (1 / len(q_dict)) ** 2
		absB += p ** 2

	try:
		cosin_sim = dot_sum / math.sqrt(absA * absB)
	except ZeroDivisionError:
		cosin_sim = 0

	return cosin_sim
