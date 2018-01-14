import requests
import sys
import os
from src.utils import squad_utils
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/proto"))
import src.proto.io

def annotate_text(text, lang="en"):
	url = "http://vilde.cs.lth.se:9000/"+lang+"/default/api/tsv"
	response = requests.post(url, data = text.encode("utf8").decode('iso-8859-1').encode("utf8"))
	return response.text

def format_text(header, text):
    alist = list()
    newHeader = header[1:].split("\t")
    for a in text:
        d = dict()
        t = a.split('\t')
        i = 0
        for c in newHeader:
            d[c] = t[i]
            i += 1
        alist.append(d)
    return alist

def find_n_grams(text, n=4):
	words = text.split(' ')
	return [tuple(words[inx:inx + n]) for inx in range(len(words) - 1)]
'''
def proto_extraction(file):
	fobj = open(file, "rb")
	article = src.proto.io.ReadArticle(fobj)
	return article
'''
def get_constituent(sent, n=4):
	constituents = squad_utils.GetConstituentSpanBySentence(sent.parseTree)

	ans = list()
	for start, end in constituents:
		l = [t.value for t in sent.token[start:end]]
		string = ""
		for word in l:
			string += word + " "
		if len(string.split(" ")) <= n:	
			ans.append(string[:-1])
		
	return ans

def get_ne(sent):

	ans = list()
	curr_ner = ""
	token = sent.token[0]
	i = 0
	for token in sent.token:
		if len(token.ner) > 1:
			curr_ner += token.word + " "
		elif curr_ner is not "":
			ans.append(curr_ner[:-1])
			curr_ner = ""
	if len(ans) == 0:
		ans.append("null")
	return ans
