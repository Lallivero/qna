import json
import util
import score_calc
import random

import sys
import os
from src.utils import squad_utils
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/proto"))
import src.proto.io

if __name__=="__main__":
	file_name = "dev-anotated.proto/dev-annotated.proto"
	fobj = open(file_name, "rb")

	big_json = {}

	article = src.proto.io.ReadArticle(fobj)

	while article:
		#print("Topic:",article.title)
		for paragraph in article.paragraphs:
			context = paragraph.context

			q_list = [paragraph.qas[i] for i in range(0,len(paragraph.qas))]
			for Q in q_list:
				
#				sent = score_calc.get_candidate_likelihood(Q.question.text, context)
				sent = context.sentence

				candidate_ans = list()
				for i in range(0,len(sent)):
					a = util.get_constituent(sent[i],n=30)
					for t in a:
						candidate_ans.append(t)

				rand = (int)(random.random()*(len(candidate_ans)))

				id = Q.id
				try:
					rand_ans = candidate_ans[rand]

					#print("Ans:",rand_ans)

					big_json[id] = rand_ans
				except:
					print("ERROR WITH INDEX:", rand)
					big_json[id] = "--null--"

		article = src.proto.io.ReadArticle(fobj)

	save = open("rand_pred_const.json", "w")
	save.write(json.dumps(big_json))
	print("Done")
	#print(big_json)
	save.close()

'''
	for topic in data:
		for paragraph in topic['paragraphs']:
			context = paragraph['context']
			q_list = [tuple([paragraph['qas'][i]['question'], paragraph['qas'][i]['id']]) for i in range(0,len(paragraph['qas']))]
			for Q in q_list:
				sent = score_calc.get_candidate_likelihood(Q[0], context)

				candidate_ans = list()
				for i in range(0,len(sent)):
					a = util.annotate_text(sent[i][0]+".", "en").split('\n')
					a = util.format_text(a[0], a[1:-1])
					a = util.get_consituent(a, str(i+1))
					for t in a:
						candidate_ans.append(t)

				rand = (int)(random.random()*(len(candidate_ans)))

				id = Q[1]
				try:
					rand_ans = candidate_ans[rand]

					print("Ans:",rand_ans[0])

					big_json[id] = rand_ans[0]
				except:
					print("ERROR WITH INDEX:", rand)
					big_json[id] = "--null--"

	save = open("rand_pred_ne.json", "w")
	save.write(json.dumps(big_json))
	print("Done")
	print(big_json)
	save.close()
'''
