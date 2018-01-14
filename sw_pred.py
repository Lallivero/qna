import json
import util
import answer_extraction_new
import score_calc
import random
import operator

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
		print("Topic:",article.title)
		for paragraph in article.paragraphs:
			
			context = paragraph.context
			 #print("Paragraph:", context.text[:20])
			q_list = [tuple([paragraph.qas[i].question, paragraph.qas[i].id]) for i in range(0,len(paragraph.qas))]
			for Q in q_list:
				#print(Q[1], Q[0].text)
				candidate_ans = dict()
				b = answer_extraction_new.sliding_window_calc(Q[0], context)
				for t in b:
					candidate_ans[t[1]] = t[0]

				id = Q[1]
	#			print(id, Q[0].text)
				try:
					ans = sorted(candidate_ans.items(), key = operator.itemgetter(1), reverse=True)
					#print(ans)
					#print(Q[0].text, ans[0][0])
					big_json[id] = ans[0][0]	
					#print(big_json[id])
				except:
					print("ERROR WITH QUESTION (ID):", id)
					big_json[id] = "--null--"
		article = src.proto.io.ReadArticle(fobj)

	save = open("sw_pred_const30.json", "w")
	save.write(json.dumps(big_json))
	print("\n\nCompleted...", len(big_json),"Questions answered")
	save.close()

