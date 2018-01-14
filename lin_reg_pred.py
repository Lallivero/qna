import json
import util
import answer_extraction_new
import score_calc
import feature_extraction
import random
import operator

import sys
import os
from src.utils import squad_utils
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/proto"))
import src.proto.io
import pickle

import sklearn.linear_model
from sklearn.feature_extraction import DictVectorizer
import numpy as np

if __name__=="__main__":
	file_name = "dev-anotated.proto/dev-annotated.proto"
	fobj = open(file_name, "rb")
	big_json = {}

	model = pickle.load(open("model.pkl","rb"))
	v = pickle.load(open("dvec.pkl","rb"))

	article = src.proto.io.ReadArticle(fobj)
	while article:
		print("Topic:",article.title)
		for paragraph in article.paragraphs:

			context = paragraph.context
			 #print("Paragraph:", context.text[:20])
			q_list = [paragraph.qas[i] for i in range(0,len(paragraph.qas))]
			for Q in q_list:
				#print(Q.id, Q.question.text)

				id = Q.id
				try:
					#print(id)
					#NEED TO GIVE THE PROPER FEATURES INTO THE PREDICT FUNCTION
					features, constituents = feature_extraction.manual_extraction_per_q(context, Q)
					#print(features)
					transformed_features = v.transform(features)
#					y, dict_classes, inv_dict_classes = feature_extraction.encode_classes(constituents)
					#print(transformed_features)
#					print(inv_dict_classes)
					ans = model.predict(transformed_features)
					#print(ans)
					#ans = v.inverse_transform[int(ans)]
					#print(ans)

					#ans = ans.split("=")[1]

					#print([i for i in ans[0].keys()][0].split("=")[1])

					big_json[id] = ans[0]

					#print(big_json[id])
				except:
					print("ERROR WITH QUESTION (ID):", id, sys.exc_info())
					big_json[id] = "--null--"
					input("waiting:")
		article = src.proto.io.ReadArticle(fobj)
	#json_format = big_json.to_dict()
	save = open("lin_reg_pred.json", "w")
	save.write(json.dumps(big_json))
	print("\n\nCompleted...", len(big_json),"Questions answered")
	save.close()
