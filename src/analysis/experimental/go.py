from collections import defaultdict
from proto.io import ReadArticles

sentence_lengths = defaultdict(int)

articles = ReadArticles('dataset/qa-1460521688980_new-train-annotated.proto')
for article in articles:
    for paragraph in article.paragraphs:
        for sentence in paragraph.context.sentence:
            sentence_lengths[len(sentence.token)] += 1
            
for length, count in sorted(sentence_lengths.items()):
    print length, count