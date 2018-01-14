def BuildPredictions(question_annotations, example, top_predicted_candidate_indices):
    predictions = {}
    for question_index, question_id in enumerate(example.question_ids):
        predicted_span = example.candidate_answers[top_predicted_candidate_indices[question_index]]
        predictions[question_id] = BuildPrediction(question_annotations[question_id], predicted_span)
    return predictions

EXTRA_WORDS = set(['the', 'a', 'an'])

def BuildPrediction(annotations, predicted_span):
    sentence_tokens = annotations.article.paragraphs[predicted_span.paragraphIndex].context.sentence[predicted_span.sentenceIndex].token
    predicted_answer = ''
    for i in xrange(predicted_span.spanBeginIndex, predicted_span.spanBeginIndex + predicted_span.spanLength):
        predicted_answer += sentence_tokens[i].originalText
        if i != predicted_span.spanBeginIndex + predicted_span.spanLength - 1:
            predicted_answer += sentence_tokens[i].after
            
    predicted_answer_tokens = predicted_answer.split(' ')
    if len(predicted_answer_tokens) and predicted_answer_tokens[0] in EXTRA_WORDS:
        predicted_answer_tokens = predicted_answer_tokens[1:]
            
    return ' '.join(predicted_answer_tokens)
