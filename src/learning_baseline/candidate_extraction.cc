#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

#include <gflags/gflags.h>

#include "learning_baseline/feature_based/feature_extractor.h"
#include "proto/dataset.pb.h"
#include "proto/CoreNLP.pb.h"
#include "proto/training_dataset.pb.h"
#include "proto/io.h"
#include "utils/base.h"
#include "utils/thread_pool.h"

using google::protobuf::RepeatedPtrField;
using namespace std;
using namespace edu::stanford::nlp::pipeline;

DEFINE_string(input, "", "Path to the input articles.");
DEFINE_string(output, "", "Path where to output the training articles.");
DEFINE_int32(extraction_threads, 1, "Number of threads to use for extraction.");
DEFINE_bool(paragraph_level, false,
            "Whether to consider candidate answers at the paragraph level or "
            "at the article level.");

DEFINE_bool(extract_features, false,
            "Whether to extract features for candidate answers.");

int IterateOverSpans(const Sentence& sentence, const ParseTree& parse_tree,
                     bool output_span, int span_begin_index,
                     function<void(const ParseTree&, int, int)> callback) {
  int span_length = parse_tree.child_size() == 0 ? 1 : 0;
  for (const ParseTree& child : parse_tree.child()) {
    bool output_child_span = parse_tree.child_size() > 1;
    span_length += IterateOverSpans(sentence, child, output_child_span,
                                    span_begin_index + span_length, callback);
  }
  if (output_span) {
    callback(parse_tree, span_begin_index, span_length);
  }
  return span_length;
}

void FillParagraphLevelCandidateAnswers(
    const Article& article, int paragraph_index,
    RepeatedPtrField<CandidateAnswer>* candidate_answers,
    vector<const ParseTree*>* parse_trees) {
  const Paragraph& paragraph = article.paragraphs(paragraph_index);
  for (int sentence_index = 0;
       sentence_index < paragraph.context().sentence_size(); ++sentence_index) {
    const Sentence& sentence = paragraph.context().sentence(sentence_index);
    IterateOverSpans(sentence, sentence.parsetree(), true, 0,
                     [&](const ParseTree& parse_tree, int span_begin_index,
                         int span_length) {
                       CandidateAnswer* candidate_answer =
                           candidate_answers->Add();
                       candidate_answer->set_paragraphindex(paragraph_index);
                       candidate_answer->set_sentenceindex(sentence_index);
                       candidate_answer->set_spanbeginindex(span_begin_index);
                       candidate_answer->set_spanlength(span_length);
                       parse_trees->emplace_back(&parse_tree);
                     });
  }
}

void FillArticleLevelCandidateAnswers(
    const Article& article,
    RepeatedPtrField<CandidateAnswer>* candidate_answers,
    vector<const ParseTree*>* parse_trees) {
  for (int paragraph_index = 0; paragraph_index < article.paragraphs_size();
       ++paragraph_index) {
    FillParagraphLevelCandidateAnswers(article, paragraph_index,
                                       candidate_answers, parse_trees);
  }
}

string ReconstructAnswer(const Document& answer) {
  ostringstream sout;
  bool first_sentence = true;
  for (const Sentence& sentence : answer.sentence()) {
    if (first_sentence) {
      first_sentence = false;
    } else {
      sout << " ";
    }
    for (const Token& token : sentence.token()) {
      sout << token.word() << token.after();
    }
  }
  return sout.str();
}

string ReconstructSpan(const Sentence& sentence, int span_begin_index,
                       int span_length) {
  ostringstream sout;
  for (int i = span_begin_index; i - span_begin_index < span_length; ++i) {
    sout << sentence.token(i).word();
    if (i - span_begin_index != span_length - 1) {
      sout << sentence.token(i).after();
    }
  }
  return sout.str();
}

bool FillCorrectAnswerIndex(
    const Article& article,
    const RepeatedPtrField<CandidateAnswer>& candidate_answers,
    int paragraph_index, const QuestionAnswer& qa,
    TrainingQuestionAnswer* training_qa) {
  const Paragraph& paragraph = article.paragraphs(paragraph_index);
  int sentence_index = -1;
  for (const Sentence& sentence : paragraph.context().sentence()) {
    if (sentence.characteroffsetbegin() <= qa.answeroffsets(0)) {
      sentence_index = sentence.sentenceindex();
    }
  }

  const string answer_text = ReconstructAnswer(qa.answers(0));

  if (sentence_index != -1) {
    const Sentence& sentence = paragraph.context().sentence(sentence_index);
    const string sentence_text =
        ReconstructSpan(sentence, 0, sentence.token_size());

    if (sentence_text.find(answer_text) == -1) {
      sentence_index = -1;
    }
  }

  int correct_answer_index = -1;
  int shortest_correct_answer_length = std::numeric_limits<int>::max();

  for (int i = 0; i < candidate_answers.size(); ++i) {
    const CandidateAnswer& candidate_answer = candidate_answers.Get(i);
    if (candidate_answer.paragraphindex() != paragraph_index) {
      continue;
    }
    if (sentence_index != -1 &&
        candidate_answer.sentenceindex() != sentence_index) {
      continue;
    }
    const Paragraph& paragraph =
        article.paragraphs(candidate_answer.paragraphindex());
    const Sentence& sentence =
        paragraph.context().sentence(candidate_answer.sentenceindex());

    const string span_text =
        ReconstructSpan(sentence, candidate_answer.spanbeginindex(),
                        candidate_answer.spanlength());
    if (span_text.size() <= shortest_correct_answer_length &&
        span_text.find(answer_text) != -1) {
      correct_answer_index = i;
      shortest_correct_answer_length = span_text.size();
    }
  }

  if (correct_answer_index == -1) {
    cout << "Unable to find correct answer for article " << article.title()
         << ", question " << qa.question().text() << ", answer "
         << qa.answers(0).text() << endl;
    return false;
  }

  training_qa->set_correctanswerindex(correct_answer_index);
  return true;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(!FLAGS_input.empty());
  CHECK(!FLAGS_output.empty());

  vector<Article> articles = ReadMessages<Article>(FLAGS_input);

  ThreadPool thread_pool(FLAGS_extraction_threads);
  unique_ptr<google::protobuf::io::ZeroCopyOutputStream> output =
      OpenForWriting(FLAGS_output);
  mutex output_lock;
  unique_ptr<FeatureExtractor> feature_extractor;
  if (FLAGS_extract_features) {
    feature_extractor.reset(new FeatureExtractor);
  }
  for (const Article& article : articles) {
    thread_pool.Submit([&]() {
      TrainingArticle training_article;
      training_article.set_title(article.title());

      RepeatedPtrField<TrainingQuestionAnswer>* training_questions;
      const RepeatedPtrField<CandidateAnswer>* candidate_answers;
      const vector<const ParseTree*>*
          parse_trees;  // Used during feature extraction.

      vector<const ParseTree*> article_level_parse_trees;
      if (!FLAGS_paragraph_level) {
        FillArticleLevelCandidateAnswers(
            article, training_article.mutable_candidateanswers(),
            &article_level_parse_trees);
        training_questions = training_article.mutable_questions();
        candidate_answers = &training_article.candidateanswers();
        parse_trees = &article_level_parse_trees;
      }

      vector<vector<FeatureExtractor::PreprocessedSentence> >
          preprocessed_sentences;
      if (FLAGS_extract_features) {
        feature_extractor->PreprocessSentences(article,
                                               &preprocessed_sentences);
      }

      for (int paragraph_index = 0; paragraph_index < article.paragraphs_size();
           ++paragraph_index) {
        TrainingParagraph* training_paragraph;
        vector<const ParseTree*> paragraph_level_parse_trees;
        if (FLAGS_paragraph_level) {
          training_paragraph = training_article.add_paragraphs();
          FillParagraphLevelCandidateAnswers(
              article, paragraph_index,
              training_paragraph->mutable_candidateanswers(),
              &paragraph_level_parse_trees);
          training_questions = training_paragraph->mutable_questions();
          candidate_answers = &training_paragraph->candidateanswers();
          parse_trees = &paragraph_level_parse_trees;
        }

        for (const QuestionAnswer& qa :
             article.paragraphs(paragraph_index).qas()) {
          TrainingQuestionAnswer* training_qa = training_questions->Add();
          training_qa->set_id(qa.id());

          if (!FillCorrectAnswerIndex(article, *candidate_answers,
                                      paragraph_index, qa, training_qa)) {
            training_questions->RemoveLast();
            continue;
          }

          if (FLAGS_extract_features) {
            feature_extractor->ExtractFeatures(article, preprocessed_sentences,
                                               *candidate_answers, *parse_trees,
                                               qa.question(), training_qa);
          }
        }
      }

      std::lock_guard<std::mutex> guard(output_lock);
      CHECK(WriteDelimitedTo(training_article, output.get()));
    });
  }
  thread_pool.ShutDown();
}
