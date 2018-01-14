#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <gflags/gflags.h>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "proto/io.h"
#include "proto/training_dataset.pb.h"
#include "utils/base.h"

using namespace std;

DEFINE_string(input_dictionary, "", "");
DEFINE_string(input_train_features, "", "");
DEFINE_string(input_dev_features, "", "");
DEFINE_string(ablate_features, "", "");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  set<string> ablated_features;
  boost::split(ablated_features, FLAGS_ablate_features, boost::is_any_of(","));

  unordered_map<int, int> keep_features;
  auto dictionary = OpenForReading(FLAGS_input_dictionary);
  while (true) {
    DictionaryEntry entry;
    if (!ReadDelimitedFrom(dictionary.get(), &entry)) {
      break;
    }
    int equal_index = entry.name().find(" =");
    CHECK(equal_index != -1);
    string feature_type = entry.name().substr(0, equal_index);
    if (ablated_features.count(feature_type)) {
      continue;
    }

    int new_index = keep_features.size();
    keep_features[entry.index()] = new_index;
    if (keep_features.size() % 1000000 == 0) {
      cout << "Feature " << keep_features.size() << endl;
    }
  }

  cout << "Using " << keep_features.size() << " features." << endl;

  for (const string& dataset :
       {FLAGS_input_train_features, FLAGS_input_dev_features}) {
    auto input = OpenForReading(dataset);
    auto output = OpenForWriting(dataset.substr(0, dataset.find(".proto")) +
                                 "-filtered.proto");

    int num_articles = 0;
    while (true) {
      TrainingArticle article;
      if (!ReadDelimitedFrom(input.get(), &article)) {
        break;
      }
      ++num_articles;
      if (num_articles % 10 == 0) {
        cout << dataset << " article " << num_articles << endl;
      }

      for (TrainingParagraph& paragraph : *article.mutable_paragraphs()) {
        for (TrainingQuestionAnswer& qa : *paragraph.mutable_questions()) {
          for (CandidateAnswerFeatures& features :
               *qa.mutable_candidateanswerfeatures()) {
            int new_size = 0;
            for (int i = 0; i < features.indices_size(); ++i) {
              auto it = keep_features.find(features.indices(i));
              if (it != keep_features.end()) {
                features.set_indices(new_size, it->second);
                ++new_size;
              }
            }
            features.mutable_indices()->Truncate(new_size);
          }
        }
      }

      WriteDelimitedTo(article, output.get());
    }
  }
}
