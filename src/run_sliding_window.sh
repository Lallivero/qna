# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run dev-annotated.proto:0xd62318/dev-annotated.proto dev.json:0x50be64 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_final_dev --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john2
   

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run dev-annotated.proto:0xd62318/dev-annotated.proto dev.json:0x50be64 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_dist_base_final_dev --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john2
   

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run test-annotated.proto:0xbccc00/test-annotated.proto test.json:0xcb7246 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_final_test --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john2
   

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run test-annotated.proto:0xbccc00/test-annotated.proto test.json:0xcb7246 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_dist_base_final_test --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3
   

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run train-annotated.proto:0x28ceb3/train-annotated.proto train.json:0x049962 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_final_train --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3


# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run train-annotated.proto:0x28ceb3/train-annotated.proto train.json:0x049962 src:src context_score.py:src/non_learning_baseline/context_score.py "python context_score.py" -n context_score_sliding_window_dist_base_final_train --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3





# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run dev-annotated.proto:0xd62318/dev-annotated.proto dev.json:0x50be64 src:src random_guess.py:src/non_learning_baseline/random_guess.py "python random_guess.py" -n random_guess_final_dev --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3
    

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run test-annotated.proto:0xbccc00/test-annotated.proto test.json:0xcb7246 src:src random_guess.py:src/non_learning_baseline/random_guess.py "python random_guess.py" -n random_guess_final_test --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3
   

# rm src.zip
# zip -r src.zip src
# cl work squad-non-learning-baseline
# cl upload src.zip
# cl run train-annotated.proto:0x28ceb3/train-annotated.proto train.json:0x049962 src:src random_guess.py:src/non_learning_baseline/random_guess.py "python random_guess.py" -n random_guess_final_train --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3

rm src.zip
zip -r src.zip src
cl upload src.zip
cl run dev-annotated.proto:0x75f01d/dev-annotated.proto dev-v1.0.json:0x4963ab 
