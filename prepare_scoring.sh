# make split_into_chunks MODE=scoring_test
make candidate_list MODE=scoring_test
make candidate_word2vec_list MODE=scoring_test
make candidate_fasttext_list MODE=scoring_test
make candidate_matrix_fact_list MODE=scoring_test
make preprocess_popular_hour_candidate MODE=scoring_test
make candidate_popular_hour_list MODE=scoring_test
make candidate_rows MODE=scoring_test
make session_features MODE=scoring_test
make item_features MODE=scoring_test
make session_item_features MODE=scoring_test
make item_hour_features MODE=scoring_test
make item_weekday_features MODE=scoring_test
make session_representation_items MODE=scoring_test

make item_covisitation_features MODE=scoring_test START=0 END=50
make item_covisitation_features MODE=scoring_test START=50 END=100

make matrix_factorization_features MODE=scoring_test START=0 END=50
make matrix_factorization_features MODE=scoring_test START=50 END=100

make word2vec_features MODE=scoring_test START=0 END=50
make word2vec_features MODE=scoring_test START=50 END=100
make fasttext_features MODE=scoring_test START=0 END=50
make fasttext_features MODE=scoring_test START=50 END=100

make combine_features  MODE=scoring_test START=0 END=50
make combine_features  MODE=scoring_test START=50 END=100
# ./run_scoring.sh
