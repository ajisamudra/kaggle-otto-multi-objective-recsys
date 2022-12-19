# make split_into_chunks MODE=scoring_test
# make candidate_list MODE=scoring_test
make candidate_word2vec_list MODE=scoring_test
# make candidate_fasttext_list MODE=scoring_test
# make candidate_matrix_fact_list MODE=scoring_test
make candidate_rows MODE=scoring_test
# make session_features MODE=scoring_test
# make item_features MODE=scoring_test
# make session_item_features MODE=scoring_test
# make item_hour_features MODE=scoring_test
# make item_weekday_features MODE=scoring_test
# make session_representation_items MODE=scoring_test

# make item_covisitation_features MODE=scoring_test START=0 END=20
# make item_covisitation_features MODE=scoring_test START=20 END=40
# make item_covisitation_features MODE=scoring_test START=40 END=60
# make item_covisitation_features MODE=scoring_test START=60 END=80

# make matrix_factorization_features MODE=scoring_test START=0 END=20
# make matrix_factorization_features MODE=scoring_test START=20 END=40
# make matrix_factorization_features MODE=scoring_test START=40 END=60
# make matrix_factorization_features MODE=scoring_test START=60 END=80

# make word2vec_features MODE=scoring_test START=0 END=40
# make word2vec_features MODE=scoring_test START=40 END=80
# make fasttext_features MODE=scoring_test START=0 END=40
# make fasttext_features MODE=scoring_test START=40 END=80

make combine_features  MODE=scoring_test START=15 END=20
make combine_features  MODE=scoring_test START=20 END=40
make combine_features  MODE=scoring_test START=40 END=60
make combine_features  MODE=scoring_test START=60 END=80
./run_scoring.sh
