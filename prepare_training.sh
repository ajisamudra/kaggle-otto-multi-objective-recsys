# split data into chunks to avoid OOM
make split_into_chunks MODE=training_train

# candidate retrieval
make candidate_covisit_list MODE=training_train
make candidate_word2vec_list MODE=training_train
make candidate_fasttext_list MODE=training_train
make candidate_matrix_fact_list MODE=training_train
make preprocess_popular_week_candidate MODE=training_train
make candidate_popular_week_list MODE=training_train
make preprocess_query_representation MODE=training_train
make candidate_word2vec_duration_list MODE=training_train
make candidate_word2vec_weighted_recency_list MODE=training_train
make candidate_word2vec_weighted_duration_list MODE=training_train

# dedup candidates and negative sampling
make candidate_rows MODE=training_train

# make features
make session_features MODE=training_train
make session_covisit_features MODE=training_train
make item_features MODE=training_train
make session_item_features MODE=training_train
make item_hour_features MODE=training_train
make item_weekday_features MODE=training_train
make session_representation_items MODE=training_train
make item_covisitation_features MODE=training_train START=0 END=30
make word2vec_features MODE=training_train START=0 END=30
make combine_features  MODE=training_train START=0 END=30
