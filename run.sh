# take last 1 week data from the whole train data
# python src/preprocess/split_local_train_label.py --mode training
# python src/preprocess/split_local_train_label.py --mode scoring

# split data last 1 week into 10 chunks
# python src/preprocess/split_data_into_chunks.py --mode scoring_train
# python src/preprocess/split_data_into_chunks.py --mode scoring_test
# python src/preprocess/split_data_into_chunks.py --mode training_train
# python src/preprocess/split_data_into_chunks.py --mode training_test

# generate candidates from retrieval
# python src/pipeline/make_candidates_list.py --mode scoring_train
# python src/pipeline/make_candidates_list.py --mode scoring_test
# python src/pipeline/make_candidates_list.py --mode training_train
# python src/pipeline/make_candidates_list.py --mode training_test

# pivot candidate list to candidate rows
# python src/pipeline/make_candidates_rows.py --mode scoring_train
# python src/pipeline/make_candidates_rows.py --mode scoring_test
# python src/pipeline/make_candidates_rows.py --mode training_test
# python src/pipeline/make_candidates_rows.py --mode training_train

# session_features
# python src/features/make_session_features.py --mode training_train
# python src/features/make_session_features.py --mode training_test
# python src/features/make_session_features.py --mode scoring_test
# python src/features/make_session_features.py --mode scoring_train

# interaction between session & item features
# python src/features/make_session_item_features.py --mode training_train
# python src/features/make_session_item_features.py --mode training_test
# python src/features/make_session_item_features.py --mode scoring_test
# python src/features/make_session_item_features.py --mode scoring_train

# combine features
# python src/pipeline/make_combine_features.py --mode training_train
# python src/pipeline/make_combine_features.py --mode training_test
# python src/pipeline/make_combine_features.py --mode scoring_test
# python src/pipeline/make_combine_features.py --mode scoring_train

python src/training/train.py --event orders

# idea: item & day/hour popularity features
# idea: word2vec for click / cart / order separately -> to recommend separately
