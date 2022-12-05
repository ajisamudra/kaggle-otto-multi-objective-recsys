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

# perform training
# python src/training/train.py --event orders
# python src/training/train.py --event orders --n 1 --algo catboost --week w2

# perform scoring
# week_data w2 means local validation
# week_data w1 means real test submission
# python src/scoring/score.py --event orders --week_data w2 --week_model w2 --artifact 2022-12-05_orders_lgbm_44683_82685
# python src/scoring/score.py --event carts --week_data w2 --week_model w2 --artifact 2022-12-05_carts_lgbm_22488_72684
# python src/scoring/score.py --event clicks --week_data w2 --week_model w2 --artifact 2022-12-05_clicks_lgbm_36860_73857

# # make submission
# python src/scoring/make_submission.py --click_model 2022-12-05_clicks_lgbm_36860_73857 --cart_model 2022-12-05_carts_lgbm_22488_72684 --order_model 2022-12-05_orders_lgbm_44683_82685 --week_data w2 --week_model w2

# eval submission, only for week_data & week_model w2
python src/scoring/eval_submission.py --click_model 2022-12-05_clicks_lgbm_36860_73857 --cart_model 2022-12-05_carts_lgbm_22488_72684 --order_model 2022-12-05_orders_lgbm_44683_82685


# idea: item & day/hour popularity features
# idea: word2vec for click / cart / order separately -> to recommend separately
