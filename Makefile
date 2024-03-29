SAMPLE_MODE="scoring" # training
MODE="training_train" # training_test / scoring_train / scoring_test

sample_last_week_data:
	python src/preprocess_v2/split_local_train_label.py --mode $(SAMPLE_MODE)

split_into_chunks:
	python src/preprocess_v2/split_data_into_chunks.py --mode $(MODE)

preprocess_query_representation:
	python src/preprocess_v2/make_query_representation.py --mode $(MODE)

preprocess_popular_daily_candidate:
	python src/preprocess_v2/make_popular_daily_item_candidates.py --mode $(MODE)

preprocess_popular_week_candidate:
	python src/preprocess_v2/make_popular_week_item_candidates.py --mode $(MODE)

preprocess_popular_datehour_candidate:
	python src/preprocess_v2/make_popular_datehour_item_candidates.py --mode $(MODE)

preprocess_popular_hour_candidate:
	python src/preprocess_v2/make_popular_hour_item_candidates.py --mode $(MODE)

candidate_past_aids_list:
	python src/pipeline_v2/make_candidates_past_aids_list.py --mode $(MODE)

candidate_covisit_list:
	python src/pipeline_v2/make_candidates_covisit_list.py --mode $(MODE)

candidate_matrix_fact_list:
	python src/pipeline_v2/make_candidates_matrix_factorization_list.py --mode $(MODE)

candidate_word2vec_list:
	python src/pipeline_v2/make_candidates_word2vec_list.py --mode $(MODE)

candidate_word2vec_duration_list:
	python src/pipeline_v2/make_candidates_word2vec_duration_list.py --mode $(MODE)

candidate_word2vec_weighted_recency_list:
	python src/pipeline_v2/make_candidates_word2vec_weighted_recency_list.py --mode $(MODE)

candidate_word2vec_weighted_duration_list:
	python src/pipeline_v2/make_candidates_word2vec_weighted_duration_list.py --mode $(MODE)

candidate_fasttext_list:
	python src/pipeline_v2/make_candidates_fasttext_list.py --mode $(MODE)

candidate_popular_daily_list:
	python src/pipeline_v2/make_candidates_popular_daily_list.py --mode $(MODE)

candidate_popular_datehour_list:
	python src/pipeline_v2/make_candidates_popular_datehour_list.py --mode $(MODE)

candidate_popular_hour_list:
	python src/pipeline_v2/make_candidates_popular_hour_list.py --mode $(MODE)

candidate_popular_week_list:
	python src/pipeline_v2/make_candidates_popular_week_list.py --mode $(MODE)

candidate_list_eval:
	python src/pipeline_v2/eval_candidate_list.py --mode $(MODE)

candidate_rows:
	python src/pipeline_v2/make_candidates_rows.py --mode $(MODE)

session_features:
	python src/features_v2/make_session_features.py --mode $(MODE)

session_covisit_features:
	python src/features_v2/make_session_covisitation_features.py --mode $(MODE)

session_item_features:
	python src/features_v2/make_session_item_features.py --mode $(MODE)

session_item_features_l2:
	python src/features_v2/make_session_item_features_l2.py --mode $(MODE)

item_features:
	python src/features_v2/make_item_features.py --mode $(MODE)

item_hour_features:
	python src/features_v2/make_item_hour_features.py --mode $(MODE)

item_weekday_features:
	python src/features_v2/make_item_weekday_features.py --mode $(MODE)

session_representation_items:
	python src/features_v2/make_session_representation_items.py --mode $(MODE)

START=0
END=10
item_covisitation_features:
	python src/features_v2/make_item_covisitation_features_v2.py --mode $(MODE) --istart $(START) --iend $(END)

item_covisitation_features_old:
	python src/features_v2/make_item_covisitation_features.py --mode $(MODE) --istart $(START) --iend $(END)

matrix_factorization_features:
	python src/features_v2/make_item_matrix_fact_features.py --mode $(MODE) --istart $(START) --iend $(END)

word2vec_features:
	python src/features_v2/make_item_word2vec_features.py --mode $(MODE) --istart $(START) --iend $(END)

fasttext_features:
	python src/features_v2/make_item_fasttext_features.py --mode $(MODE) --istart $(START) --iend $(END)

combine_features:
	python src/pipeline_v2/make_combine_features.py --mode $(MODE) --istart $(START) --iend $(END)

target_encoding:
	python src/features_v2/make_target_encoding_features.py --mode $(MODE)

combine_features_l2:
	python src/pipeline_v2/make_combine_features_l2.py --mode $(MODE) --istart $(START) --iend $(END)

remake_session_features: session_features combine_features
remake_session_item_features: session_item_features combine_features
remake_item_features: item_features combine_features
remake_item_hour_features: item_hour_features combine_features
remake_item_weekday_features: item_weekday_features combine_features

# eda command
EDA_MODE="biserial" # across_dataset/class_dist/chunk_dist/biserial/all
eda:
	python src/auto_eda/eda.py --n 2 --mode $(EDA_MODE)

# train command
ALGO="lgbm_classifier" # cat_classifier / lgbm_ranker / cat_ranker
train:
	python src/training/train.py --event all --n 1 --algo $(ALGO) --week w2 --eval 1

one_ranker_dataset:
	python src/pipeline/make_one_ranker_training_dataset.py --mode training_train --istart 0 --iend 10

train_one_ranker:
	python src/training/train_one_ranker.py --event all --n 1 --algo $(ALGO) --week w2 --eval 1

tune_one_ranker:
	python src/training/tune_one_ranker.py --event all --k 1 --algo $(ALGO) --n_estimators 1000 --n_trial 15

# tune command
ALGO="lgbm_classifier" # cat_classifier / lgbm_ranker / cat_ranker
EVENT="orders"
K=1
TRIAL=30
tune:
	python src/training/tune.py --event $(EVENT) --k $(K) --algo $(ALGO) --n_trial $(TRIAL)

# scoring command
CLICK_MODEL="2022-12-09_clicks_lgbm_classifier_12209_69509"
CART_MODEL="2022-12-09_carts_lgbm_classifier_8886_64070"
ORDER_MODEL="2022-12-09_orders_lgbm_classifier_10050_64507"
WEEK_DATA="w2" # w2 for validation / w1 for scoring

export_treelite:
	python src/training/export_treelite.py --event orders --week_model w2 --artifact $(ORDER_MODEL)
	python src/training/export_treelite.py --event carts --week_model w2 --artifact $(CART_MODEL)
	python src/training/export_treelite.py --event clicks --week_model w2 --artifact $(CLICK_MODEL)

score:
	python src/scoring/score.py --event orders --week_data $(WEEK_DATA) --week_model w2 --artifact $(ORDER_MODEL)
	python src/scoring/score.py --event carts --week_data $(WEEK_DATA) --week_model w2 --artifact $(CART_MODEL)
	python src/scoring/score.py --event clicks --week_data $(WEEK_DATA) --week_model w2 --artifact $(CLICK_MODEL)

score_treelite:
	python src/scoring/score_treelite.py --event orders --week_data $(WEEK_DATA) --week_model w2 --artifact $(ORDER_MODEL)
	python src/scoring/score_treelite.py --event carts --week_data $(WEEK_DATA) --week_model w2 --artifact $(CART_MODEL)
	python src/scoring/score_treelite.py --event clicks --week_data $(WEEK_DATA) --week_model w2 --artifact $(CLICK_MODEL)

score_one_ranker_treelite:
	python src/scoring/score_one_ranker_treelite.py --event orders --week_data $(WEEK_DATA) --week_model w2 --artifact $(ORDER_MODEL)
	python src/scoring/score_one_ranker_treelite.py --event carts --week_data $(WEEK_DATA) --week_model w2 --artifact $(CART_MODEL)
	python src/scoring/score_one_ranker_treelite.py --event clicks --week_data $(WEEK_DATA) --week_model w2 --artifact $(CLICK_MODEL)

score_one_ranker:
	python src/scoring/score_one_ranker_treelite.py --event orders --week_data $(WEEK_DATA) --week_model w2 --artifact $(ORDER_MODEL)
	python src/scoring/score_one_ranker_treelite.py --event carts --week_data $(WEEK_DATA) --week_model w2 --artifact $(CART_MODEL)
	python src/scoring/score_one_ranker_treelite.py --event clicks --week_data $(WEEK_DATA) --week_model w2 --artifact $(CLICK_MODEL)

submission:
	python src/scoring/make_submission.py --click_model $(CLICK_MODEL) --cart_model $(CART_MODEL) --order_model $(ORDER_MODEL) --week_data $(WEEK_DATA) --week_model w2

eval_submission:
	python src/scoring/eval_submission.py --click_model $(CLICK_MODEL) --cart_model $(CART_MODEL) --order_model $(ORDER_MODEL)

score_and_eval: submission eval_submission


dataset_stacking:
	python src/stacking/make_stacking_dataset.py --mode $(MODE)

train_stacking:
	python src/stacking/train_stacking.py --algo=$(ALGO)

score_stacking:
	python src/stacking/score_stacking.py --artifact $(CLICK_MODEL) --event clicks --week_data $(WEEK_DATA) --week_model w2
	python src/stacking/score_stacking.py --artifact $(CART_MODEL) --event carts --week_data $(WEEK_DATA) --week_model w2
	python src/stacking/score_stacking.py --artifact $(ORDER_MODEL) --event orders --week_data $(WEEK_DATA) --week_model w2

submission_stacking:
	python src/stacking/make_submission_stacking.py --click_model $(CLICK_MODEL)  --cart_model $(CART_MODEL) --order_model $(ORDER_MODEL) --week_data $(WEEK_DATA) --week_model w2
