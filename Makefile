SAMPLE_MODE="scoring" # training
MODE="training_train" # training_test / scoring_train / scoring_test

sample_last_week_data:
	python src/preprocess/split_local_train_label.py --mode $(SAMPLE_MODE)

split_into_chunks:
	python src/preprocess/split_data_into_chunks.py --mode $(MODE)

candidate_list:
	python src/pipeline/make_candidates_list.py --mode $(MODE)

candidate_rows:
	python src/pipeline/make_candidates_rows.py --mode $(MODE)

session_features:
	python src/features/make_session_features.py --mode $(MODE)

session_item_features:
	python src/features/make_session_item_features.py --mode $(MODE)

item_features:
	python src/features/make_item_features.py --mode $(MODE)

item_hour_features:
	python src/features/make_item_hour_features.py --mode $(MODE)

item_weekday_features:
	python src/features/make_item_weekday_features.py --mode $(MODE)

item_covisitation_features:
	python src/features/make_item_covisitation_features.py --mode $(MODE)

combine_features:
	python src/pipeline/make_combine_features.py --mode $(MODE)

remake_session_features: session_features combine_features
remake_session_item_features: session_item_features combine_features
remake_item_features: item_features combine_features
remake_item_hour_features: item_hour_features combine_features
remake_item_weekday_features: item_weekday_features combine_features

EDA_MODE="biserial" # across_dataset/class_dist/chunk_dist/biserial/all
eda:
	python src/auto_eda/eda.py --n 2 --mode $(EDA_MODE)

ALGO="lgbm_classifier" # cat_classifier / lgbm_ranker / cat_ranker
train:
	python src/training/train.py --event all --n 1 --algo $(ALGO) --week w2 --eval 1


CLICK_MODEL="2022-12-09_clicks_lgbm_classifier_12209_69509"
CART_MODEL="2022-12-09_carts_lgbm_classifier_8886_64070"
ORDER_MODEL="2022-12-09_orders_lgbm_classifier_10050_64507"
WEEK_DATA="w2" # w2 for validation / w1 for scoring
score:
	python src/scoring/score.py --event orders --week_data $(WEEK_DATA) --week_model w2 --artifact $(ORDER_MODEL)
	python src/scoring/score.py --event carts --week_data $(WEEK_DATA) --week_model w2 --artifact $(CART_MODEL)
	python src/scoring/score.py --event clicks --week_data $(WEEK_DATA) --week_model w2 --artifact $(CLICK_MODEL)

submission:
	python src/scoring/make_submission.py --click_model $(CLICK_MODEL) --cart_model $(CART_MODEL) --order_model $(ORDER_MODEL) --week_data $(WEEK_DATA) --week_model w2

eval_submission:
	python src/scoring/eval_submission.py --click_model $(CLICK_MODEL) --cart_model $(CART_MODEL) --order_model $(ORDER_MODEL)

score_and_eval: submission eval_submission