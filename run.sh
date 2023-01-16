# python src/ensemble/tuning_ensemble.py
# python src/ensemble/tuning_ensemble2.py
# python src/ensemble/score_ensemble.py
# python src/ensemble/tuning_ensemble3.py

# # make split_into_chunks MODE=training_train
# make candidate_past_aids_list MODE=training_train
# make candidate_covisit_list MODE=training_train
# make candidate_word2vec_list MODE=training_train
# make candidate_fasttext_list MODE=training_train
# make candidate_matrix_fact_list MODE=training_train
# make preprocess_popular_week_candidate MODE=training_train
# make candidate_popular_week_list MODE=training_train
# make preprocess_query_representation MODE=training_train
# make candidate_word2vec_duration_list MODE=training_train
# make candidate_word2vec_weighted_recency_list MODE=training_train
# make candidate_word2vec_weighted_duration_list MODE=training_train
# make candidate_rows MODE=training_train
make session_features MODE=training_train
# make item_features MODE=training_train
# make session_item_features MODE=training_train
# make item_hour_features MODE=training_train
# make item_weekday_features MODE=training_train
# make session_representation_items MODE=training_train
# make item_covisitation_features MODE=training_train START=0 END=30
# make matrix_factorization_features MODE=training_train START=0 END=30
# make word2vec_features START=0 END=30
# make fasttext_features START=0 END=30
# make session_item_features_l2 MODE=training_train
make combine_features  MODE=training_train START=0 END=30
# make target_encoding MODE=training_train
# make target_encoding MODE=training_test
# make combine_features_l2  MODE=training_train START=0 END=30


# # make split_into_chunks MODE=training_test
# make candidate_past_aids_list MODE=training_test
# make candidate_covisit_list MODE=training_test
# make preprocess_popular_hour_candidate MODE=training_test
# make candidate_popular_hour_list MODE=training_test
# make preprocess_popular_week_candidate MODE=training_test
# make candidate_popular_week_list MODE=training_test
# make candidate_word2vec_list MODE=training_test
# make preprocess_query_representation MODE=training_test
# make candidate_word2vec_duration_list MODE=training_test
# make candidate_word2vec_weighted_recency_list MODE=training_test
# make candidate_word2vec_weighted_duration_list MODE=training_test
# make candidate_fasttext_list MODE=training_test
# make candidate_matrix_fact_list MODE=training_test
# make candidate_list_eval MODE=training_test
# make candidate_rows MODE=training_test
make session_features MODE=training_test
# make item_features MODE=training_test
# make session_item_features MODE=training_test
# make item_hour_features MODE=training_test
# make item_weekday_features MODE=training_test
# make session_representation_items MODE=training_test
# make item_covisitation_features MODE=training_test START=0 END=40
# make item_covisitation_features MODE=training_test START=50 END=100
# make matrix_factorization_features MODE=training_test START=0 END=25
# make matrix_factorization_features MODE=training_test START=25 END=50
# make matrix_factorization_features MODE=training_test START=50 END=75
# make matrix_factorization_features MODE=training_test START=75 END=100
# make word2vec_features MODE=training_test START=0 END=40
# make word2vec_features MODE=training_test START=50 END=100
# make fasttext_features MODE=training_test START=0 END=50
# make fasttext_features MODE=training_test START=50 END=100
# make session_item_features_l2 MODE=training_test
make combine_features  MODE=training_test START=0 END=40
# make combine_features  MODE=training_test START=25 END=50
# make combine_features  MODE=training_test START=50 END=75
# make combine_features  MODE=training_test START=75 END=100
# make combine_features_l2  MODE=training_test START=0 END=40
# make combine_features_l2  MODE=training_test START=25 END=50
# make combine_features_l2  MODE=training_test START=50 END=75
# make combine_features_l2  MODE=training_test START=75 END=100

# make split_into_chunks MODE=scoring_test
# make candidate_list MODE=scoring_test
# make candidate_word2vec_list MODE=scoring_test
# make candidate_fasttext_list MODE=scoring_test
# make candidate_matrix_fact_list MODE=scoring_test
# make candidate_rows MODE=scoring_test
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
# make word2vec_features MODE=scoring_test START=0 END=20
# make word2vec_features MODE=scoring_test START=20 END=40
# make word2vec_features MODE=scoring_test START=40 END=60
# make word2vec_features MODE=scoring_test START=60 END=80

# make fasttext_features MODE=scoring_test START=0 END=20
# make fasttext_features MODE=scoring_test START=20 END=40
# make fasttext_features MODE=scoring_test START=40 END=60
# make fasttext_features MODE=scoring_test START=60 END=80

# make combine_features  MODE=scoring_test START=0 END=20
# make combine_features  MODE=scoring_test START=20 END=40
# make combine_features  MODE=scoring_test START=40 END=60
# make combine_features  MODE=scoring_test START=60 END=80


# make train_one_ranker ALGO=lgbm_ranker
# make train_one_ranker ALGO=cat_ranker

# make train ALGO=lgbm_ranker

# make combine_features  MODE=training_train START=0 END=30
# make combine_features  MODE=training_test START=0 END=40
# make combine_features  MODE=training_test START=25 END=50
# make combine_features  MODE=training_test START=50 END=75
# make combine_features  MODE=training_test START=75 END=100

make train ALGO=cat_ranker
# make train ALGO=cat_classifier
# make item_covisitation_features MODE=training_train START=0 END=30
# make item_covisitation_features MODE=training_test START=0 END=40
# make train ALGO=lgbm_ranker
# make word2vec_features START=0 END=30
# make word2vec_features MODE=training_test START=24 END=50
# make word2vec_features MODE=training_test START=50 END=100
# make train ALGO=lgbm_classifier
# python src/ensemble/evaluate_ensemble.py
# python src/ensemble/evaluate_ensemble_multi_retrieval.py
# python src/ensemble/tuning_ensemble.py
# make train ALGO=lgbm_ranker
# make candidate_list_eval MODE=training_test
# make train ALGO=lgbm_classifier
# python src/ensemble/score_ensemble_multi_retrieval.py

# make one_ranker_dataset
# make train_one_ranker ALGO=cat_ranker
# make train_one_ranker ALGO=lgbm_ranker
# make train ALGO=lgbm_ranker
# python src/training/train_one_ranker.py --event all --n 3 --algo lgbm_ranker --week w2 --eval 1


# make eda EDA_MODE=class_dist

# python src/training/train.py --event orders --n 1 --algo lgbm_classifier --week w2 --eval 0
# python src/training/train.py --event carts --n 1 --algo lgbm_classifier --week w2 --eval 0
# python src/training/train.py --event clicks --n 1 --algo lgbm_classifier --week w2 --eval 0
# make score_and_eval
# make train ALGO=lgbm_ranker
# make train ALGO=lgbm_classifier
# make train ALGO=cat_classifier
# make train ALGO=cat_ranker


# # # LB 0.564 CV 0.562 Fea 99
# CLICK_MODEL="2022-12-10_clicks_lgbm_classifier_20805_87570"
# CART_MODEL="2022-12-10_carts_lgbm_classifier_34309_89125"
# ORDER_MODEL="2022-12-10_orders_lgbm_classifier_67461_95720"
# WEEK_DATA=w1

# # TRAINING EVALUATION
# CLICK_MODEL="2022-12-20_clicks_cat_ranker_42966_88609"
# CART_MODEL="2022-12-20_carts_cat_ranker_62107_93343"
# ORDER_MODEL="2022-12-20_orders_cat_ranker_78471_96474"
# WEEK_DATA=w2

# python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL

# TRAINING EVALUATION
CLICK_MODEL="2023-01-12_clicks_cat_ranker_57782_89784"
CART_MODEL="2023-01-12_carts_cat_ranker_72870_93555"
ORDER_MODEL="2023-01-11_orders_cat_ranker_85306_96584"
WEEK_DATA=w2

# python src/training/export_treelite.py --event orders --week_model w2 --artifact $ORDER_MODEL
# python src/training/export_treelite.py --event carts --week_model w2 --artifact $CART_MODEL
# python src/training/export_treelite.py --event clicks --week_model w2 --artifact $CLICK_MODEL

# python src/scoring/score_treelite.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score_treelite.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score_treelite.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # # python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# # # python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL

# TRAINING EVALUATION
# CLICK_MODEL="2023-01-06_clicks_cat_ranker_60730_91151"
# CART_MODEL="2023-01-05_carts_cat_ranker_75642_94553"
# ORDER_MODEL="2023-01-05_orders_cat_ranker_86561_97213"
# WEEK_DATA=w2

# {
#     "cart_model": "2023-01-05_carts_cat_ranker_75642_94553",
#     "carts_gt@20": "576482",
#     "carts_hits@20": "241808",
#     "carts_recall@20": "0.4194545536547542",
#     "click_model": "2023-01-06_clicks_cat_ranker_60730_91151",
#     "clicks_gt@20": "1755534",
#     "clicks_hits@20": "930925",
#     "clicks_recall@20": "0.5302802452131374",
#     "order_model": "2023-01-05_orders_cat_ranker_86561_97213",
#     "orders_gt@20": "313303",
#     "orders_hits@20": "204165",
#     "orders_recall@20": "0.6516535111377804",
#     "overall_recall@20": "0.5698564973004082"
# }

# python src/scoring/score_one_ranker.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score_one_ranker.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score_one_ranker.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL

# make train ALGO=lgbm_classifier
# make train ALGO=cat_ranker
# # TRAINING EVALUATION
# CLICK_MODEL="2022-12-15_one_ranker_cat_ranker_70979_50246_44990"
# CART_MODEL="2022-12-15_one_ranker_cat_ranker_70979_50246_44990"
# ORDER_MODEL="2022-12-15_one_ranker_cat_ranker_70979_50246_44990"
# WEEK_DATA=w2

# python src/scoring/score_one_ranker.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score_one_ranker.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score_one_ranker.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL


# IDEA ADDITIONAL FEATURE ENGINEERING
# Item-timeseries features avg/std_num_click/cart/order_D2/3/7 -> relative deviation
# Item-timeseries features difference between D1 with avg_num_click/cart/order_D7 -> relative difference
# target encoding in category: rank_combined, retrieval combined

# current best 0.5674039950748497
