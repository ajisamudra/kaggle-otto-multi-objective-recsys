# make split_into_chunks MODE=training_train
# make candidate_list MODE=training_train
# make candidate_word2vec_list MODE=training_train
make candidate_rows MODE=training_train
# make session_features MODE=training_train
# make item_features MODE=training_train
# make session_item_features MODE=training_train
# make item_hour_features MODE=training_train
# make item_weekday_features MODE=training_train
# make session_representation_items MODE=training_train
# make item_covisitation_features MODE=training_train START=0 END=30
# make matrix_factorization_features MODE=training_train START=0 END=30
# make combine_features  MODE=training_train START=0 END=30


# make split_into_chunks MODE=training_test
# make candidate_list MODE=training_test
# make candidate_word2vec_list MODE=training_test
# make candidate_rows MODE=training_test
# make session_features MODE=training_test
# make item_features MODE=training_test
# make session_item_features MODE=training_test
# make item_hour_features MODE=training_test
# make item_weekday_features MODE=training_test
# make session_representation_items MODE=training_test
# make item_covisitation_features MODE=training_test START=0 END=40
# make item_covisitation_features MODE=training_test START=40 END=80
# make matrix_factorization_features MODE=training_test START=0 END=20
# make matrix_factorization_features MODE=training_test START=20 END=40
# make matrix_factorization_features MODE=training_test START=40 END=60
# make matrix_factorization_features MODE=training_test START=60 END=80
# make combine_features  MODE=training_test START=0 END=80

# make split_into_chunks MODE=scoring_test
# make candidate_list MODE=scoring_test
# make candidate_word2vec_list MODE=scoring_test
# make candidate_rows MODE=scoring_test
# make session_features MODE=scoring_test
# make item_features MODE=scoring_test
# make session_item_features MODE=scoring_test
# make item_hour_features MODE=scoring_test
# make item_weekday_features MODE=scoring_test
# make session_representation_items MODE=scoring_test
# make item_covisitation_features MODE=scoring_test START=0 END=40
# make item_covisitation_features MODE=scoring_test START=40 END=80
# make matrix_factorization_features MODE=scoring_test START=0 END=20
# make matrix_factorization_features MODE=scoring_test START=20 END=40
# make matrix_factorization_features MODE=scoring_test START=40 END=60
# make matrix_factorization_features MODE=scoring_test START=60 END=80
# make combine_features  MODE=scoring_test START=0 END=80


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
# CLICK_MODEL="2022-12-14_clicks_lgbm_classifier_34385_85767"
# CART_MODEL="2022-12-14_carts_lgbm_classifier_53168_90510"
# ORDER_MODEL="2022-12-14_orders_lgbm_classifier_71862_94801"
# WEEK_DATA=w2

# # python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# # python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL

# # TRAINING EVALUATION
# CLICK_MODEL="2022-12-15_one_ranker_lgbm_ranker_73451_41792_40682"
# CART_MODEL="2022-12-15_one_ranker_lgbm_ranker_73451_41792_40682"
# ORDER_MODEL="2022-12-15_one_ranker_lgbm_ranker_73451_41792_40682"
# WEEK_DATA=w2

# python src/scoring/score_one_ranker.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score_one_ranker.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score_one_ranker.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL

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

# ## SCORING
# # CV 0.563701 Fea 142
# CLICK_MODEL="2022-12-14_clicks_lgbm_classifier_34385_85767"
# CART_MODEL="2022-12-14_carts_lgbm_classifier_53168_90510"
# ORDER_MODEL="2022-12-14_orders_lgbm_classifier_71862_94801"
# # CV 0.5631726827117819 Fea 142 LB 0.574
# CLICK_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"
# CART_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"
# ORDER_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"
# WEEK_DATA=w1

# # # perform scoring
# # # week_data w2 means local validation
# # # week_data w1 means real test submission
# python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
# python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2
