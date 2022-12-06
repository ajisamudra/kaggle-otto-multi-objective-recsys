# take last 1 week data from the whole train data
# python src/preprocess/split_local_train_label.py --mode training
# python src/preprocess/split_local_train_label.py --mode scoring

# # split data last 1 week into 10 chunks
# # python src/preprocess/split_data_into_chunks.py --mode scoring_train
# # python src/preprocess/split_data_into_chunks.py --mode scoring_test
python src/preprocess/split_data_into_chunks.py --mode training_train
python src/preprocess/split_data_into_chunks.py --mode training_test

# # # generate candidates from retrieval
# # # python src/pipeline/make_candidates_list.py --mode scoring_train
# # # python src/pipeline/make_candidates_list.py --mode scoring_test
python src/pipeline/make_candidates_list.py --mode training_train
# python src/pipeline/make_candidates_list.py --mode training_test

# # # pivot candidate list to candidate rows
# # # python src/pipeline/make_candidates_rows.py --mode scoring_train
# # python src/pipeline/make_candidates_rows.py --mode scoring_test
# python src/pipeline/make_candidates_rows.py --mode training_test
python src/pipeline/make_candidates_rows.py --mode training_train

# # # session_features
# python src/features/make_session_features.py --mode training_train
# python src/features/make_session_features.py --mode training_test
# # python src/features/make_session_features.py --mode scoring_test
# # # python src/features/make_session_features.py --mode scoring_train

# # # interaction between session & item features
# python src/features/make_session_item_features.py --mode training_train
# python src/features/make_session_item_features.py --mode training_test
# # python src/features/make_session_item_features.py --mode scoring_test
# # # python src/features/make_session_item_features.py --mode scoring_train

# # # item features
# python src/features/make_item_features.py --mode training_train
# python src/features/make_item_features.py --mode training_test
# # python src/features/make_item_features.py --mode scoring_test
# # python src/features/make_item_features.py --mode scoring_train

# # # item-hour features
# python src/features/make_item_hour_features.py --mode training_train
# # python src/features/make_item_hour_features.py --mode training_test
# # python src/features/make_item_hour_features.py --mode scoring_test
# # python src/features/make_item_hour_features.py --mode scoring_train

# # # item-weekday features
# python src/features/make_item_weekday_features.py --mode training_train
# # python src/features/make_item_weekday_features.py --mode training_test
# # python src/features/make_item_weekday_features.py --mode scoring_test
# # python src/features/make_item_weekday_features.py --mode scoring_train

# # # combine features
python src/pipeline/make_combine_features.py --mode training_train
# python src/pipeline/make_combine_features.py --mode training_test
# # python src/pipeline/make_combine_features.py --mode scoring_test
# # python src/pipeline/make_combine_features.py --mode scoring_train

# # # # perform training
# # # # python src/training/train.py --event orders
# python src/training/train.py --event all --n 10 --algo lgbm --week w2
python src/training/train.py --event all --n 1 --algo cat_ranker --week w2

# CLICK_MODEL="2022-12-06_clicks_lgbm_40526_78660"
# CART_MODEL="2022-12-06_carts_lgbm_27386_78464"
# ORDER_MODEL="2022-12-06_orders_lgbm_45686_85908"
# # perform scoring
# # week_data w2 means local validation
# # week_data w1 means real test submission
# python src/scoring/score.py --event orders --week_data w2 --week_model w2 --artifact $ORDER_MODEL
# python src/scoring/score.py --event carts --week_data w2 --week_model w2 --artifact $CART_MODEL
# python src/scoring/score.py --event clicks --week_data w2 --week_model w2 --artifact $CLICK_MODEL

# # make submission
# python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data w2 --week_model w2

# # eval submission, only for week_data & week_model w2
# python src/scoring/eval_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL


# idea: item & day/hour popularity features
# idea: word2vec for click / cart / order separately -> to recommend separately


# N 10 LGBM
# CLICK_MODEL="2022-12-05_clicks_lgbm_37238_73833"
# CART_MODEL="2022-12-05_carts_lgbm_22652_73159"
# ORDER_MODEL="2022-12-05_orders_lgbm_45132_83005"
# [2022-12-05 20:01:59,291] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.42058256917838105
# [2022-12-05 20:01:59,294] {submission_evaluation.py:84} INFO - clicks hits@20 = 738347 / gt@20 = 1755534
# [2022-12-05 20:01:59,294] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.42058256917838105
# [2022-12-05 20:02:03,773] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.47881529589379823
# [2022-12-05 20:02:03,773] {submission_evaluation.py:84} INFO - carts hits@20 = 197914 / gt@20 = 576482
# [2022-12-05 20:02:03,773] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3433134078774359
# [2022-12-05 20:02:05,524] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.6915284696114344
# [2022-12-05 20:02:05,524] {submission_evaluation.py:84} INFO - orders hits@20 = 191752 / gt@20 = 313303
# [2022-12-05 20:02:05,524] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6120337181578217
# [2022-12-05 20:02:05,524] {submission_evaluation.py:87} INFO - =============
# [2022-12-05 20:02:05,524] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5122725101757619

# N 1 LGBM the same as N 10 LGBM
# CLICK_MODEL="2022-12-05_clicks_lgbm_36860_73857"
# CART_MODEL="2022-12-05_carts_lgbm_22488_72684"
# ORDER_MODEL="2022-12-05_orders_lgbm_44683_82685"
# [2022-12-05 21:24:12,629] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.41965407676524635
# [2022-12-05 21:24:12,631] {submission_evaluation.py:84} INFO - clicks hits@20 = 736717 / gt@20 = 1755534
# [2022-12-05 21:24:12,631] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.41965407676524635
# [2022-12-05 21:24:17,879] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4765887268409194
# [2022-12-05 21:24:17,879] {submission_evaluation.py:84} INFO - carts hits@20 = 196783 / gt@20 = 576482
# [2022-12-05 21:24:17,879] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3413515079395367
# [2022-12-05 21:24:19,641] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.6893471409414995
# [2022-12-05 21:24:19,641] {submission_evaluation.py:84} INFO - orders hits@20 = 190936 / gt@20 = 313303
# [2022-12-05 21:24:19,642] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6094292107001849
# [2022-12-05 21:24:19,642] {submission_evaluation.py:87} INFO - =============
# [2022-12-05 21:24:19,642] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5100283864784966

# N1 LGBM with item, item-hour, item-weekday features
# CLICK_MODEL="2022-12-05_clicks_lgbm_39950_79014"
# CART_MODEL="2022-12-05_carts_lgbm_27192_78112"
# ORDER_MODEL="2022-12-05_orders_lgbm_45734_85740"
# [2022-12-05 23:58:29,254] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.4690880381695826
# [2022-12-05 23:58:29,254] {submission_evaluation.py:84} INFO - clicks hits@20 = 823500 / gt@20 = 1755534
# [2022-12-05 23:58:29,254] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.4690880381695826
# [2022-12-05 23:58:34,875] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.49935886166227206
# [2022-12-05 23:58:34,875] {submission_evaluation.py:84} INFO - carts hits@20 = 208033 / gt@20 = 576482
# [2022-12-05 23:58:34,875] {submission_evaluation.py:85} INFO - carts recall@20 = 0.36086642774622624
# [2022-12-05 23:58:36,796] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.7015113073088604
# [2022-12-05 23:58:36,796] {submission_evaluation.py:84} INFO - orders hits@20 = 193869 / gt@20 = 313303
# [2022-12-05 23:58:36,796] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6187907552752447
# [2022-12-05 23:58:36,796] {submission_evaluation.py:87} INFO - =============
# [2022-12-05 23:58:36,796] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.526443185305973
# [2022-12-05 23:58:36,796] {submission_evaluation.py:89} INFO - =============

# N10 LGBM with item, item-hour, item-weekday features + non-stratified split chunks
# CLICK_MODEL="2022-12-06_clicks_lgbm_40526_78660"
# CART_MODEL="2022-12-06_carts_lgbm_27386_78464"
# ORDER_MODEL="2022-12-06_orders_lgbm_45686_85908"
# [2022-12-06 09:18:23,555] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.47210706258038865
# [2022-12-06 09:18:23,558] {submission_evaluation.py:84} INFO - clicks hits@20 = 828800 / gt@20 = 1755534
# [2022-12-06 09:18:23,558] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.47210706258038865
# [2022-12-06 09:18:28,058] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5015614894871938
# [2022-12-06 09:18:28,059] {submission_evaluation.py:84} INFO - carts hits@20 = 208795 / gt@20 = 576482
# [2022-12-06 09:18:28,059] {submission_evaluation.py:85} INFO - carts recall@20 = 0.362188238314466
# [2022-12-06 09:18:29,823] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.7067686545722155
# [2022-12-06 09:18:29,823] {submission_evaluation.py:84} INFO - orders hits@20 = 194943 / gt@20 = 313303
# [2022-12-06 09:18:29,823] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6222187467084579
# [2022-12-06 09:18:29,823] {submission_evaluation.py:87} INFO - =============
# [2022-12-06 09:18:29,823] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5291984257774534
# [2022-12-06 09:18:29,823] {submission_evaluation.py:89} INFO - =============

# N 1 CATBOOST
# CLICK_MODEL="2022-12-05_clicks_catboost_39780_75037"
# CART_MODEL="2022-12-05_carts_catboost_23530_73139"
# ORDER_MODEL="2022-12-05_orders_catboost_46313_82997
# 022-12-05 21:01:03,182] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.4200704742830387
# [2022-12-05 21:01:03,183] {submission_evaluation.py:84} INFO - clicks hits@20 = 737448 / gt@20 = 1755534
# [2022-12-05 21:01:03,183] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.4200704742830387
# [2022-12-05 21:01:08,031] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4751405723864572
# [2022-12-05 21:01:08,031] {submission_evaluation.py:84} INFO - carts hits@20 = 196100 / gt@20 = 576482
# [2022-12-05 21:01:08,031] {submission_evaluation.py:85} INFO - carts recall@20 = 0.34016673547482834
# [2022-12-05 21:01:09,920] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.6888830242976132
# [2022-12-05 21:01:09,920] {submission_evaluation.py:84} INFO - orders hits@20 = 190833 / gt@20 = 313303
# [2022-12-05 21:01:09,920] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6091004554696252
# [2022-12-05 21:01:09,920] {submission_evaluation.py:87} INFO - =============
# [2022-12-05 21:01:09,920] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5095173413525275
# [2022-12-05 21:01:09,920] {submission_evaluation.py:89} INFO - =============


# N 10 CATBOOST
# CLICK_MODEL="2022-12-05_clicks_catboost_40009_74982"
# CART_MODEL="2022-12-05_carts_catboost_23706_73393"
# ORDER_MODEL="2022-12-05_orders_catboost_46214_83065"
# [2022-12-05 22:49:39,237] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.420984156387743
# [2022-12-05 22:49:39,240] {submission_evaluation.py:84} INFO - clicks hits@20 = 739052 / gt@20 = 1755534
# [2022-12-05 22:49:39,241] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.420984156387743
# [2022-12-05 22:49:43,715] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.47834687765352374
# [2022-12-05 22:49:43,716] {submission_evaluation.py:84} INFO - carts hits@20 = 197574 / gt@20 = 576482
# [2022-12-05 22:49:43,716] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3427236236343893
# [2022-12-05 22:49:45,513] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.6914299509381677
# [2022-12-05 22:49:45,513] {submission_evaluation.py:84} INFO - orders hits@20 = 191712 / gt@20 = 313303
# [2022-12-05 22:49:45,513] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6119060462236238
# [2022-12-05 22:49:45,513] {submission_evaluation.py:87} INFO - =============
# [2022-12-05 22:49:45,513] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5120591304632653
# [2022-12-05 22:49:45,513] {submission_evaluation.py:89} INFO - =============
