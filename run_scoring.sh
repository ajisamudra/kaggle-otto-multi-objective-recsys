# ## SCORING
# # CV 0.563701 Fea 142
# CLICK_MODEL="2022-12-14_clicks_lgbm_classifier_34385_85767"
# CART_MODEL="2022-12-14_carts_lgbm_classifier_53168_90510"
# ORDER_MODEL="2022-12-14_orders_lgbm_classifier_71862_94801"
# # CV 0.5631726827117819 Fea 142 LB 0.574
# CLICK_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"
# CART_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"
# ORDER_MODEL="2022-12-15_one_ranker_lgbm_ranker_73223_42030_40696"

# # CV 0.5660705273474493 Fea 152 LB XX
# CLICK_MODEL="2022-12-17_clicks_cat_ranker_33287_85844"
# CART_MODEL="2022-12-17_carts_cat_ranker_54090_90869"
# ORDER_MODEL="2022-12-17_orders_cat_ranker_72338_94748"

# CV 0.5681765240808944 Fea 172 LB XX
CLICK_MODEL="2022-12-18_clicks_cat_ranker_44864_89643"
CART_MODEL="2022-12-18_carts_cat_ranker_63198_93923"
ORDER_MODEL="2022-12-18_orders_cat_ranker_79367_96784"
WEEK_DATA=w1

# # perform scoring
# # week_data w2 means local validation
# # week_data w1 means real test submission
python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2
