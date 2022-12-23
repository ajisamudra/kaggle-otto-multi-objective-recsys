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

# #sub1 CV 0.5681765240808944 Fea 172 recall@120 0.60955 LB 0.579
# CLICK_MODEL="2022-12-18_clicks_cat_ranker_44864_89643"
# CART_MODEL="2022-12-18_carts_cat_ranker_63198_93923"
# ORDER_MODEL="2022-12-18_orders_cat_ranker_79367_96784"

# sub2 CV 0.5686940699602498 Fea 172 recall@120 0.61464 LB 0.579 (better than sub1)
# CLICK_MODEL="2022-12-20_clicks_cat_ranker_42966_88609"
# CART_MODEL="2022-12-20_carts_cat_ranker_62107_93343"
# ORDER_MODEL="2022-12-20_orders_cat_ranker_78471_96474"

# #sub3 CV 0.5670441199234523 Fea 172 recall@120 0.61464 LB 0.579 (better than sub1 sub2) feature wor2vec vect32
# CLICK_MODEL="2022-12-20_clicks_lgbm_classifier_46744_89740"
# CART_MODEL="2022-12-20_carts_lgbm_classifier_63895_93925"
# ORDER_MODEL="2022-12-20_orders_lgbm_classifier_79792_97018"

# #sub4 CV 0.566384332228243 Fea 172 recall@120 0.621 feature wor2vec vect64 LB 0.578
# CLICK_MODEL="2022-12-21_clicks_lgbm_classifier_44528_88096"
# CART_MODEL="2022-12-21_carts_lgbm_classifier_61299_92797"
# ORDER_MODEL="2022-12-21_orders_lgbm_classifier_78722_96235"

# #sub5 CV 0.5679821898665464 Fea 172 recall@120 0.621 feature wor2vec vect64 LB 0.579
# CLICK_MODEL="2022-12-21_clicks_cat_ranker_41346_87071"
# CART_MODEL="2022-12-21_carts_cat_ranker_59583_92239"
# ORDER_MODEL="2022-12-21_orders_cat_ranker_77498_95714"

#sub6 CV 0.5667144769335394 Fea 172 recall@120 0.6248116 feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-23_clicks_lgbm_classifier_46496_89386"
CART_MODEL="2022-12-23_carts_lgbm_classifier_63946_93662"
ORDER_MODEL="2022-12-23_orders_lgbm_classifier_79580_96565"
WEEK_DATA=w1

# # perform scoring
# # week_data w2 means local validation
# # week_data w1 means real test submission
python src/scoring/score_treelite.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score_treelite.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score_treelite.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

#sub7 CV 0.5681419681902891 Fea 172 recall@150 0.624811feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-23_clicks_cat_ranker_43213_88370"
CART_MODEL="2022-12-23_carts_cat_ranker_62356_93099"
ORDER_MODEL="2022-12-23_orders_cat_ranker_78326_96000"
WEEK_DATA=w1

# # perform scoring
# # week_data w2 means local validation
# # week_data w1 means real test submission
python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2
