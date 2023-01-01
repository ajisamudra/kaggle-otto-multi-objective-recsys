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

# #sub6 CV 0.5667144769335394 Fea 172 recall@150 0.6248116 feature wor2vec vect32 LB 0.578
# CLICK_MODEL="2022-12-23_clicks_lgbm_classifier_46496_89386"
# CART_MODEL="2022-12-23_carts_lgbm_classifier_63946_93662"
# ORDER_MODEL="2022-12-23_orders_lgbm_classifier_79580_96565"

# #sub7 CV 0.5681419681902891 Fea 172 recall@150 0.624811 feature wor2vec vect32 LB 0.579
# CLICK_MODEL="2022-12-23_clicks_cat_ranker_43213_88370"
# CART_MODEL="2022-12-23_carts_cat_ranker_62356_93099"
# ORDER_MODEL="2022-12-23_orders_cat_ranker_78326_96000"


#sub8 CV 0.5687338005167984 Fea 172 recall@150 0.6185134 feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-25_clicks_cat_ranker_45439_89900"
CART_MODEL="2022-12-25_carts_cat_ranker_65389_94260"
ORDER_MODEL="2022-12-25_orders_cat_ranker_80132_96912"
WEEK_DATA=w1

# # perform scoring
python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2


# sub9 CV 0.567130953598196 Fea 172 recall@150 0.6185134 feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-25_clicks_lgbm_classifier_48807_90926"
CART_MODEL="2022-12-25_carts_lgbm_classifier_66769_94771"
ORDER_MODEL="2022-12-25_orders_lgbm_classifier_81484_97385"
WEEK_DATA=w1

# # perform scoring
python src/scoring/score_treelite.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score_treelite.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score_treelite.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

#sub10 CV 0.5662789963494648 Fea 172 recall@150 0.6185134 feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-26_clicks_cat_classifier_49243_90940"
CART_MODEL="2022-12-26_carts_cat_classifier_67369_94709"
ORDER_MODEL="2022-12-26_orders_cat_classifier_81621_97386"
WEEK_DATA=w1

# # perform scoring
python src/scoring/score.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

#sub11 CV 0.5672425543156044 Fea 172 recall@150 0.6185134 feature wor2vec vect32 LB XXX
CLICK_MODEL="2022-12-26_clicks_lgbm_ranker_48370_89600"
CART_MODEL="2022-12-26_carts_lgbm_ranker_66196_93867"
ORDER_MODEL="2022-12-26_orders_lgbm_ranker_78372_95814"
WEEK_DATA=w1

# # perform scoring
python src/scoring/score_treelite.py --event orders --week_data $WEEK_DATA --week_model w2 --artifact $ORDER_MODEL
python src/scoring/score_treelite.py --event carts --week_data $WEEK_DATA --week_model w2 --artifact $CART_MODEL
python src/scoring/score_treelite.py --event clicks --week_data $WEEK_DATA --week_model w2 --artifact $CLICK_MODEL

# make submission
python src/scoring/make_submission.py --click_model $CLICK_MODEL --cart_model $CART_MODEL --order_model $ORDER_MODEL --week_data $WEEK_DATA --week_model w2

# Recall@170 0.626504 (n_cand 130) fea 142 without word2vec/fasttext/matrix fact distances
# cat_ranker overall recall@20 0.569340758508474
# CLICK_MODEL="2023-01-01_clicks_cat_ranker_60360_91190"
# CART_MODEL="2023-01-01_carts_cat_ranker_74769_94383"
# ORDER_MODEL="2023-01-01_orders_cat_ranker_86250_97127"

# Recall@170 0.626504 (n_cand 130) fea 144 without word2vec/fasttext/matrix fact distances addd rank_combined & retrieval_combined
# cat_ranker overall recall@20 0.5694135088803328
# CLICK_MODEL="2023-01-01_clicks_cat_ranker_60425_91197"
# CART_MODEL="2023-01-01_carts_cat_ranker_74776_94381"
# ORDER_MODEL="2023-01-01_orders_cat_ranker_86246_97127"
