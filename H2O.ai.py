# My version of the code at this blog post:
# https://towardsdatascience.com/automl-a-tool-to-improve-your-workflow-1a132248371f

import h2o
from h2o.automl import H2OAutoML

h2o.init()

train = h2o.import_file("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")
y = "C1" #e = edible, p = poisonous

# Train AutoML for 10 mins
aml = H2OAutoML(max_runtime_secs=600, seed=1)
aml.train(y=y, training_frame=train)

# Look at Leaderboard (will have more or fewer models, depending on the hardware used)
aml.leaderboard

# model_id                                               auc      logloss    mean_per_class_error         rmse          mse
# ---------------------------------------------------  -----  -----------  ----------------------  -----------  -----------
# StackedEnsemble_BestOfFamily_AutoML_20190612_161331      1  0.00136702                        0  0.00144618   2.09143e-06
# XGBoost_grid_1_AutoML_20190612_161331_model_4            1  0.000613344                       0  0.00231538   5.361e-06
# GBM_5_AutoML_20190612_161331                             1  1.46072e-13                       0  1.62172e-12  2.62997e-24
# GBM_3_AutoML_20190612_161331                             1  1.07278e-16                       0  6.88271e-15  4.73717e-29
# GBM_grid_1_AutoML_20190612_161331_model_1                1  0.000462979                       0  0.000671548  4.50977e-07
# StackedEnsemble_AllModels_AutoML_20190612_161331         1  0.000839458                       0  0.000880344  7.75006e-07
# GBM_2_AutoML_20190612_161331                             1  2.06903e-17                       0  1.05991e-15  1.1234e-30
# XGBoost_grid_1_AutoML_20190612_161331_model_1            1  0.000164531                       0  0.000872258  7.60834e-07
# XGBoost_1_AutoML_20190612_161331                         1  0.00210774                        0  0.0085784    7.3589e-05
# DeepLearning_grid_1_AutoML_20190612_161331_model_2       1  0.000968467                       0  0.0167997    0.00028223
# 
# [26 rows x 6 columns]



# Or if you want to evaluate performance on a test set instead of using leaderboard metrics,
# you can create a test set and do the following:

df = h2o.import_file("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")
y = "C1" #e = edible, p = poisonous
train, test = df.split_frame(ratios=[.8])
aml = H2OAutoML(max_runtime_secs=600, seed=1)
aml.train(y=y, training_frame=train)
perf = aml.leader.model_performance(test)
print(perf)
