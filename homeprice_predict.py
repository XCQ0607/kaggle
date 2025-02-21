import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import OrdinalEncoder

# 读取数据
xy_train = pd.read_csv('data/homeprice_data/train.csv')
x_test = pd.read_csv('data/homeprice_data/test.csv')

# 合并数据
xy_all = pd.concat([xy_train, x_test], axis=0)  # 将训练集和测试集合并为一个数据集

# 识别分类特征
cat_features = xy_all.columns[xy_all.dtypes == 'object']    # 获取所有数据类型为object的列名

# 对分类特征进行编码
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
).set_output(transform="pandas")
xy_all[cat_features] = ordinal_encoder.fit_transform(xy_all[cat_features])

# 分离训练集和测试集
x_test = xy_all[xy_all["SalePrice"].isna()].drop(columns=["SalePrice"]) # 提取测试集数据，并删除SalePrice列
xy_train = xy_all[~xy_all["SalePrice"].isna()]
x_train = xy_train.drop(columns=["SalePrice"])  # 提取训练集数据，并删除SalePrice列
y_train = xy_train["SalePrice"] # 提取训练集的目标变量

# 创建LightGBM回归模型，并调整参数
model = lgb.LGBMRegressor()
model.set_params(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)
#XGBRegressor()
model1 = xgb.XGBRegressor()
model1.set_params(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=40,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)
#CatBoostRegressor()
model2 = cb.CatBoostRegressor()
# model2.set_params(
#     n_estimators=1000,
#     learning_rate=0.05,
#     depth=6,
#     min_child_samples=40,
#     subsample=0.8,
#     colsample_bylevel=0.7,
#     reg_lambda=0.1,
#     random_seed=42,
# )

modelx = model2

modelx.fit(x_train, y_train) # 使用训练集数据进行训练

# 预测结果
y_pred = modelx.predict(x_test)  # 使用测试集数据进行预测

# 构造并保存结果数据框
pd.DataFrame({
    "Id": x_test["Id"],
    "SalePrice": y_pred,
}).to_csv("output/lgb_baseline.csv", index=False)