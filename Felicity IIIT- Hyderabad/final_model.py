import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, '/Users/ebby/Documents/AnalyticsVidya/untitled folder')
sub_dir = os.path.join(root_dir, '/Users/ebby/Documents/AnalyticsVidya/untitled folder')

train9 = pd.read_csv(os.path.join(data_dir, 'train_DcAf1c7', 'train9.csv'))
train1 = pd.read_csv(os.path.join(data_dir, 'train_DcAf1c7', 'train1.csv'))
hero_data = pd.read_csv(os.path.join(data_dir, 'train_DcAf1c7', 'hero_data.csv'))

test9 = pd.read_csv(os.path.join(data_dir, 'test_rHx1itc','test9.csv'))
test1 = pd.read_csv(os.path.join(data_dir, 'test_rHx1itc','test1.csv'))

train9 = pd.merge(train9, hero_data, on='hero_id')
train1 = pd.merge(train1, hero_data, on='hero_id')
test9 = pd.merge(test9, hero_data, on='hero_id')
test1 = pd.merge(test1, hero_data, on='hero_id')

train1["is_ten"]=1
train9["is_ten"]=0
test1["is_ten"]=1
test9["is_ten"]=0

nominal_columns = ["primary_attr",'attack_type']
dummy_df = pd.get_dummies(train1[nominal_columns])
train1 = pd.concat([train1, dummy_df], axis=1)
train1 = train1.drop(nominal_columns, axis=1)

dummy_df = pd.get_dummies(test1[nominal_columns])
test1 = pd.concat([test1, dummy_df], axis=1)
test1 = test1.drop(nominal_columns, axis=1)

dummy_df = pd.get_dummies(train9[nominal_columns])
train9 = pd.concat([train9, dummy_df], axis=1)
train9 = train9.drop(nominal_columns, axis=1)

dummy_df = pd.get_dummies(test9[nominal_columns])
test9 = pd.concat([test9, dummy_df], axis=1)
test9 = test9.drop(nominal_columns, axis=1)

train=pd.concat([train1,train9,test9], ignore_index=True)
test=test1
trainroles = pd.DataFrame(train["roles"])
testroles = pd.DataFrame(test["roles"])
train.drop(["roles"],inplace=True,axis=1)
test.drop(["roles"],inplace=True,axis=1)

X_train1 = train.drop(["id","kda_ratio","num_wins"], axis=1)
Y_train1 = train["num_wins"]
X_test1  = test.drop(["id"], axis=1).copy()

import xgboost as xgb
from sklearn.metrics import mean_squared_error
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05, 
                             learning_rate=0.08, max_depth=3, base_score=0.6,
                             min_child_weight=10, n_estimators=6000,
                             reg_alpha=0.5, reg_lambda=0.9,
                             subsample=0.5213, silent=1,)

model_xgb.fit(X_train1,Y_train1)
pred_test_y = model_xgb.predict(X_test1)

test["num_wins"]=pred_test_y
test["num_wins"]=test["num_wins"].astype(int)
train["roles1"]=trainroles["roles"]
test["roles1"]=testroles["roles"]

def getPurchaseVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
        min_dict = {}
        max_dict = {}
        mean_dict = {}
        twentyfive_dict = {}
        seventyfive_dict = {}
        for name, group in grouped_df:
                min_dict[name] = min(np.array(group["kda_ratio"]))
                max_dict[name] = max(np.array(group["kda_ratio"]))
                mean_dict[name] = np.mean(np.array(group["kda_ratio"]))
                twentyfive_dict[name] = np.percentile(np.array(group["kda_ratio"]),25)
                seventyfive_dict[name] = np.percentile(np.array(group["kda_ratio"]),75)

        min_list = []
        max_list = []
        mean_list = []
        twentyfive_list = []
        seventyfive_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                min_list.append(min_dict.get(name,0))
                max_list.append(max_dict.get(name,0))
                mean_list.append(mean_dict.get(name,0))
                twentyfive_list.append( twentyfive_dict.get(name,0))
                seventyfive_list.append( seventyfive_dict.get(name,0))

        return min_list, max_list, mean_list, twentyfive_list, seventyfive_list

min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(train, train, "user_id")
train["kda_ratio_Minuser"] = min_price_list
train["kda_ratio_Maxuser"] = max_price_list
train["kda_ratio_Meanuser"] = mean_price_list
train["kda_ratio_25Percuser"] = twentyfive_price_list
train["kda_ratio_75Percuser"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(test, train, "user_id")
test["kda_ratio_Minuser"] = min_price_list
test["kda_ratio_Maxuser"] = max_price_list
test["kda_ratio_Meanuser"] = mean_price_list
test["kda_ratio_25Percuser"] = twentyfive_price_list
test["kda_ratio_75Percuser"] = seventyfive_price_list

min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(train, train, "hero_id")
train["kda_ratio_Min_hero_id"] = min_price_list
train["kda_ratio_Max_hero_id"] = max_price_list
train["kda_ratio_Mean_hero_id"] = mean_price_list
train["kda_ratio_25Perc_hero_id"] = twentyfive_price_list
train["kda_ratio_75Perc_hero_id"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(test, train, "hero_id")
test["kda_ratio_Min_hero_id"] = min_price_list
test["kda_ratio_Max_hero_id"] = max_price_list
test["kda_ratio_Mean_hero_id"] = mean_price_list
test["kda_ratio_25Perc_hero_id"] = twentyfive_price_list
test["kda_ratio_75Perc_hero_id"] = seventyfive_price_list

min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(train, train, "roles1")
train["kda_ratio_Min_roles"] = min_price_list
train["kda_ratio_Max_roles"] = max_price_list
train["kda_ratio_Mean_roles"] = mean_price_list
train["kda_ratio_25Perc_roles"] = twentyfive_price_list
train["kda_ratio_75Perc_roles"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list, seventyfive_price_list = getPurchaseVar(test, train, "roles1")
test["kda_ratio_Min_roles"] = min_price_list
test["kda_ratio_Max_roles"] = max_price_list
test["kda_ratio_Mean_roles"] = mean_price_list
test["kda_ratio_25Perc_roles"] = twentyfive_price_list
test["kda_ratio_75Perc_roles"] = seventyfive_price_list

train.drop(["roles1"],inplace=True,axis=1)
test.drop(["roles1"],inplace=True,axis=1)
train.sort_index(axis=1, inplace=True)
test.sort_index(axis=1,inplace=True)

X_train1 = train.drop(["id","kda_ratio"], axis=1)
Y_train1 = train["kda_ratio"]
X_test1  = test.drop(["id"], axis=1).copy()


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.07,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=10, min_samples_split=10, 
                                   loss='huber', random_state =5)

GBoost.fit(X_train1,Y_train1)
pred_test_y = GBoost.predict(X_test1)

submission = pd.DataFrame({
        "id": test["id"],
        "kda_ratio": pred_test_y
    })
submission.to_csv("submission.csv",sep=',',index=False)