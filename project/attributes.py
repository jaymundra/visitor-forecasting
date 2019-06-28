import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'tra': pd.read_csv('air_visit_data.csv'),
    'as': pd.read_csv('air_store_info.csv'),
    'hs': pd.read_csv('hpg_store_info.csv'),
    'ar': pd.read_csv('air_reserve.csv'),
    'hr': pd.read_csv('hpg_reserve.csv'),
    'id': pd.read_csv('store_id_relation.csv'),
    'tes': pd.read_csv('sample_submission.csv'),
    'hol': pd.read_csv('date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }
plt.style.use('ggplot')

print("air_visit_data.csv", end=":")
print(data['tra'].shape)

print("air_store_info.csv", end=":")
print(data['as'].shape)

print("air_reserve.csv", end=":")
print(data['ar'].shape)

print("hpg_reserve.csv", end=":")
print(data['hs'].shape)

print("hpg_store_info.csv", end=":")
print(data['hr'].shape)

print("store_id_relation.csv", end=":")
print(data['id'].shape)

print("date_info.csv", end=":")
print(data['hol'].shape)

print("")
print("")

data['ar']['visit_date'] = data['ar']['visit_datetime'].map(lambda x: str(x).split(' ')[0])
data["ar"] = data["ar"].drop("visit_datetime",axis=1)
data["ar"] = data["ar"].drop("reserve_datetime",axis=1)

ar1 = data['ar'].groupby(["air_store_id",'visit_date'],as_index=False)['reserve_visitors'].sum()
# ar1.to_csv("und/121.csv", index=False)
ar1 = pd.merge(ar1, data['id'], how='left', on=['air_store_id'])
# ar1.to_csv("und/ar1.csv", index=False)

tra1 = pd.merge(data['tra'], ar1, how='left', on=['air_store_id','visit_date'])

data['hr']['visit_date'] = data['hr']['visit_datetime'].map(lambda x: str(x).split(' ')[0])
data["hr"] = data["hr"].drop("visit_datetime",axis=1)
data["hr"] = data["hr"].drop("reserve_datetime",axis=1)
hr1 = data['hr'].groupby(["hpg_store_id",'visit_date'],as_index=False)['reserve_visitors'].sum()
tra2 = pd.merge(tra1, hr1, how='left', on=['hpg_store_id','visit_date'])

# tra2.to_csv("und/tra21.csv", index=False)

tra2['reserve_visitors_x'] = tra2['reserve_visitors_x'].fillna(0)
tra2['reserve_visitors_y'] = tra2['reserve_visitors_y'].fillna(0)
tra2['reserve'] = tra2['reserve_visitors_x'] + tra2['reserve_visitors_y']
tra2 = tra2.drop("reserve_visitors_x",axis=1)
tra2 = tra2.drop("reserve_visitors_y",axis=1)
tra2 = tra2.drop("hpg_store_id",axis=1)


a1 = 20.20593126348421 
a2 = 0.4512725804640162 
er = 16.38046401542735
# for i in range(0,tra2['reserve'].shape[0]):
#     if tra2['reserve'][i]!=0:
#         ert = a1 + a2*tra2['reserve'][i]
#         if ert> (tra2['visitors'][i] + er ):
#             tra2['reserve'][i] = 0

tra2['reserve'] = tra2['reserve'].replace(0,-1)

data['tra'] = tra2
# data['tra'].to_csv("und/tra1.csv", index=False, encoding='utf8')

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

# data['tra'].to_csv("und/tra2.csv", index=False, encoding='utf8')

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
# stores.to_csv("und/stores.csv", index=False, encoding='utf8')

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))


lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
train = train.fillna(-1)
test = test.fillna(-1)

corr = train.corr()
sns.heatmap(corr)
plt.show()

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']


train = train.drop("dow",axis=1)
test = test.drop("dow",axis=1)
train = train.fillna(-1)
test = test.fillna(-1)

train = train.drop("mean_visitors",axis=1)
train = train.drop("longitude",axis=1)
train = train.drop("latitude",axis=1)
test = test.drop("longitude",axis=1)
test = test.drop("latitude",axis=1)

train.to_csv("und/Final_data.csv", index=False)
test.to_csv("und/sumbission_file.csv", index=False)
