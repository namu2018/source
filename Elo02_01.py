
# coding: utf-8

# ### https://www.kaggle.com/artgor/elo-eda-and-models

# In[1]:

import numpy as np 
import pandas as pd 
import os
# import seaborn as sns 
# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
# plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
import gc
from catboost import CatBoostRegressor

#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)

# import workalendar
# from workalendar.america import Brazil

# In[2]:
train = pd.read_csv("/root/data/Elo/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("/root/data/Elo/test.csv", parse_dates=["first_active_month"])
submission = pd.read_csv('/root/data/Elo/sample_submission.csv')
start = time.time()
print("time : ", time.time())

# 메모리 사용량 줄이기
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# category change
train['feature_1'] = train['feature_1'].astype('category')
train['feature_2'] = train['feature_2'].astype('category')
train['feature_3'] = train['feature_3'].astype('category')
#print( train.head() )
#print( train.info() )

#fig, ax = plt.subplots(1, 3, figsize = (16, 6))
#plt.suptitle('Violineplots for features and target');
#sns.violinplot(x="feature_1", y="target", data=train, ax=ax[0], title='feature_1');
#sns.violinplot(x="feature_2", y="target", data=train, ax=ax[1], title='feature_2');
#sns.violinplot(x="feature_3", y="target", data=train, ax=ax[2], title='feature_3');

fea_sel = ['feature_1', 'feature_2', 'feature_3']
for sel in fea_sel:
    print(train[sel].value_counts())

#fig, ax = plt.subplots(1, 3, figsize = (16, 6));
#train['feature_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='feature_1');
#train['feature_2'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='feature_2');
#train['feature_3'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='gold', title='feature_3');
#plt.suptitle('Counts of categiories for features');

test['feature_1'] = test['feature_1'].astype('category')
test['feature_2'] = test['feature_2'].astype('category')
test['feature_3'] = test['feature_3'].astype('category')

d1 = train['first_active_month'].value_counts().sort_index()
d2 = test['first_active_month'].value_counts().sort_index()

# import plotly.graph_objs as go

#data = [go.Scatter(x=d1.index, y=d1.values, name='train'), 
#        go.Scatter(x=d2.index, y=d2.values, name='test')]

#layout = go.Layout(dict(title = "Counts of first active",
#                  xaxis = dict(title = 'Month'),
#                  yaxis = dict(title = 'Count'),
#                  ),legend=dict(
#                orientation="v"))
#py.iplot(dict(data=data, layout=layout))


# 학습 및 테스트 데이터의 추세 동향은 비슷하며 이는 매우 좋습니다. 왜 그 기간이 끝나면 급격한 감소가 있나? 나는 **그것이 의도적으로 생각**한다. 또는 새로운 카드는 일부 조건을 충족 한 후에 만 ​​고려 될 수 있습니다.
# 또한 테스트에서 누락 된 데이터가있는 한 줄이 있습니다. 동일한 기능 값을 갖는 첫 번째 데이터로 채울 것입니다

test.loc[test['first_active_month'].isna(), 'first_active_month'] = test.loc[(test['feature_1'] == 5) & (test['feature_2'] == 2) & (test['feature_3'] == 1), 'first_active_month'].min()

print('There are {0} samples with target lower than -20.'.format(train.loc[train.target < -20].shape[0]))
print( train.loc[train.target < -20] )

# And they have 1 unique value: -33.21928095. This seems to be a special case. Maybe it would be reasonable to simply exclude these samples. We'll try later.<br>
# 그리고 그들은 1 고유 값 : -33.21928095를가집니다. 이것은 특별한 경우 인 것 같습니다. 아마 이러한 샘플을 제외하는 것이 합리적 일 수 있습니다. 우리는 나중에 시도 할 것이다.

# ### Feature engineering
print("[Feature engineering] time : ", time.time() - start)
# In[18]:

max_date = train['first_active_month'].dt.date.max()
def process_main(df):
    date_parts = ["year", "weekday", "month"]
    for part in date_parts:
        part_col = 'first_active_month' + "_" + part
        df[part_col] = getattr(df['first_active_month'].dt, part).astype(int)
            
    df['elapsed_time'] = (max_date - df['first_active_month'].dt.date).dt.days
    
    return df

print("[] time : ", time.time() - start)


print("[19] time : ", time.time() - start)
train = process_main(train)
test = process_main(test)

print("process_main :")
print(train.columns)
print(test.columns)

# ### historical_transactions
# * 각각의 card_id의 지난 데이터 가져오기
# #### Up to 3 months' worth of historical transactions for each card_id

historical_transactions = pd.read_csv('/root/data/Elo/historical_transactions.csv')

print('{} samples in data'.format(historical_transactions.shape[0]))
print(historical_transactions.head())

# let's convert the authorized_flag to a binary value.
historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)

print("At average {:.4f}% transactions are authorized".format(historical_transactions['authorized_flag'].mean() * 100))
# historical_transactions['authorized_flag'].value_counts().plot(kind='barh', title='authorized_flag value counts');


# 승인 된 거래 비율이 가장 낮은 카드 및 가장 높은 카드

autorized_card_rate = historical_transactions.groupby(['card_id'])['authorized_flag'].mean().sort_values()


# 대부분의 거래가 거부 된 카드가있는 것으로 보입니다. 이 사기 거래가 있었습니까?

# ### installments
print ( historical_transactions['installments'].value_counts() )


# #### 흥미롭군. 가장 일반적인 설치 수는 0과 1로 예상됩니다. 그러나 -1과 999는 이상합니다. 나는이 값들이 누락 된 값을 채우기 위해 사용되었다고 생각한다.

print ( historical_transactions.groupby(['installments'])['authorized_flag'].mean() )

# ### 반면에 999는 이러한 거래 중 3 %만이 승인 된 것으로 간주되어 사기 거래를 의미 할 수 있습니다. 한 가지 흥미로운 점은 할부 수가 많을수록 승인 비율이 낮다는 것입니다.

historical_transactions['installments'] = historical_transactions['installments'].astype('category')

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])


# ### purchase_amount

# ### 슬프게도 purchase_amount가 정규화되었습니다. 그럼에도 불구하고 그것을 보자.

# In[26]:

for i in [-1, 0]:
    n = historical_transactions.loc[historical_transactions['purchase_amount'] < i].shape[0]
    print("There are {} transactions with purchase_amount less than {}.".format(n,i))
for i in [0, 10, 100]:
    n = historical_transactions.loc[historical_transactions['purchase_amount'] > i].shape[0]
    print("There are {} transactions with purchase_amount more than {}.".format(n,i))

#plt.title('Purchase amount distribution for negative values.');
#historical_transactions.loc[historical_transactions['purchase_amount'] < 0, 'purchase_amount'].plot(kind='hist');


# ### 거의 모든 거래가 범위 (-1, 0)의 구매 금액을 가진 것으로 보입니다. 상당히 강한 정규화와 높은 특이 치를 처리해야합니다.
# ### Categories

# In[28]:
map_dict = {'Y': 0, 'N': 1}
historical_transactions['category_1'] = historical_transactions['category_1'].apply(lambda x: map_dict[x])
historical_transactions.groupby(['category_1']).agg({'purchase_amount': ['mean', 'std', 'count'], 'authorized_flag': ['mean', 'std']})

historical_transactions.groupby(['category_2']).agg({'purchase_amount': ['mean', 'std', 'count'], 
                                                     'authorized_flag': ['mean', 'std']})

map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
historical_transactions['category_3'] = historical_transactions['category_3'].apply(lambda x: map_dict[str(x)])
historical_transactions.groupby(['category_3']).agg({'purchase_amount': ['mean', 'std', 'count'], 
                                                     'authorized_flag': ['mean', 'std']})
# ### All categories are quite different

for col in ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']:
    print("There are {} unique values in {}.".format(historical_transactions[col].nunique(), col))

# ### Feature engineering
def aggregate_historical_transactions(trans, prefix):
    # more features from this kernel: https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
    trans['purchase_month'] = trans['purchase_date'].dt.month
#     trans['year'] = trans['purchase_date'].dt.year
#     trans['weekofyear'] = trans['purchase_date'].dt.weekofyear
#     trans['month'] = trans['purchase_date'].dt.month
#     trans['dayofweek'] = trans['purchase_date'].dt.dayofweek
#     trans['weekend'] = (trans.purchase_date.dt.weekday >=5).astype(int)
#     trans['hour'] = trans['purchase_date'].dt.hour
    trans['month_diff'] = ((datetime.datetime.today() - trans['purchase_date']).dt.days)//30
    trans['month_diff'] += trans['month_lag']
    trans['installments'] = trans['installments'].astype(int)

    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).                                         astype(np.int64) * 1e-9
    trans = pd.get_dummies(trans, columns=['category_2', 'category_3'])
    agg_func =( {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean', 'sum'],
        'category_2_2.0': ['mean', 'sum'],
        'category_2_3.0': ['mean', 'sum'],
        'category_2_4.0': ['mean', 'sum'],
        'category_2_5.0': ['mean', 'sum'],
        'category_3_1': ['sum', 'mean'],
        'category_3_2': ['sum', 'mean'],
        'category_3_3': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'month_lag': ['min', 'max'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'city_id': ['nunique'],
        'month_diff': ['min', 'max', 'mean']
    } )
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    return agg_trans

print("[33] time : ", time.time() - start)

def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])
    history['installments'] = history['installments'].astype(int)
    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group

final_group = aggregate_per_month(historical_transactions) 


# In[34]:

del d1, d2, autorized_card_rate
gc.collect()
historical_transactions = reduce_mem_usage(historical_transactions)
history = aggregate_historical_transactions(historical_transactions, prefix='hist_')
history = reduce_mem_usage(history)
gc.collect()

print("[35] time : ", time.time() - start)
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
del history

del historical_transactions
gc.collect()


# ### new_merchant_transactions
# * 기록 데이터에서 방문하지 않은 merchant_ids에서 만든 card_id를 모두 구매 한 각 card_id에 대해 2 개월 분의 데이터가 필요합니다.

print("[37 read_new_merchant_transactions] time : ", time.time() - start)
new_merchant_transactions = pd.read_csv('/root/data/Elo/new_merchant_transactions.csv')


# In[38]:

print('{} samples in data'.format(new_merchant_transactions.shape[0]))
#new_merchant_transactions.head()


# In[39]:

# let's convert the authorized_flag to a binary value.
new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)


# In[40]:

print("At average {:.4f}% transactions are authorized".format(new_merchant_transactions['authorized_flag'].mean() * 100  ))
# new_merchant_transactions['authorized_flag'].value_counts().plot(kind='barh', title='authorized_flag value counts');


# ### 이전 데이터와 달리 모든 거래가 승인되었습니다.

# ### 총 구매 금액이 가장 낮고 가장 높은 카드

card_total_purchase = new_merchant_transactions.groupby(['card_id'])['purchase_amount'].sum().sort_values()
#card_total_purchase.head()
#card_total_purchase.tail()


# ### 대부분의 거래가 거부 된 카드가있는 것으로 보입니다. 이 사기 거래가 있었습니까?

# ### installments
print("installment : ", new_merchant_transactions['installments'].value_counts() )


# ### 흥미 롭 군. 가장 일반적인 설치 수는 0과 1로 예상됩니다. 그러나 -1과 999는 이상합니다. 나는이 값들이 누락 된 값을 채우기 위해 사용되었다고 생각한다.


#new_merchant_transactions.groupby(['installments'])['purchase_amount'].sum()


# In[45]:
print("[45_installments] time : ", time.time() - start)
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype('category')


# ### purchase_amount

# In[46]:

# plt.title('Purchase amount distribution.');
# new_merchant_transactions['purchase_amount'].plot(kind='hist');


# In[47]:

for i in [-1, 0]:
    n = new_merchant_transactions.loc[new_merchant_transactions['purchase_amount'] < i].shape[0]
    print("There are {} transactions with purchase_amount less than {}.".format(n,i))
for i in [0, 10, 100]:
    n = new_merchant_transactions.loc[new_merchant_transactions['purchase_amount'] > i].shape[0]
    print("There are {} transactions with purchase_amount more than {}.".format(n,i))

# 거의 모든 거래가 범위 (-1, 0)의 구매 금액을 가진 것으로 보입니다. 상당히 강한 정규화와 높은 특이 치를 처리해야합니다.

# ### Categories

print("[49] time : ", time.time() - start)

map_dict = {'Y': 0, 'N': 1}
new_merchant_transactions['category_1'] = new_merchant_transactions['category_1'].apply(lambda x: map_dict[x])
new_merchant_transactions.groupby(['category_1']).agg({'purchase_amount': ['mean', 'std', 'count']})

print("[50] time : ", time.time() - start)
# In[50]:
new_merchant_transactions.groupby(['category_2']).agg({'purchase_amount': ['mean', 'std', 'count']})


print("[51] time : ", time.time() - start)

map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].apply(lambda x: map_dict[str(x)])
new_merchant_transactions.groupby(['category_3']).agg({'purchase_amount': ['mean', 'std', 'count']})

for col in ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']:
    print("There are {} unique values in {}".format(new_merchant_transactions[col].nunique(), col))


print("[53] time : ", time.time() - start)

new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])


# In[54]:

def aggregate_historical_transactions(trans, prefix):
    # more features from this kernel: https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
    trans['purchase_month'] = trans['purchase_date'].dt.month
    trans['year'] = trans['purchase_date'].dt.year
    trans['weekofyear'] = trans['purchase_date'].dt.weekofyear
    trans['month'] = trans['purchase_date'].dt.month
    trans['dayofweek'] = trans['purchase_date'].dt.dayofweek
    trans['weekend'] = (trans.purchase_date.dt.weekday >=5).astype(int)
    trans['hour'] = trans['purchase_date'].dt.hour
    trans['installments'] = trans['installments'].astype(int)
    trans['month_diff'] = ((datetime.datetime.today() - trans['purchase_date']).dt.days)//30
    trans['month_diff'] += trans['month_lag']

    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).astype(np.int64) * 1e-9
    trans['installments'] = trans['installments'].astype(int)
    trans = pd.get_dummies(trans, columns=['category_2', 'category_3'])
    agg_func = {
        'category_1': ['sum', 'mean'],        \
        'category_2_1.0': ['mean', 'sum'], \
        'category_2_2.0': ['mean', 'sum'], \
        'category_2_3.0': ['mean', 'sum'], \
        'category_2_4.0': ['mean', 'sum'], \
        'category_2_5.0': ['mean', 'sum'], \
        'category_3_1': ['sum', 'mean'], \
        'category_3_2': ['sum', 'mean'], \
        'category_3_3': ['sum', 'mean'], \
        'merchant_id': ['nunique'], \
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'], \
        'installments': ['sum', 'mean', 'max', 'min', 'std'], \
        'purchase_month': ['mean', 'max', 'min', 'std'], \
        'purchase_date': [np.ptp, 'max', 'min'], \
        'month_lag': ['min', 'max'], \
        'merchant_category_id': ['nunique'], \
        'state_id': ['nunique'], \
        'subsector_id': ['nunique'], \
        'city_id': ['nunique'],
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (  trans.groupby('card_id').size().reset_index(name='{}transactions_count'.format(prefix)))
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    return agg_trans

gc.collect()
new_transactions = reduce_mem_usage(new_merchant_transactions)
history = aggregate_historical_transactions(new_merchant_transactions, prefix='new')
history = reduce_mem_usage(history)
del new_merchant_transactions
gc.collect()
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
del history
gc.collect()


# In[56]:
train = pd.merge(train, final_group, on='card_id')
test = pd.merge(test, final_group, on='card_id')
gc.collect()
del final_group


# ### merchants

# 각 merchant_id에 대한 집계 정보

# In[57]:
print("[57 read_csv(merchants.csv) : time : ", time.time() - start)
merchants = pd.read_csv('/root/data/Elo/merchants.csv')

print("[58] time : ", time.time() - start)
print('{} merchants in data'.format( merchants.shape[0] ))
# merchants.head()

# In[59]:

# encoding categories.
map_dict = {'Y': 0, 'N': 1}
merchants['category_1'] = merchants['category_1'].apply(lambda x: map_dict[x])
merchants.loc[merchants['category_2'].isnull(), 'category_2'] = 0
merchants['category_4'] = merchants['category_4'].apply(lambda x: map_dict[x])


print("[60] time : ", time.time() - start)
merchants['merchant_category_id'].nunique(), merchants['merchant_group_id'].nunique()

# plt.hist(merchants['numerical_1']);
# plt.title('Distribution of numerical_1');


# In[62]:
print("[62] time : ", time.time() - start)
print( np.percentile(merchants['numerical_1'], 95) )


# ### 음, 95 %의 값이 0.1보다 작 으면, 우리는 이상치를 처리해야합니다.

# In[63]:

#plt.hist(merchants.loc[merchants['numerical_1'] < 0.1, 'numerical_1']);
#plt.title('Distribution of numerical_1 less than 0.1');


print("[64] time : ", time.time() - start)
min_n1 = merchants['numerical_1'].min()
_ = sum(merchants['numerical_1'] == min_n1) / merchants['numerical_1'].shape[0]
print('{:.4f}% of values in numerical_1 are equal to {}'.format( _ * 100, min_n1 ) )


# ### 사실 반값 이상은 최소값과 같습니다. 아주 한쪽으로 치우친 분포이다.

# In[65]:

#plt.hist(merchants['numerical_2']);
#plt.title('Distribution of numerical_2');


# In[66]:

#plt.hist(merchants.loc[merchants['numerical_2'] < 0.1, 'numerical_2']);
#plt.title('Distribution of numerical_2 less than 0.1');
min_n1 = merchants['numerical_1'].min()
_ = sum(merchants['numerical_1'] == min_n1) / merchants['numerical_1'].shape[0]
print('{:.4f}% of values in numerical_1 are equal to {}'.format( _ * 100, min_n1  ))


# In[67]:

(merchants['numerical_1'] != merchants['numerical_2']).sum() / merchants.shape[0]


# ### 이 두 변수는 매우 유사합니다. 실제로 90 %의 상인을 위해 그들은 동일합니다.

# ```
# most_recent_sales_range most_recent_purchases_range avg_sales_lag3 avg_purchases_lag3 active_months_lag3 avg_sales_lag6 avg_purchases_lag6 active_months_lag6 avg_sales_lag12 avg_purchases_lag12 active_months_lag12
# ```

# In[68]:

# merchants['most_recent_sales_range'].value_counts().plot('bar');


# In[69]:

#d = merchants['most_recent_sales_range'].value_counts().sort_index()
#e = merchants.loc[merchants['numerical_2'] < 0.1].groupby('most_recent_sales_range')['numerical_1'].mean()
#data = [go.Bar(x=d.index, y=d.values, name='counts'), go.Scatter(x=e.index, y=e.values, name='mean numerical_1', yaxis='y2')]
#layout = go.Layout(dict(title = "Counts of values in categories of most_recent_sales_range",
#                        xaxis = dict(title = 'most_recent_sales_range'),
##                        yaxis = dict(title = 'Counts'),
#                        yaxis2=dict(title='mean numerical_1', overlaying='y', side='right')),
#                   legend=dict(orientation="v"))
# py.iplot(dict(data=data, layout=layout))


# ### 이상치를 제거한 후에도이 범위의 수치가 다르며 numeric_1의 평균이 다른 것을 볼 수 있습니다.

# ### most_recent_purchases_range

# In[70]:

#d = merchants['most_recent_purchases_range'].value_counts().sort_index()
#e = merchants.loc[merchants['numerical_2'] < 0.1].groupby('most_recent_purchases_range')['numerical_1'].mean()
#data = [go.Bar(x=d.index, y=d.values, name='counts'), go.Scatter(x=e.index, y=e.values, name='mean numerical_1', yaxis='y2')]
#layout = go.Layout(dict(title = "Counts of values in categories of most_recent_purchases_range",
#                        xaxis = dict(title = 'most_recent_purchases_range'),
#                        yaxis = dict(title = 'Counts'),
#                        yaxis2=dict(title='mean numerical_1', overlaying='y', side='right')),
#                   legend=dict(orientation="v"))
#py.iplot(dict(data=data, layout=layout))


# ### These two variables seem to be quite similar.

# ### avg_sales_lag

# In[71]:

#plt.hist(merchants['avg_sales_lag3'].fillna(0));
#plt.hist(merchants['avg_sales_lag6'].fillna(0));
#plt.hist(merchants['avg_sales_lag12'].fillna(0));


print("[72] time : ", time.time() - start)
for col in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']:
    print('Max value of {} is {}'.format(col,  merchants[col].max() ))
    print('Min value of {} is {}'.format(col,  merchants[col].max() ))

print("[73] time : ", time.time() - start)
# In[73]:

#plt.hist(merchants.loc[(merchants['avg_sales_lag12'] < 3) & (merchants['avg_sales_lag12'] > -10), 'avg_sales_lag12'].fillna(0), label='avg_sales_lag12');
#plt.hist(merchants.loc[(merchants['avg_sales_lag6'] < 3) & (merchants['avg_sales_lag6'] > -10), 'avg_sales_lag6'].fillna(0), label='avg_sales_lag6');
#plt.hist(merchants.loc[(merchants['avg_sales_lag3'] < 3) & (merchants['avg_sales_lag3'] > -10), 'avg_sales_lag3'].fillna(0), label='avg_sales_lag3');
#plt.legend();


# ### 이러한 값의 분포는 매우 유사하며 대부분의 값은 0과 2 사이입니다.

# ### avg_purchases_lag

# In[74]:

merchants['avg_purchases_lag3'].nlargest()


# In[75]:

merchants.loc[merchants['avg_purchases_lag3'] == np.inf, 'avg_purchases_lag3'] = 6000
merchants.loc[merchants['avg_purchases_lag6'] == np.inf, 'avg_purchases_lag6'] = 6000
merchants.loc[merchants['avg_purchases_lag12'] == np.inf, 'avg_purchases_lag12'] = 6000


# In[76]:

#plt.hist(merchants['avg_purchases_lag3'].fillna(0));
#plt.hist(merchants['avg_purchases_lag6'].fillna(0));
#plt.hist(merchants['avg_purchases_lag12'].fillna(0));


# In[77]:

#plt.hist(merchants.loc[(merchants['avg_purchases_lag12'] < 4), 'avg_purchases_lag12'].fillna(0), label='avg_purchases_lag12');
#plt.hist(merchants.loc[(merchants['avg_purchases_lag6'] < 4), 'avg_purchases_lag6'].fillna(0), label='avg_purchases_lag6');
#plt.hist(merchants.loc[(merchants['avg_purchases_lag3'] < 4), 'avg_purchases_lag3'].fillna(0), label='avg_purchases_lag3');
#plt.legend();


# ### 지금은 모델에서 판매자 데이터를 사용하지 않겠습니다.

# In[78]:

#train.head()


# In[79]:

for col in train.columns:
    if train[col].isna().any():
        train[col] = train[col].fillna(0)


# In[80]:

for col in test.columns:
    if test[col].isna().any():
        test[col] = test[col].fillna(0)


# In[81]:

y = train['target']


# In[82]:

col_to_drop = ['first_active_month', 'card_id', 'target']


# In[83]:

for col in col_to_drop:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)


# In[84]:

train['feature_3'] = train['feature_3'].astype(int)
test['feature_3'] = test['feature_3'].astype(int)


# In[85]:

categorical_feats = ['feature_1', 'feature_2']

for col in categorical_feats:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


# In[86]:

#train.head()


# In[87]:

for col in ['newpurchase_amount_max', 'newpurchase_date_max', 'purchase_amount_max_mean']:
    train[col + '_to_mean'] = train[col] / train[col].mean()
    test[col + '_to_mean'] = test[col] / test[col].mean()


# ### Basic LGB model

# In[88]:
X = train
X_test = test


print("[model training start] time : ", time.time() - start)
# ### Code for training models

# In[89]:

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
# folds = RepeatedKFold(n_splits=n_fold, n_repeats=2, random_state=11)


# In[90]:

def train_model(X=X, X_test=X_test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type=='lgb':
	#model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
            model = lgb.LGBMRegressor(n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse', verbose=1000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
            
        if model_type == 'rcv':
            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=3)
            model.fit(X_train, y_train)
            print(model.alpha_)

            y_pred_valid = model.predict(X_valid)
            score = mean_squared_error(y_valid, y_pred_valid) ** 0.5
            print('Fold {}. RMSE: {:.4f}.'.format(fold_n, score) )
            print('')
            
            y_pred = model.predict(X_test)
            
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='RMSE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
        
        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #plt.figure(figsize=(16, 12));
            #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            #plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction

params = {'num_leaves': 54, 'min_data_in_leaf': 79, 'objective': 'regression',  'max_depth': 15,  'learning_rate': 0.018545526395058548, "boosting": "gbdt", "feature_fraction": 0.8354507676881442,"bagging_freq": 5, "bagging_fraction": 0.8126672064208567,"bagging_seed": 11, "metric": 'rmse',"lambda_l1": 0.1, "verbosity": -1,'min_child_weight': 5.343384366323818,'reg_alpha': 1.1302650970728192,'reg_lambda': 0.3603427518866501,'subsample': 0.8767547959893627}

# ===========================================
print("[92 model lgb start] time : ", time.time() - start)
oof_lgb, prediction_lgb = train_model(params=params, model_type='lgb', plot_feature_importance=False)


submission['target'] = prediction_lgb
submission.to_csv('lgb.csv', index=False)
print("[92 model lgb end] time : ", time.time() - start)
# ===========================================


# ===========================================
print("[model xgb start] time : ", time.time() - start)
xgb_params = {'eta': 0.01, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')


submission['target'] = prediction_xgb
submission.to_csv('xgb.csv', index=False)
print("[model xgb end] time : ", time.time() - start)
# ===========================================

# ===========================================
print("[model rcv start] time : ", time.time() - start)
oof_rcv, prediction_rcv = train_model(params=None, model_type='rcv')

submission['target'] = prediction_rcv
submission.to_csv('rcv.csv', index=False)

print("[model rcv end] time : ", time.time() - start)
# ===========================================

# ===========================================
print("[model cat start] time : ", time.time() - start)
cat_params = {'learning_rate': 0.01, 'depth': 13, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

oof_cat, prediction_cat = train_model(params=cat_params, model_type='cat')

submission['target'] = (prediction_lgb + prediction_xgb + prediction_rcv + prediction_cat) / 4
submission.to_csv('blend.csv', index=False)
print("[model blend end] time : ", time.time() - start)
# ===========================================


train_stack = np.vstack([oof_lgb, oof_xgb, oof_rcv, oof_cat]).transpose()
train_stack = pd.DataFrame(train_stack)
test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_rcv, prediction_cat]).transpose()
test_stack = pd.DataFrame(test_stack)

# ===========================================
print("[model lgb stack start] time : ", time.time() - start)
oof_lgb_stack, prediction_lgb_stack = train_model(X=train_stack, X_test=test_stack, params=params, model_type='lgb')


sample_submission = pd.read_csv('/root/data/Elo/sample_submission.csv')
sample_submission['target'] = prediction_lgb_stack
sample_submission.to_csv('stacker_lgb.csv', index=False)
print("[model lgb stack end] time : ", time.time() - start)
# ===========================================

# ===========================================
print("[model rcv stack start] time : ", time.time() - start)
oof_rcv_stack, prediction_rcv_stack = train_model(X=train_stack, X_test=test_stack, params=None, model_type='rcv')

sample_submission = pd.read_csv('/root/data/Elo/sample_submission.csv')
sample_submission['target'] = prediction_rcv_stack
sample_submission.to_csv('stacker_rcv.csv', index=False)

print("[rcv stack end] time : ", time.time() - start)
# ===========================================

print("[end] time : ", time.time() - start)



