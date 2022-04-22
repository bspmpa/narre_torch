'''
Data pre process

'''

import os
import json
import pandas as pd
import pickle
import numpy as np
import random

TPS_DIR = '/code/r9/xai/NARRE/data/boss/'
df = pd.read_csv(os.path.join(TPS_DIR, 'data.csv'))

pairs = ['geek_id', 'job_id']
user_ft = ['geek_degree_code','expect_city','expect_pos_code', 'expect_low_salary', 'expect_high_salary']
item_ft = ['job_city_code', 'job_pos_code', 'job_low_salary', 'job_high_salary', 'job_degree_code', 'job_experience_code']

uniform_fts = user_ft + item_ft + pairs
kk = [df[i].factorize()[0]+1 for i in uniform_fts]
uuu = [j+'_uniform' for j in uniform_fts]
for i in range(len(kk)):
    df[uuu[i]]=kk[i] 
    
df['user_review'] = list(df[uuu[:len(user_ft)]].values)
df['item_review'] = list(df[uuu[len(user_ft):-2]].values)
        
num_users = [int(df[i].max()) for i in uuu[:len(user_ft)]]
num_items = [int(df[i].max()) for i in uuu[len(user_ft):]]

print('num_users', num_users)
print('num_items', num_items)

df = df.rename({'geek_id_uniform':'user_id', 'job_id_uniform':'item_id', 'is_addf':'ratings'}, axis='columns')
data = df[['user_id', 'item_id', 'ratings', 'user_review', 'item_review']]


user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
for i in data.values:
    if i[0] in user_reviews:
        user_reviews[i[0]].append(i[4]) # item attribute 代替 item review
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[i[4]]
    if i[1] in item_reviews:
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]]=[i[0]]

def num_standard(user_reviews):
    nk = 1
    dd = {}
    mm = {}
    for i in list(user_reviews.keys()):
        if len(user_reviews[i])>1:
            dd[nk] = user_reviews[i][:2]
            mm[i] = nk
            nk += 1 
        else:
            user_reviews.pop(i)   
    return dd, mm

# def num_standard_id(user_reviews, mm):
#     nk = 1
#     dd = {}
#     for i in list(user_reviews.keys()):
#         if len(user_reviews[i])>2:
#             try:
#                 aa=[]
#                 aa.append(mm[user_reviews[i][0]])
#                 aa.append(mm[user_reviews[i][1]])
#                 dd[nk] = aa
#                 nk += 1 
#             except:
#                 dd[nk] =[0,0]
                
#         else:
#             user_reviews.pop(i)   
#     return dd

user_reviews, mm_user = num_standard(user_reviews)
item_reviews, mm_item = num_standard(item_reviews)

user_rid,_ = num_standard(user_rid)
item_rid,_ = num_standard(item_rid)



print('============')
print(np.array(list(user_rid.keys())).max())
print(np.array(list(item_rid.keys())).max())

data = data[data['user_id'].isin(list(user_reviews.keys()))]
data = data[data['item_id'].isin(list(item_reviews.keys()))]





aa = list(data[['user_id','item_id','ratings']].groupby('user_id', as_index=True))
aaa=[]
bbb=[]
for i in range(len(aa)):
    bb = list(set(data['item_id'].unique())-set(list(aa)[i][1]['item_id'].values))
    bbb.append(random.sample(bb, 10))
    aaa.append([list(aa)[i][0]]*10)
    
rrr = np.zeros((len(bbb)*10,)) 

df_neg = pd.DataFrame({'user_id':np.array(aaa).reshape(-1), 'item_id':np.array(bbb).reshape(-1), 
                       'ratings':rrr})


tp_rating=data[['user_id','item_id','ratings']]
tp_rating = pd.concat([tp_rating, df_neg])

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train= tp_rating[~test_idx]


n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]


print(sum(tp_train['ratings']==0))
print(sum(tp_valid['ratings']==0))
print(sum(tp_test['ratings']==0))

tp_train.to_csv(os.path.join(TPS_DIR, 'music_train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'music_valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'music_test.csv'), index=False,header=None)


padding_user = np.zeros(data['item_review'].iloc[0].shape)
padding_item = np.zeros(data['user_review'].iloc[0].shape)

print('padding_user', padding_user)
print('padding_item', padding_item)

for i in tp_1.values:
    if i[0] in user_reviews:
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=padding_user
    if i[1] in item_reviews:
        l=1
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]]= padding_item
        

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

# usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')


# print(np.sort(np.array(usercount.values)))

# print(np.sort(np.array(itemcount.values)))
