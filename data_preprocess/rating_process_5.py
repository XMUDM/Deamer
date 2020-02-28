import numpy as np
import copy

user_list = []
item_list = []

category = 'Movies_and_TV'
k_core = '_5'
loot = '_loot'

#数据处理第四步，将rating文件处理成userid,itemid,rating的形式
def load_unique_user_list(path):
    user_list = []
    with open(path, 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user, rating = line.split(',')
            user_list.append(user)
    print('Load unique users over, amount = ', len(user_list))
    return user_list

def load_unique_item_list(path):
    item_list = []
    with open(path, 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item, rating = line.split(',')
            item_list.append(item)
    print('Load unique items over, amount = ', len(item_list))
    return item_list

fout = open(category + '/ratings_' + category + k_core + '.dat','w', encoding = 'UTF-8')
fout_train = open(category + '/ratings_' + category + k_core + '.dat.train','w', encoding = 'UTF-8')
fout_test = open(category + '/ratings_' + category + k_core + '.dat.test','w', encoding = 'UTF-8')

user_list = load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
item_list = load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')

user_num_list = []
item_num_list = []
rating_list = []
with open(category + '/ratings_' + category + k_core + '.csv', 'r', encoding = 'UTF-8') as f:
    for line in f.readlines() :
        user, item, rating = line.split(',')
        user_num_list.append(user_list.index(user)+1)
        item_num_list.append(item_list.index(item)+1)
        rating_list.append(int(rating))
for user, item, rating in zip(user_num_list, item_num_list, rating_list):
    fout.write(str(user) + ',' + str(item) + ',' + str(rating)+'\n')

user_num_list = []
item_num_list = []
rating_list = []
with open(category + '/ratings_' + category + k_core + loot + '.train', 'r', encoding = 'UTF-8') as f:
    for line in f.readlines() :
        user, item, rating = line.split(',')
        user_num_list.append(user_list.index(user)+1)
        item_num_list.append(item_list.index(item)+1)
        rating_list.append(int(rating))
for user, item, rating in zip(user_num_list, item_num_list, rating_list):
    fout_train.write(str(user) + ',' + str(item) + ',' + str(rating)+'\n')

user_num_list = []
item_num_list = []
rating_list = []
with open(category + '/ratings_' + category + k_core + loot + '.test', 'r', encoding = 'UTF-8') as f:
    for line in f.readlines() :
        user, item, rating = line.split(',')
        user_num_list.append(user_list.index(user)+1)
        item_num_list.append(item_list.index(item)+1)
        rating_list.append(int(rating))
for user, item, rating in zip(user_num_list, item_num_list, rating_list):
    fout_test.write(str(user) + ',' + str(item) + ',' + str(rating)+'\n')
