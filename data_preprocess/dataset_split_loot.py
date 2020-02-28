import numpy as np
import array
import copy

#数据处理第三步，将数据集进行划分为train和test文件，并产生evaluate的样本
category = 'Movies_and_TV'
k_core = '_5'
loot = '_loot'

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

#split test set as 'leave-one-out'
def split_test_set_loot(user_list, item_list):
    np.random.seed(0)
    ui_id_list = []
    ui_rating_list = []
    test_loot_list = []
    for user in user_list:
        ui_id_list.append([user])
        ui_rating_list.append([user])
    fout_train_rating = open(category + '//ratings_' + category + k_core + loot + '.train', 'w', encoding = 'UTF-8')
    fout_test_rating = open(category + '//ratings_' + category +  k_core + loot + '.test', 'w', encoding = 'UTF-8')
    #load user item id list 
    with open(category+'//ratings_'+category+k_core+'.csv', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user, item, rating = line.split(',')
            ui_id_list[user_list.index(user)].append(item_list.index(item))
            ui_rating_list[user_list.index(user)].append(int(rating))
    #print(ui_rating_list)
    training_set_count = 0
    for i in range(len(user_list)):
        user = user_list[i]
        #In fact, user_rate_num need to subtract 1
        user_rate_num = len(ui_id_list[i])
        loot_item_id = np.random.randint(1, user_rate_num)
        test_loot_list.append([i, ui_id_list[i][loot_item_id]])
        fout_test_rating.write(user + ',' + item_list[ui_id_list[i][loot_item_id]] + ',' + str(ui_rating_list[i][loot_item_id]) +'\n')
        for j in range(1, len(ui_id_list[i])):
            if j != loot_item_id:
                training_set_count += 1
                fout_train_rating.write(user + ',' + item_list[ui_id_list[i][j]] + ',' + str(ui_rating_list[i][j]) +'\n')
    print('Dataset split training count = ', training_set_count)
    return ui_id_list, test_loot_list

def generate_negative_test_sample(user_list, item_list, ui_id_list, test_loot_list): 
    np.random.seed(0)
    sample_num = 99
    fout_test_negative = open(category + '/' + category +  k_core + loot + '.test.negative', 'w', encoding = 'UTF-8')
    negativeList = []
    for i in range(len(user_list)):
        negatives = []
        #start to get samples for unrating dataset, and make sure there is not a complete element in the new list
        for t in range(sample_num):
            item_id = np.random.randint(0, len(item_list))
            while item_id in ui_id_list[i]:
                item_id = np.random.randint(0, len(item_list))
            negatives.append(item_id)
        fout_test_negative.write('(' + str(test_loot_list[i][0]) + ',' + str(test_loot_list[i][1]) + ') ')
        fout_test_negative.write(" ".join([str(x) for x in negatives]) + "\n")
        negativeList.append(negatives)

if __name__ == '__main__':
    user_list = load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    ui_id_list, test_loot_list = split_test_set_loot(user_list, item_list)
    generate_negative_test_sample(user_list, item_list, ui_id_list, test_loot_list)

