import numpy as np
import array
import copy
import random

doc_vec_dim = 500
image_vec_dim = 4096
train_amount = 300000
test_amount = 75000
minimal_flag = 0
user_embedding_dim = 10
item_embedding_dim = 10
k_core = '_5'

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

#每个用户平均分配样本数量，其值=总的用户的打分数量*sample_ratio/用户数量
def sampling_response_with_unrating_training_bak(user_list, item_list, category, k_core, sample_ratio = 5):
    np.random.seed(0)
    ui_id_list = []
    user_sample_train_list = []
    item_sample_train_list = []
    user_mlp_sample_input, item_mlp_sample_input = [], []
    for user in user_list:
        ui_id_list.append([user])
    count = 0
    print('Load data from training set')
    #load user item id list 
    with open(category+'//ratings_'+category+k_core+'_loot.train', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            count += 1
            user, item, rating= line.split(',')
            ui_id_list[user_list.index(user)].append(item_list.index(item))

    train_count = count - len(user_list)

    sample_sum = train_count*sample_ratio
    sample_num = int(sample_sum/len(user_list))
    print('Sampling as all the users')
    for i in range(len(user_list)):
        user = user_list[i]
        #start to get samples for unrating dataset, and make sure there is not a complete element in the new list
        for t in range(sample_num):
            item_id = np.random.randint(0, len(item_list))
            while item_id in ui_id_list[i]:
                item_id = np.random.randint(0, len(item_list))
            ui_id_list[i].append(item_id)
            user_sample_train_list.append(user)
            item_sample_train_list.append(item_list[item_id])
            user_mlp_sample_input.append(i)
            item_mlp_sample_input.append(item_id)

    print('Sampling amount: training set = ', len(user_sample_train_list))
    return user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input

#如果一个用户有对一个商品的打分，那么生成sample_ratio个新的样本
def sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio = 5):
    np.random.seed(0)
    ui_id_list = []
    user_sample_train_list = []
    item_sample_train_list = []
    user_mlp_sample_input, item_mlp_sample_input = [], []
    for user in user_list:
        ui_id_list.append([user])
    #load user item id list 
    with open(category+'//ratings_'+category+k_core+'_loot.train', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user, item, rating= line.split(',')
            ui_id_list[user_list.index(user)].append(item_list.index(item))
    for i in range(len(user_list)):
        user = user_list[i]
        #-1是因为ui_id_list中的每个元素列表都先增加了用户名，占据了当前元素的一个长度
        sample_num = (len(ui_id_list[i])-1)*sample_ratio

        #start to get samples for unrating dataset, and make sure there is not a complete element in the new list
        for t in range(sample_num):
            item_id = np.random.randint(0, len(item_list))
            while item_id in ui_id_list[i]:
                item_id = np.random.randint(0, len(item_list))
            ui_id_list[i].append(item_id)
            #user_sample_train_list.append(user)
            #item_sample_train_list.append(item_list[item_id])
            user_mlp_sample_input.append(i)
            item_mlp_sample_input.append(item_id)
    print('Sampling amount: training set = ', len(user_mlp_sample_input))
    return user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input

def load_negative_testing(user_list, item_list, category, k_core, loot):
    user_sample_test_list = []
    item_sample_test_list = []
    for i in range(len(user_list)):
        user_sample_test_list.append(i)
    with open(category+'//'+category+k_core+loot+'.test.negative', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            arr = line.split(" ")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            item_sample_test_list.append(negatives)
    #print(item_sample_test_list)
    return user_sample_test_list, item_sample_test_list

#modality: contains users_reviews, items_reviews, items_metadata and items_image
def load_vec_by_list(tr_ui_list, category, k_core, modality, ui_list):
    vec_dim = 0
    if 'image' in modality:
        vec_dim = 4096
    else :
        vec_dim = 500
    tr_vec_array = np.zeros((len(tr_ui_list), vec_dim))
    ui_vec_list = []
    #First: load vectors of modality by order
    with open(category+'/feature_data/'+modality+'_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            ui_vec_list.append(list(map(float, line.split(' ')[:vec_dim])))
    #Then: map vectors by user or item training or testing list
    for ui_id in tr_ui_list:
        if ui_id in ui_list:
            _index = ui_list.index(ui_id)
            tr_vec_array[_index] = np.array(ui_vec_list[_index]).copy()
    print('Load', modality, ' over, amount = ', len(tr_ui_list))
    return tr_vec_array

def gauss_noise(x):
    mu = 0
    sigma = 0.01
    return float(x)+random.gauss(mu,sigma)

#modality: contains users_reviews, items_reviews, items_metadata and items_image
#在每次读取数据时增加了高斯噪声
def load_vec_by_list_add_noise(tr_ui_list, category, k_core, modality, ui_list):
    vec_dim = 0
    if 'image' in modality:
        vec_dim = 4096
    else :
        vec_dim = 500
    tr_vec_array = np.zeros((len(tr_ui_list), vec_dim))
    ui_vec_list = []
    #First: load vectors of modality by order
    with open(category+'/feature_data/'+modality+'_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            ui_vec_list.append(list(map(gauss_noise, line.split(' ')[:vec_dim])))
    #Then: map vectors by user or item training or testing list
    for ui_id in tr_ui_list:
        if ui_id in ui_list:
            _index = ui_list.index(ui_id)
            tr_vec_array[_index] = np.array(ui_vec_list[_index]).copy()
    print('Load', modality, ' over, amount = ', len(tr_ui_list))
    return tr_vec_array

#modality: contains users_reviews, items_reviews, items_metadata and items_image
#给每个数据增加了高斯噪声，同样的数据每次增加的噪声可能会不同
def load_vec_by_list_add_noise2(tr_ui_list, category, k_core, modality, ui_list):
    vec_dim = 0
    if 'image' in modality:
        vec_dim = 4096
    else :
        vec_dim = 500
    tr_vec_array = np.zeros((len(tr_ui_list), vec_dim))
    ui_vec_list = []
    #First: load vectors of modality by order
    with open(category+'/feature_data/'+modality+'_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            ui_vec_list.append(list(map(float, line.split(' ')[:vec_dim])))

    #Then: map vectors by user or item training or testing list
    for ui_id in tr_ui_list:
        if ui_id in ui_list:
            _index = ui_list.index(ui_id)
            _ui_vec = list(map(gauss_noise, ui_vec_list[_index][:vec_dim]))
            tr_vec_array[_index] = np.array(_ui_vec).copy()
    print('Load', modality, ' over, amount = ', len(tr_ui_list))
    return tr_vec_array

def load_initial_training_list(user_list, item_list, category, k_core, loot):
    user_initial_train_list = []
    item_initial_train_list = []
    rating_train = []
    user_mlp_initial_input, item_mlp_initial_input = [], []
    with open(category+'/ratings_'+category+k_core+loot+'.train', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user, item, rating = line.split(',')
            user_initial_train_list.append(user)
            item_initial_train_list.append(item)
            rating_train.append(float(rating)/5)
            user_mlp_initial_input.append(user_list.index(user))
            item_mlp_initial_input.append(item_list.index(item))
    return user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input

def load_initial_testing_list(user_list, item_list, category, k_core, loot):
    user_initial_test_list = []
    item_initial_test_list = []
    rating_test = []
    user_mlp_test_input, item_mlp_test_input = [], []
    with open(category+'/ratings_'+category+k_core+loot+'.test', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user, item, rating = line.split(',')
            user_initial_test_list.append(user)
            item_initial_test_list.append(item)
            rating_test.append(float(rating)/5)
            user_mlp_test_input.append(user_list.index(user))
            item_mlp_test_input.append(item_list.index(item))
    return user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input


#The first one is the ground-truth, while the other are negative samples
def load_item_test_id(category, k_core, loot):
    item_id_test_list = []
    with open(category+'//'+category+k_core+loot+'.test.negative', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            arr = line.split(" ")
            id_list = []
            user, item = arr[0].strip('()').split(',')
            for x in arr[1: ]:
                id_list.append(int(x))
            #The ground truth is the last one
            id_list.append(int(item))
            item_id_test_list.append(id_list)
    return item_id_test_list

def map_to_0_1(x, threshold = 0.5):
    if x>=threshold:
        return 1
    else :
        return 0

def process_response(labels, threshold = 0.5):
    labels = np.array(list(map(lambda x :map_to_0_1(x, threshold), labels)))
    return labels

if __name__ == '__main__':
    category = 'Musical_Instruments'
    k_core = '_5'
    loot = '_loot'
    user_list = load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list = sampling_response_with_unrating_training(user_list, item_list, category, k_core, 5)
    user_sample_test_list, item_sample_test_list = load_negative_testing(user_list, item_list, category, k_core, loot)










