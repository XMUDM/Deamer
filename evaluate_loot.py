#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import math
import numpy as np
import dataset_loot as dataset
from keras.models import load_model
import heapq # for retrieval topK

def evaluate_combine_model(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item iamge vec inorder list over')

    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_rating'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_rating = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], item_img_vec_inorder[j]])
            predicted.append(pre_rating[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_rating'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            ndcg = getNDCG(ranklist, gtItem)
            NDCG.append(ndcg)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG =  = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_combine_model_CF(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item iamge vec inorder list over')

    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_rating'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_rating = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted.append(pre_rating[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_rating'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            ndcg = getNDCG(ranklist, gtItem)
            NDCG.append(ndcg)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG =  = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_rating_model(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_res = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], item_img_vec_inorder[j]])
            predicted.append(pre_res[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG =  = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_rating_model_CF(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    
    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')
    

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    #item_meta_vec_inorder = np.zeros((len(item_list),1, 1, DOC_VEC_DIM))
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_res = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted.append(pre_res[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_rating_model_CF_just_review(model, topK, category, k_core, loot):
    print('Start ranking evaluate on rating model CF just use review')
    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    print('Data reshape over')
    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_res = model.predict([user_vec_inorder[i], item_rev_vec_inorder[j], np.array([i]), np.array([j])])
            predicted.append(pre_res[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_rating_model_CF_review_image(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_res = model.predict([user_vec_inorder[i], item_rev_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted.append(pre_res[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_rating_model_CF_review_metadata(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    
    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')
    

    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'w')
    HitRatio = []
    NDCG = []
    MRR = []
    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted = []
        for j in item_id_test_list[i]:
            pre_res = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], np.array([i]), np.array([j])])
            predicted.append(pre_res[0])
        predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted]) + "\n" )
        map_item_score = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score[item] = predicted[k]
        # Evaluate top rank list
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        HitRatio.append(hr)
        ndcg = getNDCG(ranklist, gtItem)
        NDCG.append(ndcg)
        mrr = getMRR(ranklist, gtItem)
        MRR.append(mrr)
    

    '''
    #Load the predicted result from saved file
    pred_lens = len(item_id_test_list[0])
    HitRatio = []
    MRR = []
    i = 0
    with open(category+'/result/ui_predicted_response_on_rating_only'+loot+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            predicted = list(map(float, line.split(' ')[:pred_lens]))
            predicted = np.array(predicted).reshape(len(item_id_test_list[i]))
            gtItem = item_id_test_list[i][-1]
            map_item_score = {}
            #print(predicted)
            for k in range(len(item_id_test_list[i])):
                item = item_id_test_list[i][k]
                map_item_score[item] = predicted[k]
            # Evaluate top rank list
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            #if hr>0:
                #print(gtItem, ranklist)
            HitRatio.append(hr)
            mrr = getMRR(ranklist, gtItem)
            #print(ranklist)
            MRR.append(mrr)
            i += 1
    '''

    hits = np.array(HitRatio).mean()
    mrrs = np.array(MRR).mean()
    ndcgs = np.array(NDCG).mean()
    print('HitRatio = %.4f'%hits, ',NDCG = %.4f'%ndcgs, ',MRR = %.4f'%mrrs)
    
    return hits, ndcgs, mrrs

def evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    
    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')
    

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    #item_meta_vec_inorder = np.zeros((len(item_list),1, 1, DOC_VEC_DIM))
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response'+loot+k_core+'.txt', 'w')
    HitRatio_re, NDCG_re, MRR_re = [], [], []
    HitRatio_ra, NDCG_ra, MRR_ra = [], [], []

    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted_response = []
        predicted_rating = []
        for j in item_id_test_list[i]:
            pre_rating, pre_response = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted_response.append(pre_response[0])
            predicted_rating.append(pre_rating[0])
        predicted_response = np.array(predicted_response).reshape(len(item_id_test_list[i]))
        predicted_rating = np.array(predicted_rating).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted_response]) + "\n" )
        map_item_score_response = {}
        map_item_score_rating = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score_response[item] = predicted_response[k]
            map_item_score_rating[item] = predicted_rating[k]
        # Evaluate top rank list use reponse
        ranklist_response = heapq.nlargest(topK, map_item_score_response, key=map_item_score_response.get)
        hr_re = getHitRatio(ranklist_response, gtItem)
        HitRatio_re.append(hr_re)
        ndcg_re = getNDCG(ranklist_response, gtItem)
        NDCG_re.append(ndcg_re)
        mrr_re = getMRR(ranklist_response, gtItem)
        MRR_re.append(mrr_re)

        # Evaluate top rank list use rating
        ranklist_rating = heapq.nlargest(topK, map_item_score_rating, key=map_item_score_rating.get)
        hr_ra = getHitRatio(ranklist_rating, gtItem)
        HitRatio_ra.append(hr_ra)
        ndcg_ra = getNDCG(ranklist_rating, gtItem)
        NDCG_ra.append(ndcg_ra)
        mrr_ra = getMRR(ranklist_rating, gtItem)
        MRR_ra.append(mrr_ra)

    hits_re = np.array(HitRatio_re).mean()
    mrrs_re = np.array(MRR_re).mean()
    ndcgs_re = np.array(NDCG_re).mean()
    print('Use response : HitRatio = %.4f'%hits_re, ',NDCG = %.4f'%ndcgs_re, ',MRR = %.4f'%mrrs_re)
    

    hits_ra = np.array(HitRatio_ra).mean()
    mrrs_ra = np.array(MRR_ra).mean()
    ndcgs_ra = np.array(NDCG_ra).mean()
    print('Use rating : HitRatio = %.4f'%hits_ra, ',NDCG = %.4f'%ndcgs_ra, ',MRR = %.4f'%mrrs_ra)

    return hits_re, ndcgs_re, mrrs_re, hits_ra, ndcgs_ra, mrrs_ra

def evaluate_response_on_MNAR_base_model_just_use_review(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    #item_meta_vec_inorder = np.zeros((len(item_list),1, 1, DOC_VEC_DIM))
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_just_use_review_'+loot+k_core+'.txt', 'w')
    HitRatio_re, NDCG_re, MRR_re = [], [], []
    HitRatio_ra, NDCG_ra, MRR_ra = [], [], []

    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted_response = []
        predicted_rating = []
        for j in item_id_test_list[i]:
            pre_rating, pre_response = model.predict([user_vec_inorder[i], item_rev_vec_inorder[j], np.array([i]), np.array([j])])
            predicted_response.append(pre_response[0])
            predicted_rating.append(pre_rating[0])
        predicted_response = np.array(predicted_response).reshape(len(item_id_test_list[i]))
        predicted_rating = np.array(predicted_rating).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted_response]) + "\n" )
        map_item_score_response = {}
        map_item_score_rating = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score_response[item] = predicted_response[k]
            map_item_score_rating[item] = predicted_rating[k]
        # Evaluate top rank list use reponse
        ranklist_response = heapq.nlargest(topK, map_item_score_response, key=map_item_score_response.get)
        hr_re = getHitRatio(ranklist_response, gtItem)
        HitRatio_re.append(hr_re)
        ndcg_re = getNDCG(ranklist_response, gtItem)
        NDCG_re.append(ndcg_re)
        mrr_re = getMRR(ranklist_response, gtItem)
        MRR_re.append(mrr_re)

        # Evaluate top rank list use rating
        ranklist_rating = heapq.nlargest(topK, map_item_score_rating, key=map_item_score_rating.get)
        hr_ra = getHitRatio(ranklist_rating, gtItem)
        HitRatio_ra.append(hr_ra)
        ndcg_ra = getNDCG(ranklist_rating, gtItem)
        NDCG_ra.append(ndcg_ra)
        mrr_ra = getMRR(ranklist_rating, gtItem)
        MRR_ra.append(mrr_ra)

    hits_re = np.array(HitRatio_re).mean()
    mrrs_re = np.array(MRR_re).mean()
    ndcgs_re = np.array(NDCG_re).mean()
    print('Use response : HitRatio = %.4f'%hits_re, ',NDCG = %.4f'%ndcgs_re, ',MRR = %.4f'%mrrs_re)
    

    hits_ra = np.array(HitRatio_ra).mean()
    mrrs_ra = np.array(MRR_ra).mean()
    ndcgs_ra = np.array(NDCG_ra).mean()
    print('Use rating : HitRatio = %.4f'%hits_ra, ',NDCG = %.4f'%ndcgs_ra, ',MRR = %.4f'%mrrs_ra)

    return hits_re, ndcgs_re, mrrs_re, hits_ra, ndcgs_ra, mrrs_ra

def evaluate_response_on_MNAR_base_model_no_item_review(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_meta_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')

    
    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')
    

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    #item_meta_vec_inorder = np.zeros((len(item_list),1, 1, DOC_VEC_DIM))
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_no_item_review'+loot+k_core+'.txt', 'w')
    HitRatio_re, NDCG_re, MRR_re = [], [], []
    HitRatio_ra, NDCG_ra, MRR_ra = [], [], []

    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted_response = []
        predicted_rating = []
        for j in item_id_test_list[i]:
            pre_rating, pre_response = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted_response.append(pre_response[0])
            predicted_rating.append(pre_rating[0])
        predicted_response = np.array(predicted_response).reshape(len(item_id_test_list[i]))
        predicted_rating = np.array(predicted_rating).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted_response]) + "\n" )
        map_item_score_response = {}
        map_item_score_rating = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score_response[item] = predicted_response[k]
            map_item_score_rating[item] = predicted_rating[k]
        # Evaluate top rank list use reponse
        ranklist_response = heapq.nlargest(topK, map_item_score_response, key=map_item_score_response.get)
        hr_re = getHitRatio(ranklist_response, gtItem)
        HitRatio_re.append(hr_re)
        ndcg_re = getNDCG(ranklist_response, gtItem)
        NDCG_re.append(ndcg_re)
        mrr_re = getMRR(ranklist_response, gtItem)
        MRR_re.append(mrr_re)

        # Evaluate top rank list use rating
        ranklist_rating = heapq.nlargest(topK, map_item_score_rating, key=map_item_score_rating.get)
        hr_ra = getHitRatio(ranklist_rating, gtItem)
        HitRatio_ra.append(hr_ra)
        ndcg_ra = getNDCG(ranklist_rating, gtItem)
        NDCG_ra.append(ndcg_ra)
        mrr_ra = getMRR(ranklist_rating, gtItem)
        MRR_ra.append(mrr_ra)

    hits_re = np.array(HitRatio_re).mean()
    mrrs_re = np.array(MRR_re).mean()
    ndcgs_re = np.array(NDCG_re).mean()
    print('Use response : HitRatio = %.4f'%hits_re, ',NDCG = %.4f'%ndcgs_re, ',MRR = %.4f'%mrrs_re)
    

    hits_ra = np.array(HitRatio_ra).mean()
    mrrs_ra = np.array(MRR_ra).mean()
    ndcgs_ra = np.array(NDCG_ra).mean()
    print('Use rating : HitRatio = %.4f'%hits_ra, ',NDCG = %.4f'%ndcgs_ra, ',MRR = %.4f'%mrrs_ra)

    return hits_re, ndcgs_re, mrrs_re, hits_ra, ndcgs_ra, mrrs_ra

def evaluate_response_on_MNAR_base_model_no_item_metadata(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_img_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    #load item metadata vec inorder
    with open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_inorder.append(list(map(float, line.split(' ')[:4096])))
    print('Load item image vec inorder list over')

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_img_vec_inorder = np.array(item_img_vec_inorder).reshape(len(item_list),1, 64, 64, 1)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response_no_item_metadata'+loot+k_core+'.txt', 'w')
    HitRatio_re, NDCG_re, MRR_re = [], [], []
    HitRatio_ra, NDCG_ra, MRR_ra = [], [], []

    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted_response = []
        predicted_rating = []
        for j in item_id_test_list[i]:
            pre_rating, pre_response = model.predict([user_vec_inorder[i], item_rev_vec_inorder[j], item_img_vec_inorder[j], np.array([i]), np.array([j])])
            predicted_response.append(pre_response[0])
            predicted_rating.append(pre_rating[0])
        predicted_response = np.array(predicted_response).reshape(len(item_id_test_list[i]))
        predicted_rating = np.array(predicted_rating).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted_response]) + "\n" )
        map_item_score_response = {}
        map_item_score_rating = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score_response[item] = predicted_response[k]
            map_item_score_rating[item] = predicted_rating[k]
        # Evaluate top rank list use reponse
        ranklist_response = heapq.nlargest(topK, map_item_score_response, key=map_item_score_response.get)
        hr_re = getHitRatio(ranklist_response, gtItem)
        HitRatio_re.append(hr_re)
        ndcg_re = getNDCG(ranklist_response, gtItem)
        NDCG_re.append(ndcg_re)
        mrr_re = getMRR(ranklist_response, gtItem)
        MRR_re.append(mrr_re)

        # Evaluate top rank list use rating
        ranklist_rating = heapq.nlargest(topK, map_item_score_rating, key=map_item_score_rating.get)
        hr_ra = getHitRatio(ranklist_rating, gtItem)
        HitRatio_ra.append(hr_ra)
        ndcg_ra = getNDCG(ranklist_rating, gtItem)
        NDCG_ra.append(ndcg_ra)
        mrr_ra = getMRR(ranklist_rating, gtItem)
        MRR_ra.append(mrr_ra)

    hits_re = np.array(HitRatio_re).mean()
    mrrs_re = np.array(MRR_re).mean()
    ndcgs_re = np.array(NDCG_re).mean()
    print('Use response : HitRatio = %.4f'%hits_re, ',NDCG = %.4f'%ndcgs_re, ',MRR = %.4f'%mrrs_re)
    

    hits_ra = np.array(HitRatio_ra).mean()
    mrrs_ra = np.array(MRR_ra).mean()
    ndcgs_ra = np.array(NDCG_ra).mean()
    print('Use rating : HitRatio = %.4f'%hits_ra, ',NDCG = %.4f'%ndcgs_ra, ',MRR = %.4f'%mrrs_ra)

    return hits_re, ndcgs_re, mrrs_re, hits_ra, ndcgs_ra, mrrs_ra

def evaluate_response_on_MNAR_base_model_no_item_image(model, topK, category, k_core, loot):

    DOC_VEC_DIM = 500
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    
    user_vec_inorder = []
    item_rev_vec_inorder = []
    item_meta_vec_inorder = []

    #load user vec inorder
    with open(category+'//feature_data//users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load user review vec inorder list over')
    #load item review vec inorder
    with open(category+'//feature_data//items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_rev_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item review vec inorder list over')

    
    #load item metadata vec inorder
    with open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_inorder.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    print('Load item metadata vec inorder list over')
    

    #print('metadata is initial to 0s')
    user_vec_inorder = np.array(user_vec_inorder).reshape(len(user_list), 1, 1, DOC_VEC_DIM)
    #item_meta_vec_inorder = np.zeros((len(item_list),1, 1, DOC_VEC_DIM))
    item_meta_vec_inorder = np.array(item_meta_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    item_rev_vec_inorder = np.array(item_rev_vec_inorder).reshape(len(item_list),1, 1, DOC_VEC_DIM)
    print('Data reshape over')
    

    
    #Predict the test list anew
    fout_ui_pred = open(category+'/result/ui_predicted_response'+loot+k_core+'.txt', 'w')
    HitRatio_re, NDCG_re, MRR_re = [], [], []
    HitRatio_ra, NDCG_ra, MRR_ra = [], [], []

    for i in range(len(user_list)):
        if i%300 == 0:
            print(i)
        gtItem = item_id_test_list[i][-1]
        # Get prediction scores
        predicted_response = []
        predicted_rating = []
        for j in item_id_test_list[i]:
            pre_rating, pre_response = model.predict([user_vec_inorder[i], item_meta_vec_inorder[j], item_rev_vec_inorder[j], np.array([i]), np.array([j])])
            predicted_response.append(pre_response[0])
            predicted_rating.append(pre_rating[0])
        predicted_response = np.array(predicted_response).reshape(len(item_id_test_list[i]))
        predicted_rating = np.array(predicted_rating).reshape(len(item_id_test_list[i]))
        fout_ui_pred.write(" ".join([str(x) for x in predicted_response]) + "\n" )
        map_item_score_response = {}
        map_item_score_rating = {}
        #print(predicted)
        for k in range(len(item_id_test_list[i])):
            item = item_id_test_list[i][k]
            map_item_score_response[item] = predicted_response[k]
            map_item_score_rating[item] = predicted_rating[k]
        # Evaluate top rank list use reponse
        ranklist_response = heapq.nlargest(topK, map_item_score_response, key=map_item_score_response.get)
        hr_re = getHitRatio(ranklist_response, gtItem)
        HitRatio_re.append(hr_re)
        ndcg_re = getNDCG(ranklist_response, gtItem)
        NDCG_re.append(ndcg_re)
        mrr_re = getMRR(ranklist_response, gtItem)
        MRR_re.append(mrr_re)

        # Evaluate top rank list use rating
        ranklist_rating = heapq.nlargest(topK, map_item_score_rating, key=map_item_score_rating.get)
        hr_ra = getHitRatio(ranklist_rating, gtItem)
        HitRatio_ra.append(hr_ra)
        ndcg_ra = getNDCG(ranklist_rating, gtItem)
        NDCG_ra.append(ndcg_ra)
        mrr_ra = getMRR(ranklist_rating, gtItem)
        MRR_ra.append(mrr_ra)

    hits_re = np.array(HitRatio_re).mean()
    mrrs_re = np.array(MRR_re).mean()
    ndcgs_re = np.array(NDCG_re).mean()
    print('Use response : HitRatio = %.4f'%hits_re, ',NDCG = %.4f'%ndcgs_re, ',MRR = %.4f'%mrrs_re)
    

    hits_ra = np.array(HitRatio_ra).mean()
    mrrs_ra = np.array(MRR_ra).mean()
    ndcgs_ra = np.array(NDCG_ra).mean()
    print('Use rating : HitRatio = %.4f'%hits_ra, ',NDCG = %.4f'%ndcgs_ra, ',MRR = %.4f'%mrrs_ra)

    return hits_re, ndcgs_re, mrrs_re, hits_ra, ndcgs_ra, mrrs_ra



def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getMRR(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1/(ranklist.index(item)+1)
    return 0

# Global variables that are shared across processes
#y_true and y_pred are array here
def evaluate_rating_model(y_true, y_pred):
    rmse = getRmse(y_true, y_pred)
    mae = getMae(y_true, y_pred)
    return rmse, mae

def getRmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred-y_true), axis=-1))
    #return np.sqrt(np.mean((y_pred-y_true)**2))
    #return np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1)/len(y_pred))

def getMae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)
    #return np.sum(np.abs(y_pred - y_true), axis=-1)/len(y_pred)

if __name__ == '__main__':
    epoch = 1
    topK = 10
    category = 'Musical_Instruments'
    k_core = '_5'
    loot = '_loot'
    #item_id_test_list = dataset.load_item_test_id(category, k_core, loot)
    #print(item_id_test_list)
    res_model_filepath = category+'/model/Multi_modalities_response_on_rating_loot_' + str(epoch) + '.model' + k_core
    model = load_model(res_model_filepath)
    hits, ndcgs, mrrs = evaluate_response_on_rating_model_CF(model, topK, category, k_core, loot)
    #print(model.get_weights())

