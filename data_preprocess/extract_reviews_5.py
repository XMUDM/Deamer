import json
import gzip
import array
import numpy as np

#数据处理第一步，抽取出不同的用户和商品的聚合

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

category = 'Movies_and_TV'
k_core = '_5'

#ratings = []
#reviews = []
users_id = []
items_id = []
unique_users = []
unique_items = []
users_reviews = []
items_reviews = []
users_ratings = []
items_ratings = []
users_avg_ratings = []
items_avg_ratings = []
count = 0

print('category = ', category, ', start extracting reviews')
for review in parse(category + "/reviews_" + category + k_core + ".json.gz"): 
    if review['reviewerID'] not in unique_users:
        unique_users.append(review['reviewerID'])
        users_reviews.append(review['reviewText'])
        users_ratings.append([review['overall']])
    else :
        users_reviews[unique_users.index(review['reviewerID'])] += review['reviewText']
        users_ratings[unique_users.index(review['reviewerID'])].append(review['overall'])
    if review['asin'] not in unique_items:
        unique_items.append(review['asin'])
        items_reviews.append(review['reviewText'])
        items_ratings.append([review['overall']])
    else :
        items_reviews[unique_items.index(review['asin'])] += review['reviewText']
        items_ratings[unique_items.index(review['asin'])].append(review['overall'])
    #ratings.append(review['overall'])
    #reviews.append(review['reviewText'])
    count += 1
    if count < 2:
        print(review['asin'])
    if count % 10000 == 0:
        print('Already read data amount = %d'%(count))
    #if count > 1000:
        #break
#print(sum(ratings) / len(ratings))
for i in range(len(users_ratings)):
    users_avg_ratings.append([sum(users_ratings[i]) / len(users_ratings[i]) *1.0])
for i in range(len(items_ratings)):
    items_avg_ratings.append([sum(items_ratings[i]) / len(items_ratings[i])*1.0])

#print(count)
print(len(unique_users), len(users_reviews), len(users_avg_ratings))
print(len(unique_items), len(items_reviews), len(items_avg_ratings))

fCategory = category + '/PreData/'
f_user_rating = open(fCategory + 'unique_users_rating'+ k_core +'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_users)):
    f_user_rating.write('%s,%.6f\n'%(unique_users[i], users_avg_ratings[i][0]))
f_item_rating = open(fCategory + 'unique_items_rating'+ k_core +'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_items)):
    f_item_rating.write('%s,%.6f\n'%(unique_items[i], items_avg_ratings[i][0]))
f_user_review = open(fCategory + 'users_reviews'+ k_core +'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_users)):
    f_user_review.write('%s\n'%(users_reviews[i]))
f_item_review = open(fCategory + 'items_reviews'+ k_core +'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_items)):
    f_item_review.write('%s\n'%(items_reviews[i]))

'''
count = 0
for item, pic in readImageFeatures("Instruments/image_features_Musical_Instruments.b"):
    count += 1
    if count < 2:
        print(str(item, encoding = "utf8"))
        print(pic)
    #print(item)

print(count)
'''