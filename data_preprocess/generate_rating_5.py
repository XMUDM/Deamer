import json
import gzip
import array
import numpy as np
#数据处理第二步，根据5-core的评论生成rating的csv文件

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def readImageAsin(path):
    f = open(path, 'rb')
    try:
        while True:
            asin = f.read(10)
            if asin == '' : 
                break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield asin
    except EOFError:
        pass

category = 'Movies_and_TV'
k_core = '_5'
users = []
items = []
ratings = []
count = 0

for review in parse(category + "/reviews_" + category + k_core +".json.gz"):
    users.append(review['reviewerID'])
    items.append(review['asin'])
    ratings.append(review['overall'])
    count += 1
data_amount = count
print('Read reviews over, data amount = ', data_amount)

frating = open(category + "/ratings_" + category + k_core + ".csv", 'w')
for user, item, rating in zip(users, items, ratings):
    frating.write(user + ',' + item + ',' + str(int(rating)) +'\n')
