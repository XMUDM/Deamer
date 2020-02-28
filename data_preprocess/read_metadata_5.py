import json
import gzip
import array
import numpy as np
import copy

#数据处理第一步之后，抽取出商品的metadata聚合
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

category = 'Movies_and_TV'
k_core = '_5'

unique_items = []
items_metadata = []

count = 0

for metadata in parse(category + "/meta_" + category + ".json.gz"): 
    if True:
        item_append_flag = 1
        metadata_text = ''
        if 'description' in metadata and 'title' in metadata:
            metadata_text += (metadata['description'] + metadata['title'])
        elif 'description' in metadata :
            metadata_text += metadata['description']
        elif 'title' in metadata :
            metadata_text += metadata['title']
        else :
            metadata_text += ''
        metadata_text = metadata_text.replace('\n','')
        if metadata_text != '':
            items_metadata.append(metadata_text)
            unique_items.append(metadata['asin'])
    else :
        metadata_text = ''
        if 'description' in metadata and 'title' in metadata:
            metadata_text += (metadata['description'] + metadata['title'])
        elif 'description' in metadata :
            metadata_text += metadata['description']
        elif 'title' in metadata :
            metadata_text += metadata['title']
        else :
            metadata_text += ''
            print('empty else')
        items_metadata[unique_items.index(metadata['asin'])] += metadata_text
    count += 1
    if count < 2:
        print(metadata['asin'])
    if count % 10000 == 0:
        print('Already read data amount = %d'%(count))

print(count)
print(len(unique_items), len(items_metadata))

unique_items_copy = copy.deepcopy(unique_items)
items_in_k_core = []
count = 0
if k_core != '' :
    with open(category + '/PreData/unique_items_rating' + k_core +'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item, rating = line.split(',')
            items_in_k_core.append(item)
    for item in unique_items_copy:
        count += 1
        if item not in items_in_k_core:
            del(items_metadata[unique_items.index(item)])
            del(unique_items[unique_items.index(item)])
        if count %10000 == 0:
            print('already process data amount = ', count)

print(len(unique_items), len(items_metadata))

fCategory = category + '/PreData/'
f_metadata_items = open(fCategory + 'metadata_items'+k_core+'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_items)):
    f_metadata_items.write('%s\n'%(unique_items[i]))
f_item_metadata = open(fCategory + 'metadata_description_title'+k_core+'.txt','w', encoding = 'UTF-8')
for i in range(len(unique_items)):
    f_item_metadata.write('%s\n'%(items_metadata[i]))
