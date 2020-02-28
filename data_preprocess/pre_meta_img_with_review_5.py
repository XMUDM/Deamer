import numpy as np
import array

#将metadata转化好的向量和image处理成按照用户和商品评论一样的顺序
category = 'Electronics'
k_core = '_5'

print('category = ', category , ', preprocess metadata and image as what unique user and item order')
def readImageFeatures(path):
    f = open(path, 'rb')
    try:
        while True:
            asin = f.read(10)
            if asin == '' : 
                break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield asin, a.tolist()
    except EOFError:
        pass

if __name__ == '__main__':
    
    item_list = []
    item_metadata_list = []
    item_metadata_vec = []
    item_metadata_vec_inorder = []
    #load item list
    with open(category+'//PreData//unique_items_rating'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item, rating = line.split(',')
            item_list.append(item)
    print('Load item list over')


    #load item metadata list
    with open(category+'//PreData//metadata_items'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item = line.replace('\n', '')
            item_metadata_list.append(item)
    print('Load item metadata list over')
    #load item metadata vector list
    with open(category+'//feature_data//metadata_description_title_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_metadata_vec.append(list(map(float, line.split(' ')[:500])))
    print('Load item metadata vector list over')

    for item in item_list:
        if item in item_metadata_list:
            item_metadata_vec_inorder.append(item_metadata_vec[item_metadata_list.index(item)])
        else:
            item_metadata_vec_inorder.append(np.zeros(500))

    fout_meta = open(category+'//feature_data//items_metadata_vectors'+k_core+'.txt', 'w')
    for metadata_vec in item_metadata_vec_inorder:
        fout_meta.write( " ".join([str(x) for x in metadata_vec]) + "\n" )
    print('Process metadata over')


    item_in_image_list = []
    item_image_vec_list = []
    item_image_vec_inorder = []
    img_path = category + '//image_features_'+ category +'.b'
    for item, pic in readImageFeatures(img_path):
        _item = str(item, encoding = "utf8")
        if _item in item_list:
            item_in_image_list.append(_item)
            item_image_vec_list.append(pic)
    print('Load item image over')
    
    for item in item_list:
        if item in item_in_image_list:
            item_image_vec_inorder.append(item_image_vec_list[item_in_image_list.index(item)])
        else:
            item_image_vec_inorder.append(np.zeros(4096))

    fout_image = open(category+'//feature_data//items_image_vectors'+k_core+'.txt', 'w')
    for image_vec in item_image_vec_inorder:
        fout_image.write( " ".join([str(x) for x in image_vec]) + "\n" )
    print('Process image over')
    




