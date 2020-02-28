import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import datetime
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Activation, Dropout, LSTM, Multiply, Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Average, Maximum, AveragePooling1D,MaxPooling1D, Reshape
from keras.models import Sequential, load_model
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l1, l2, l1_l2
from keras import initializers
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import dataset_loot as dataset
#from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import evaluate_loot
from keras.utils.generic_utils import get_custom_objects
from DataGenerator import MNARModelDataGenerator, MNARModelDataGenerator_just_review, MNARModelDataGenerator_no_item_review, MNARModelDataGenerator_no_item_metadata, MNARModelDataGenerator_no_item_image

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def parse_args():
    parser = argparse.ArgumentParser(description="Run DEAMER.")
    parser.add_argument('--category', nargs='?', default='Musical_Instruments',
                        help='category of data set.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--represent_dim', type=int, default=256,
                        help='Dimensions of representation.')
    parser.add_argument('--one_hot_embedding_dim', type=int, default=256,
                        help='Dimensions of one hot embedding.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--vector_size', type=int, default=500,
                        help='Text vector size.')



USER_ITEM_DIM = 256 #16 #100
LSTM_OUTPUT_DIM = 256 #16 #100
CNN_OUTPUT_DIM = 256 #16 #100
USER_ITEM_ONE_HOT_DIM = 64 #16 #100
VECTOR_SIZE = 500
EPOCHS = 80
learning_rate = 0.001
cof_loss1 = 1.2
cof_loss2 = 1
num_interval = 100 #参数实验，每多少次进行一次评估
DOC_VEC_DIM = 500
IMG_VEC_DIM = 4096
lstm_reg_params = 0.001
cnn_reg_params = 0.01
cnn_dropout = 0.25
user_lstm_dropout = 0.25
item_lstm_dropout = 0.25
resp_embedding_reg = 0.001
topK = 10
sample_ratio = 5

category = 'Musical_Instruments' # 'Electronics' 'Movies_and_TV' 'Sports_and_Outdoors' 'Health_and_Personal_Care' 'Office_Products' 'Digital_Music' 'Musical_Instruments'
procedure_id = 0 # ---1 :base_version_reverse, 2: just_use_review, 3: no_item_review, 4:no_item_metadata, 5:no_item_image, 6:parameter_experiment, 7:PE_only_train, 8:PE_only_evaluate, 9:PE_embedding
#11 :parallel_base_version_reverse, 12: parallel_just_use_review, 13: parallel_no_item_review, 14:parallel_no_item_metadata, 15:parallel_no_item_image
k_core = '_5'
loot = '_loot'
max_num = 999
rmse_list = [max_num]
lost_list = [max_num]

#为了使当某个y_true = 0时，得到该数据的loss = 0，即有y_pred = 0 
def redef_mse(y_true, y_pred):
    y_temp = K.clip(y_true * K.constant(5), 0, 1)
    return K.mean(K.square(y_pred * y_temp - y_true), axis=-1)

def cosine(x):
    return K.clip(K.cos(x), 0, 1)

#get_custom_objects().update({'cosine': Activation(cosine)})

#MNAR模型的基本版本(rating层之后是response)，rating层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)
    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)

    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)
    '''
    conv_2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                  activation='relu', name='item_img_conv2_layer', 
                  input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(maxpool_1)
    #conv_2 = Dropout(cnn_dropout)(conv_2)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp2_layer')(conv_2)
    maxpool_2 = Dropout(cnn_dropout)(maxpool_2)
    '''
    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_metadata_feature, item_review_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(content_MF_vector)
    #rating_predict = Dense(1, activation=cosine, use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    response_vector = Concatenate()([rating_predict, resp_user_latent, resp_item_latent])
    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(response_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, item_review_vector, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base version) Compiled')
    #print('Model(rating prediction with ranking model base version use cosine in rating layer) Compiled')
    return model

#MNAR模型的基本版本(response层之后是rating层)，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version_reverse(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)

    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)
    '''
    conv_2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                  activation='relu', name='item_img_conv2_layer', 
                  input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(maxpool_1)
    #conv_2 = Dropout(cnn_dropout)(conv_2)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp2_layer')(conv_2)
    maxpool_2 = Dropout(cnn_dropout)(maxpool_2)
    '''
    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_metadata_feature, item_review_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    #item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)
    #repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)
    #repsonse_predict = Dense(1, activation=cosine, use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    rating_vector = Concatenate()([repsonse_predict, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)
    #rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros', name = 'rating_prediction')(rating_vector)
    #rating_predict = Dense(1, activation=cosine, use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, item_review_vector, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base-reverse version revise cnn to 1 conv and maxpool layer use relu in conv_1) Compiled')
    #print('Model(rating prediction with ranking model base-reverse version use cosine in response layer) Compiled')
    #print('Model(rating prediction with ranking model base-reverse version use cosine in rating layer) Compiled')
    return model

#MNAR模型的基本版本(response层之后是rating层)，只使用用户和商品的评论信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version_reverse_just_use_review(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)

    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    content_MF_vector = Multiply()([user_review_feature, item_review_feature])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    rating_vector = Concatenate()([repsonse_predict, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)

    model = Model(inputs=[user_review_vector, item_review_vector, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base-reverse version just use user and item review) Compiled')

    return model

#MNAR模型的基本版本(response层之后是rating层)，去掉了商品的评论信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version_reverse_no_item_review(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)

    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)

    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_metadata_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    rating_vector = Concatenate()([repsonse_predict, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base-reverse version not use item review ) Compiled')
    return model

#MNAR模型的基本版本(response层之后是rating层)，去掉了商品的metadata信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version_reverse_no_item_metadata(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)

    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_review_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    rating_vector = Concatenate()([repsonse_predict, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)

    model = Model(inputs=[user_review_vector, item_review_vector, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base-reverse version no item metadata) Compiled')
    return model

#MNAR模型的基本版本(response层之后是rating层)，去掉了商品的image信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_base_version_reverse_no_item_image(num_users, num_items):

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))

    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    item_concat_feature = Concatenate()([item_metadata_feature, item_review_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(content_MF_vector)

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    rating_vector = Concatenate()([repsonse_predict, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(rating_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, item_review_vector, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model base-reverse version no item image) Compiled')

    return model

#MNAR模型，rating层与response层是平行的输出层
def build_MNAR_model_parallel_version(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)
    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)
    '''
    conv_2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                  activation='relu', name='item_img_conv2_layer', 
                  input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(maxpool_1)
    #conv_2 = Dropout(cnn_dropout)(conv_2)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp2_layer')(conv_2)
    maxpool_2 = Dropout(cnn_dropout)(maxpool_2)
    '''
    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_metadata_feature, item_review_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    predict_vector = Concatenate()([content_MF_vector, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(predict_vector)

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(predict_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, item_review_vector, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model , rating layer parallel to response layer version) Compiled')
    return model

#MNAR模型的基本版本(rating层与response层是平行的输出层)，只使用用户和商品的评论信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_parallel_version_just_use_review(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)

    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    content_MF_vector = Multiply()([user_review_feature, item_review_feature])

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    predict_vector = Concatenate()([content_MF_vector, resp_user_latent, resp_item_latent])
    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(predict_vector)
    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(predict_vector)

    model = Model(inputs=[user_review_vector, item_review_vector, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model parallel version just use user and item review) Compiled')

    return model

#MNAR模型的基本版本(rating层与response层是平行的输出层)，去掉了商品的评论信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_parallel_version_no_item_review(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)

    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)

    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_metadata_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    predict_vector = Concatenate()([content_MF_vector, resp_user_latent, resp_item_latent])

    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(predict_vector)
    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(predict_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model parallel version not use item review ) Compiled')
    return model

#MNAR模型的基本版本(rating层与response层是平行的输出层)，去掉了商品的metadata信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_parallel_version_no_item_metadata(num_users, num_items):

    cnn_inputshape = (64,64,1)

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    picture_features = Input(shape=(64,64,1,))
    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    #build cnn network
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', name='item_img_convl_layer', 
                 input_shape=cnn_inputshape, kernel_regularizer=l2(cnn_reg_params))(picture_features)
    #conv_1 = Dropout(cnn_dropout)(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='item_img_maxp1_layer')(conv_1)
    maxpool_1 = Dropout(cnn_dropout)(maxpool_1)

    flat = Flatten()(maxpool_1)
    pic_feature = Dense(CNN_OUTPUT_DIM, activation='relu', name='item_img_dense_layer')(flat)
    item_concat_feature = Concatenate()([item_review_feature, pic_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    predict_vector = Concatenate()([content_MF_vector, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(predict_vector)
    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(predict_vector)

    model = Model(inputs=[user_review_vector, item_review_vector, picture_features, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model parallel version no item metadata) Compiled')
    return model

#MNAR模型的基本版本(rating层与response层是平行的输出层)，去掉了商品的image信息，response层与user和item的1-hot embedding的最后一层拼接
def build_MNAR_model_parallel_version_no_item_image(num_users, num_items):

    lstm_inputshape = (1, VECTOR_SIZE)
    user_review_vector = Input(shape=(1,DOC_VEC_DIM,))
    metadata_vectors = Input(shape=(1,DOC_VEC_DIM,))
    item_review_vector = Input(shape=(1,DOC_VEC_DIM,))

    #build lstm network
    user_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = user_lstm_dropout, name='user_rev_layer', return_sequences=False)(user_review_vector)
    #user_review_feature = Dropout(user_lstm_dropout)(user_review_feature)
    item_metadata_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_meta_layer', return_sequences=False)(metadata_vectors)
    #item_metadata_feature = Dropout(item_lstm_dropout)(item_metadata_feature)
    item_review_feature = LSTM(LSTM_OUTPUT_DIM, activation='softsign', input_shape=lstm_inputshape, 
            kernel_regularizer=l2(lstm_reg_params), dropout = item_lstm_dropout, name='item_rev_layer', return_sequences=False)(item_review_vector)
    #item_review_feature = Dropout(item_lstm_dropout)(item_review_feature)

    item_concat_feature = Concatenate()([item_metadata_feature, item_review_feature])
    item_feature = Dense(USER_ITEM_DIM, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name='item_map_layer')(item_concat_feature)
    content_MF_vector = Multiply()([user_review_feature, item_feature])

    user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_mlp_input')
    item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_mlp_input')
    
    Resp_Embedding_User = Embedding(input_dim = num_users, output_dim = USER_ITEM_ONE_HOT_DIM, name = "resp_embedding_user",
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    Resp_Embedding_Item = Embedding(input_dim = num_items, output_dim = USER_ITEM_ONE_HOT_DIM, name = 'resp_embedding_item',
                                  embeddings_initializer='uniform',  embeddings_regularizer = l2(resp_embedding_reg), input_length=1)
    resp_user_latent = Flatten()(Resp_Embedding_User(user_mlp_input))
    resp_item_latent = Flatten()(Resp_Embedding_Item(item_mlp_input))
    

    predict_vector = Concatenate()([content_MF_vector, resp_user_latent, resp_item_latent])

    rating_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'rating_prediction')(predict_vector)
    repsonse_predict = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros', name = 'repsonse_layer')(predict_vector)

    model = Model(inputs=[user_review_vector, metadata_vectors, item_review_vector, user_mlp_input, item_mlp_input], 
                  outputs = [rating_predict, repsonse_predict])
    losses = {
        "rating_prediction" : redef_mse,
        "repsonse_layer" : "binary_crossentropy",
    }
    metrics_of_all = {
        "rating_prediction" : "mae",
        "repsonse_layer" : "accuracy",
    }
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses, 
        loss_weights={'rating_prediction': cof_loss1, 'repsonse_layer': cof_loss2}, 
        metrics = metrics_of_all)
    print('Model(rating prediction with ranking model parallel version no item image) Compiled')

    return model

#使用分批次方法训练MNAR模型
def train_model_MNAR_base_version(model, identifier=''):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/'+identifier+'multi_modalities_base_version_reverse' + k_core + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        '''
        for i in range(len(predict_rating)):
            print(np.abs(predict_rating[i]*5 - rating_test[i]*5), end = ' ')
        print()
        '''

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
        #rmse, mae = evaluate_loot.evaluate_rating_model(rating_test*5, predict_rating*5)
        #print(rating_test)
        #print(predict_rating)
        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_'+ str(cof_loss1) + '.model' + k_core)

        '''
        rmse_list.append(RMSE)
        if rmse_list[-1] >= rmse_list[-2]:
            _lr = backend.get_value(model.optimizer.lr)
            backend.set_value(model.optimizer.lr, _lr * 0.1) 
            print('learning rate descent')
        '''

        lost_list.append(hist.history['loss'][0])

        final_epoch = epoch+1

        
        if lost_list[-1] > (lost_list[-2]*1.01) : 
        #if lost_list[-1] > lost_list[-2] : 
            print('Training early stop')
            break
        

    print('All Best Model Saved')
    MNAR_model_filepath = category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    #print('Start rating evaluate')
    predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
    predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )
    RMSE_ra = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
    MAE_ra = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
    print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')
    #print('Final RMSE = %.4f, '%RMSE_ra, 'MAE = %.4f, '%MAE_ra, 'at epoch %d '%final_epoch, ' (use rating)')
    #f_result_out.write('Best rating performance use rating: Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f,'%best_mae + ' at epoch %d'%best_epoch + '\n')
    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    predict_response = np.clip(np.array(predict_response).reshape(len(predict_response)), 0.2, 1)
    RMSE_re = np.sqrt(np.mean(np.square(predict_response*5-rating_test*5), axis=-1))
    MAE_re = np.mean(np.abs(rating_test*5 - predict_response*5), axis=-1)
    print('Best RMSE = %.4f, '%RMSE_re, 'MAE = %.4f, '%MAE_re, 'at epoch %d '%best_epoch, ' (use response)')
    f_result_out.write('Rating performance use response : Best RMSE = %.4f, '%RMSE_re + 'MAE = %.4f, '%MAE_re + 'at epoch %d '%best_epoch + '\n')

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型（只使用用户和商品的评论信息）
def train_model_MNAR_base_version_just_use_review(model, identifier=''):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator_just_review(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/'+identifier+'multi_modalities_base_version_reverse_just_use_review' + k_core + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_rev_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        '''
        for i in range(len(predict_rating)):
            print(np.abs(predict_rating[i]*5 - rating_test[i]*5), end = ' ')
        print()
        '''

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
        #rmse, mae = evaluate_loot.evaluate_rating_model(rating_test*5, predict_rating*5)
        #print(rating_test)
        #print(predict_rating)
        if RMSE <= best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_just_use_review_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core)

        '''
        rmse_list.append(RMSE)
        if rmse_list[-1] >= rmse_list[-2]:
            _lr = backend.get_value(model.optimizer.lr)
            backend.set_value(model.optimizer.lr, _lr * 0.1) 
            print('learning rate descent')
        '''

        lost_list.append(hist.history['loss'][0])

        final_epoch = epoch+1

        
        if lost_list[-1] > (lost_list[-2]*1.01) : 
        #if lost_list[-1] > lost_list[-2] : 
            print('Training early stop')
            #break
        

    print('All Best Model Saved')
    MNAR_model_filepath = category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_just_use_review_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    #print('Start rating evaluate')
    predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_rev_vec_test, user_mlp_test_input, item_mlp_test_input])
    predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )
    RMSE_ra = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
    MAE_ra = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
    print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')
    #print('Final RMSE = %.4f, '%RMSE_ra, 'MAE = %.4f, '%MAE_ra, 'at epoch %d '%final_epoch, ' (use rating)')
    #f_result_out.write('Best rating performance use rating: Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f,'%best_mae + ' at epoch %d'%best_epoch + '\n')
    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    predict_response = np.clip(np.array(predict_response).reshape(len(predict_response)), 0.2, 1)
    RMSE_re = np.sqrt(np.mean(np.square(predict_response*5-rating_test*5), axis=-1))
    MAE_re = np.mean(np.abs(rating_test*5 - predict_response*5), axis=-1)
    print('Best RMSE = %.4f, '%RMSE_re, 'MAE = %.4f, '%MAE_re, 'at epoch %d '%best_epoch, ' (use response)')
    f_result_out.write('Rating performance use response : Best RMSE = %.4f, '%RMSE_re + 'MAE = %.4f, '%MAE_re + 'at epoch %d '%best_epoch + '\n')

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model_just_use_review(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型（去掉了商品的评论信息）
def train_model_MNAR_base_version_no_item_review(model, identifier=''):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator_no_item_review(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_meta_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/'+identifier+'multi_modalities_base_version_reverse_no_item_review' + k_core + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')

        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_review_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core)

        lost_list.append(hist.history['loss'][0])

        final_epoch = epoch+1

        
        if lost_list[-1] > (lost_list[-2]*1.01) : 
            print('Training early stop')
            #break
        

    print('All Best Model Saved')
    MNAR_model_filepath = category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_review_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    #print('Start rating evaluate')
    predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
    predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )
    RMSE_ra = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
    MAE_ra = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
    print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')

    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    predict_response = np.clip(np.array(predict_response).reshape(len(predict_response)), 0.2, 1)
    RMSE_re = np.sqrt(np.mean(np.square(predict_response*5-rating_test*5), axis=-1))
    MAE_re = np.mean(np.abs(rating_test*5 - predict_response*5), axis=-1)
    print('Best RMSE = %.4f, '%RMSE_re, 'MAE = %.4f, '%MAE_re, 'at epoch %d '%best_epoch, ' (use response)')
    f_result_out.write('Rating performance use response : Best RMSE = %.4f, '%RMSE_re + 'MAE = %.4f, '%MAE_re + 'at epoch %d '%best_epoch + '\n')

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model_no_item_review(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型（去掉了商品的metadata信息）
def train_model_MNAR_base_version_no_item_metadata(model, identifier=''):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator_no_item_metadata(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/'+identifier+'multi_modalities_base_version_reverse_no_item_metadata' + k_core + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')

        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_metadata_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core)

        lost_list.append(hist.history['loss'][0])

        final_epoch = epoch+1

        
        if lost_list[-1] > (lost_list[-2]*1.01) : 
        #if lost_list[-1] > lost_list[-2] : 
            print('Training early stop')
            #break
        

    print('All Best Model Saved')
    MNAR_model_filepath = category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_metadata_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    #print('Start rating evaluate')
    predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
    predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )
    RMSE_ra = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
    MAE_ra = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
    print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')
    #print('Final RMSE = %.4f, '%RMSE_ra, 'MAE = %.4f, '%MAE_ra, 'at epoch %d '%final_epoch, ' (use rating)')
    #f_result_out.write('Best rating performance use rating: Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f,'%best_mae + ' at epoch %d'%best_epoch + '\n')
    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    predict_response = np.clip(np.array(predict_response).reshape(len(predict_response)), 0.2, 1)
    RMSE_re = np.sqrt(np.mean(np.square(predict_response*5-rating_test*5), axis=-1))
    MAE_re = np.mean(np.abs(rating_test*5 - predict_response*5), axis=-1)
    print('Best RMSE = %.4f, '%RMSE_re, 'MAE = %.4f, '%MAE_re, 'at epoch %d '%best_epoch, ' (use response)')
    f_result_out.write('Rating performance use response : Best RMSE = %.4f, '%RMSE_re + 'MAE = %.4f, '%MAE_re + 'at epoch %d '%best_epoch + '\n')

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model_no_item_metadata(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型（去掉了商品的image信息）
def train_model_MNAR_base_version_no_item_image(model, identifier=''):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator_no_item_image(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_meta_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/'+identifier+'multi_modalities_base_version_reverse_no_item_image' + k_core + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        '''
        for i in range(len(predict_rating)):
            print(np.abs(predict_rating[i]*5 - rating_test[i]*5), end = ' ')
        print()
        '''

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
        #rmse, mae = evaluate_loot.evaluate_rating_model(rating_test*5, predict_rating*5)
        #print(rating_test)
        #print(predict_rating)
        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_image_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core)

        '''
        rmse_list.append(RMSE)
        if rmse_list[-1] >= rmse_list[-2]:
            _lr = backend.get_value(model.optimizer.lr)
            backend.set_value(model.optimizer.lr, _lr * 0.1) 
            print('learning rate descent')
        '''

        lost_list.append(hist.history['loss'][0])

        final_epoch = epoch+1

        
        if lost_list[-1] > (lost_list[-2]*1.01) : 
        #if lost_list[-1] > lost_list[-2] : 
            print('Training early stop')
            #break
        

    print('All Best Model Saved')
    MNAR_model_filepath = category+'/model/'+identifier+'Multi_modalities_model_base_version_reverse_no_item_image_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1_-'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    #print('Start rating evaluate')
    predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, user_mlp_test_input, item_mlp_test_input])
    predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )
    RMSE_ra = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
    MAE_ra = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
    print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')

    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    predict_response = np.clip(np.array(predict_response).reshape(len(predict_response)), 0.2, 1)
    RMSE_re = np.sqrt(np.mean(np.square(predict_response*5-rating_test*5), axis=-1))
    MAE_re = np.mean(np.abs(rating_test*5 - predict_response*5), axis=-1)
    print('Best RMSE = %.4f, '%RMSE_re, 'MAE = %.4f, '%MAE_re, 'at epoch %d '%best_epoch, ' (use response)')
    f_result_out.write('Rating performance use response : Best RMSE = %.4f, '%RMSE_re + 'MAE = %.4f, '%MAE_re + 'at epoch %d '%best_epoch + '\n')

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model_no_item_image(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型，参数实验，每十次一次evaluation，最后再对best_rmse_performance的模型进行evaluate
def train_model_MNAR_base_version_parameter_experiment(model):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/multi_modalities_base_version_reverse_parameter_experiment' + k_core + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        '''
        for i in range(len(predict_rating)):
            print(np.abs(predict_rating[i]*5 - rating_test[i]*5), end = ' ')
        print()
        '''

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)

        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)
        elif (epoch+1) % num_interval == 0:
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)

        if (epoch+1) % num_interval == 0:
            f_result_out.write('EPOCH = %d, rating model performance :'%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
            startTime = datetime.datetime.now()
            print('Ranking evaluating...EPOCH = %d: '%(epoch+1))
            HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
            #print('EPOCH = %d, '%(epoch+1), 'HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
            f_result_out.write('EPOCH = %d, ranking model performance :'%(epoch+1) + ' use response : HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + ', MRR = %.4f '% MRR_re + '\n')
            #print('HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
            print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

        final_epoch = epoch+1

    print('All Best Model and interval evaluation model Saved')
    if best_epoch % num_interval == 0:
        print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')
        f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
        print('Best ranking performance please check out the previous outputs at epoch %d '%best_epoch)
        return 
    f_result_out.write('Best Rating performance : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    startTime = datetime.datetime.now()
    print('Start ranking evaluate for the epoch of best rating model')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
    print('(Best rating model for ranking evaluation)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model for ranking evaluation)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型，参数实验，每十次一次rating evaluation，最后再对best_rmse_performance的模型进行evaluate
def train_model_MNAR_base_version_PE_only_train(model):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/multi_modalities_base_version_reverse_PE_only_train' + k_core + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('Rating model performance : EPOCH = %d, '%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)
        elif (epoch+1) % num_interval == 0:
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)

        final_epoch = epoch+1

    print('Best at epoch %d'%best_epoch)
    f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')

#使用分批次方法训练MNAR模型，参数实验，只对训练完成的模型进行evaluate
def train_model_MNAR_base_version_PE_only_evaluate(start_epoch = 1, best_epoch = 65):
    print('start at epoch %d'%start_epoch, ' and best epoch at %d '%best_epoch)
    f_result_out = open(category+'/result/multi_modalities_base_version_reverse_PE_only_train' + k_core + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.txt', 'a')

    for epoch in range(start_epoch, EPOCHS+1):
        if epoch % num_interval == 0:
            MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core
            model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
            startTime = datetime.datetime.now()
            print('Ranking evaluating...EPOCH = %d: '%epoch)
            HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
            f_result_out.write('EPOCH = %d, '%epoch + ' use response : HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + ', MRR = %.4f '% MRR_re + '\n')
            print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

    print('All Best Model and interval evaluation model Saved')
    if best_epoch % num_interval == 0:
        print('Best ranking performance please check out the previous outputs at epoch %d '%best_epoch)
        return 
    MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(best_epoch) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    startTime = datetime.datetime.now()
    print('Start ranking evaluate ')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
    print('(Best rating model)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

#使用分批次方法训练MNAR模型，one-hot embedding维度的参数实验，每十次一次evaluation，最后再对best_rmse_performance的模型进行evaluate
def train_model_MNAR_base_version_parameter_experiment_onehot_embedding(model):

    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    user_sample_train_list, item_sample_train_list, user_mlp_sample_input, item_mlp_sample_input = \
    dataset.sampling_response_with_unrating_training(user_list, item_list, category, k_core, sample_ratio)
    response_sample_train = np.zeros((len(user_mlp_sample_input),1))

    user_mlp_sample_input = np.array(user_mlp_sample_input).reshape(len(user_mlp_sample_input),1)
    item_mlp_sample_input = np.array(item_mlp_sample_input).reshape(len(item_mlp_sample_input),1)

    #load initial training set list
    user_initial_train_list, item_initial_train_list, rating_train, user_mlp_initial_input, item_mlp_initial_input = \
    dataset.load_initial_training_list(user_list, item_list, category, k_core, loot)
    rating_train = np.array(rating_train).reshape(len(rating_train),1)
    user_mlp_initial_input = np.array(user_mlp_initial_input).reshape(len(user_mlp_initial_input),1)
    item_mlp_initial_input = np.array(item_mlp_initial_input).reshape(len(item_mlp_initial_input),1)
    response_initial_train = np.ones((len(rating_train),1)).reshape(len(rating_train),1)

    #concat training data
    user_mlp_input_concat = np.vstack((user_mlp_initial_input, user_mlp_sample_input))
    item_mlp_input_concat = np.vstack((item_mlp_initial_input, item_mlp_sample_input))
    rating_train_concat = np.vstack((rating_train, response_sample_train))
    response_train_concat = np.vstack((response_initial_train, response_sample_train))
    print('Concat amount: training set = ', len(rating_train_concat))

    #load initial testing set list
    user_initial_test_list, item_initial_test_list, rating_test, user_mlp_test_input, item_mlp_test_input = \
    dataset.load_initial_testing_list(user_list, item_list, category, k_core, loot)
    rating_test = np.array(rating_test)
    user_initial_rev_vec_test = dataset.load_vec_by_list(user_initial_test_list, category, k_core, 'users_reviews', user_list)
    item_initial_rev_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_reviews', item_list)
    item_initial_meta_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_metadata', item_list)
    item_initial_img_vec_test = dataset.load_vec_by_list(item_initial_test_list, category, k_core, 'items_image', item_list)
    user_mlp_test_input = np.array(user_mlp_test_input).reshape(len(user_mlp_test_input),1)
    item_mlp_test_input = np.array(item_mlp_test_input).reshape(len(item_mlp_test_input),1)
    user_initial_rev_vec_test = user_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_rev_vec_test = item_initial_rev_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_meta_vec_test = item_initial_meta_vec_test.reshape(len(rating_test), 1, DOC_VEC_DIM)
    item_initial_img_vec_test = item_initial_img_vec_test.reshape(len(rating_test), 64, 64, 1)
    print('Initial testing dataset reshape over: amount = ', len(rating_test))

    #load vectors from files
    user_doc_vec_list = []
    with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            user_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    user_doc_vec = np.array(user_doc_vec_list).reshape((len(user_doc_vec_list), 1, DOC_VEC_DIM))

    item_doc_vec_list = []
    with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_doc_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_doc_vec = np.array(item_doc_vec_list).reshape((len(item_doc_vec_list), 1, DOC_VEC_DIM))

    item_meta_vec_list = []
    with open(category+'/feature_data/items_metadata_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_meta_vec_list.append(list(map(float, line.split(' ')[:DOC_VEC_DIM])))
    item_meta_vec = np.array(item_meta_vec_list).reshape((len(item_meta_vec_list), 1, DOC_VEC_DIM))

    item_img_vec_list = []
    with open(category+'/feature_data/items_image_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
        for line in f.readlines() :
            item_img_vec_list.append(list(map(float, line.split(' ')[:IMG_VEC_DIM])))
    item_img_vec = np.array(item_img_vec_list).reshape((len(item_img_vec_list), 64, 64, 1))
    print('Load vector over')

# Parameters
    params = {'batch_size': 256,
              'shuffle': True}

    steps_pp_args = len(rating_train_concat)/params['batch_size']
    training_generator = MNARModelDataGenerator(user_mlp_input_concat, item_mlp_input_concat, rating_train_concat, response_train_concat, category, k_core, \
                        user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, **params)

    best_rmse, best_mae, best_epoch = 10, 10, 0
    final_epoch = 0
    f_result_out = open(category+'/result/multi_modalities_base_version_reverse_parameter_experiment' + k_core + '_one_hot_DIM' + str(USER_ITEM_ONE_HOT_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.txt', 'w')
    f_result_out.write('USER_ITEM_DIM = '+str(USER_ITEM_DIM)+', learning_rate = '+str(learning_rate)+', lstm_reg_params = '+str(lstm_reg_params) + '\n')
    f_result_out.write('USER_ITEM_ONE_HOT_DIM = '+str(USER_ITEM_ONE_HOT_DIM)+ ', cof_loss1 = '+str(cof_loss1)+ ', cof_loss2 = '+str(cof_loss2) + '\n')
    f_result_out.write('cnn_reg_params = '+str(cnn_reg_params)+', cnn_dropout = '+ str(cnn_dropout)+ ', lstm_dropout = '+ str(user_lstm_dropout) + '\n')
    for epoch in range(EPOCHS):
        hist = model.fit_generator(generator=training_generator,
                          epochs=1, verbose=2, steps_per_epoch = steps_pp_args, shuffle=True)

        predict_rating, predict_response = model.predict([user_initial_rev_vec_test, item_initial_meta_vec_test, item_initial_rev_vec_test, item_initial_img_vec_test, user_mlp_test_input, item_mlp_test_input])
        predict_rating = np.clip(np.array(predict_rating).reshape(len(predict_rating)), 0.2, 1 )

        RMSE = np.sqrt(np.mean(np.square(predict_rating*5-rating_test*5), axis=-1))
        MAE = np.mean(np.abs(rating_test*5 - predict_rating*5), axis=-1)
        print('EPOCH = %d, '%(epoch+1), 'RMSE = %.4f, '%RMSE, 'MAE = %.4f '%MAE)
        f_result_out.write('EPOCH = %d, rating model performance :'%(epoch+1) + 'RMSE = %.4f, '%RMSE + 'MAE = %.4f '% MAE + '\n')
        if RMSE < best_rmse :
            best_rmse = RMSE
            best_mae = MAE
            best_epoch = epoch+1
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_one_hot_DIM' + str(USER_ITEM_ONE_HOT_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)
        elif (epoch+1) % num_interval == 0:
            model.save(category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(epoch+1) + '_one_hot_DIM' + str(USER_ITEM_ONE_HOT_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core)

        if (epoch+1) % num_interval == 0:
            startTime = datetime.datetime.now()
            print('Ranking evaluating...EPOCH = %d: '%(epoch+1))
            HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
            f_result_out.write('EPOCH = %d, ranking model performance :'%(epoch+1) + ' use response : HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + ', MRR = %.4f '% MRR_re + '\n')
            print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))

        final_epoch = epoch+1

    print('All Best Model and interval evaluation model Saved')
    if best_epoch % num_interval == 0:
        print('Best RMSE = %.4f, '%best_rmse, 'MAE = %.4f,'%best_mae, ' at epoch %d'%best_epoch, ' (use rating)')
        f_result_out.write('Rating performance use rating : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
        print('Best ranking performance please check out the previous outputs at epoch %d '%best_epoch)
        return 
    f_result_out.write('Best Rating performance : Best RMSE = %.4f, '%best_rmse + 'MAE = %.4f, '%best_mae + 'at epoch %d '%best_epoch + '\n')
    MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(best_epoch) + '_one_hot_DIM' + str(USER_ITEM_ONE_HOT_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    print('Load best epoch of model in performance of rating over, start evaluating the epoch in %d of the model'%best_epoch)

    startTime = datetime.datetime.now()
    print('Start ranking evaluate for the epoch of best rating model')
    HitRatio_re, NDCG_re, MRR_re, HitRatio_ra, NDCG_ra, MRR_ra = evaluate_loot.evaluate_response_on_MNAR_base_model(model, topK, category, k_core, loot)
    print('(Best rating model for ranking evaluation)HitRatio = %.4f'%HitRatio_re, ',NDCG = %.4f'%NDCG_re, ',MRR = %.4f'%MRR_re, ' (use response)')
    f_result_out.write('Ranking performance use response : Best HitRatio = %.4f, ' % HitRatio_re + 'NDCG = %.4f '% NDCG_re + 'MRR = %.4f '% MRR_re + '\n')

    print('(Best rating model for ranking evaluation)HitRatio = %.4f'%HitRatio_ra, ',NDCG = %.4f'%NDCG_ra, ',MRR = %.4f'%MRR_ra, ' (use rating)')
    f_result_out.write('Ranking performance use rating : Best HitRatio = %.4f, ' % HitRatio_ra + 'NDCG = %.4f '% NDCG_ra + 'MRR = %.4f '% MRR_ra + '\n')
    print('Evaluation cost %s s'%((datetime.datetime.now() - startTime).seconds))


if __name__ == '__main__':
    
    user_list = dataset.load_unique_user_list(category+'//PreData//unique_users_rating'+k_core+'.txt')
    item_list = dataset.load_unique_item_list(category+'//PreData//unique_items_rating'+k_core+'.txt')
    num_users = len(user_list)
    num_items = len(item_list)
    
    print('USER_ITEM_DIM = ', USER_ITEM_DIM, ', USER_ITEM_ONE_HOT_DIM = ', USER_ITEM_ONE_HOT_DIM, ', learning_rate = ', learning_rate, ', sample_ratio = ', sample_ratio)
    print('lstm_reg_params = ', lstm_reg_params, ', resp_embedding_reg = ', resp_embedding_reg, ', cnn_reg_params = ', cnn_reg_params )
    print('cof_loss1 = ', cof_loss1, ', cof_loss2 = ', cof_loss2, ', cnn_dropout = ', cnn_dropout, ', lstm_dropout = ', user_lstm_dropout)

    '''
    model = build_MNAR_model_base_version(num_users, num_items)
    print('category = ', category, ', start training')
    train_model_MNAR_base_version(model)
    '''

    '''
    bestEpoch = 99
    MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_' + str(bestEpoch+1) + '_DIM' + str(USER_ITEM_DIM) + 'cof_loss1'+ str(cof_loss1) + '.model' + k_core
    model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
    '''
    if procedure_id == 0:
        model = build_MNAR_model_base_version_reverse(num_users, num_items)
        lstm_weights = model.get_layer('rating_prediction').get_weights()
        lstm_weights = np.array(lstm_weights)
        print(lstm_weights)
        print(len(lstm_weights))
        print(len(lstm_weights[0]))
        print(len(lstm_weights[0][0]))

    if procedure_id == 1:
        model = build_MNAR_model_base_version_reverse(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version(model)
    elif procedure_id == 2:
        model = build_MNAR_model_base_version_reverse_just_use_review(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_just_use_review(model)
    elif procedure_id == 3:
        model = build_MNAR_model_base_version_reverse_no_item_review(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_review(model)
    elif procedure_id == 4:
        model = build_MNAR_model_base_version_reverse_no_item_metadata(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_metadata(model)
    elif procedure_id == 5:
        model = build_MNAR_model_base_version_reverse_no_item_image(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_image(model)
    elif procedure_id == 6:
        print('Start parameter experiment...')
        model = build_MNAR_model_base_version_reverse(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_parameter_experiment(model)
    elif procedure_id == 7:
        print('Start parameter experiment only train...')
        model = build_MNAR_model_base_version_reverse(num_users, num_items)
        #MNAR_model_filepath = category+'/model/Multi_modalities_model_base_version_reverse_parameter_experiment_' + str(30) + '_DIM' + str(USER_ITEM_DIM) + '_cof_loss1_'+ str(cof_loss1) + '.model' + k_core
        #model = load_model(MNAR_model_filepath, custom_objects={'redef_mse': redef_mse})
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_PE_only_train(model)
    elif procedure_id == 8:
        print('Start parameter experiment only evaluate...')
        train_model_MNAR_base_version_PE_only_evaluate(start_epoch = 100, best_epoch = 68)
    elif procedure_id == 9:
        print('Start parameter experiment of one hot embedding...')
        model = build_MNAR_model_base_version_reverse(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_parameter_experiment_onehot_embedding(model)
    elif procedure_id == 11:
        model = build_MNAR_model_parallel_version(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version(model, 'Parallel_')
    elif procedure_id == 12:
        model = build_MNAR_model_parallel_version_just_use_review(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_just_use_review(model, 'Parallel_')
    elif procedure_id == 13:
        model = build_MNAR_model_parallel_version_no_item_review(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_review(model, 'Parallel_')
    elif procedure_id == 14:
        model = build_MNAR_model_parallel_version_no_item_metadata(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_metadata(model, 'Parallel_')
    elif procedure_id == 15:
        model = build_MNAR_model_parallel_version_no_item_image(num_users, num_items)
        print('category = ', category, ', start training')
        train_model_MNAR_base_version_no_item_image(model, 'Parallel_')
    else:
        print('procedure_id is set the wrong number')
