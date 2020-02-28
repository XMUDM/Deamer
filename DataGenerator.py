import numpy as np
import keras
import copy

class ReviewDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, labels, category, k_core, batch_size=32, doc_dim=(1, 500), n_channels=4,
                  shuffle=True):
        'Initialization'
        self.doc_dim = doc_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        #print(np.empty((self.batch_size, *self.doc_dim, self.n_channels)))
        #print(np.empty((10, *(1,10), 4)))
        self.user_doc_vec_list = []
        with open(category+'/feature_data/users_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
            for line in f.readlines() :
                self.user_doc_vec_list.append(list(map(float, line.split(' ')[:self.doc_vec_dim])))
        self.item_doc_vec_list = []
        with open(category+'/feature_data/items_reviews_vectors'+k_core+'.txt', 'r', encoding = 'UTF-8') as f:
            for line in f.readlines() :
                self.item_doc_vec_list.append(list(map(float, line.split(' ')[:self.doc_vec_dim])))


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_label_IDs_temp = np.array([self.labels[k] for k in indexes])

        # Generate data
        X_user, X_item = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y = list_label_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item, list_user_IDs_temp, list_item_IDs_temp], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X_user = np.empty((self.batch_size, *self.doc_dim, self.n_channels))
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        #X_item = np.empty((self.batch_size, *self.doc_dim, self.n_channels))
        X_item = np.zeros((self.batch_size, 1, self.doc_vec_dim))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = np.array(self.user_doc_vec_list[list_user_IDs_temp[i][0]]).reshape(1, self.doc_vec_dim).copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item[i] = np.array(self.item_doc_vec_list[list_item_IDs_temp[i][0]]).reshape(1, self.doc_vec_dim).copy()

        return X_user, X_item

class MNARDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, labels, category, k_core, 
                  user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_doc_vec = item_doc_vec
        self.item_meta_vec = item_meta_vec
        self.item_img_vec = item_img_vec
        #print(np.empty((self.batch_size, *self.doc_dim, self.n_channels)))
        #print(np.empty((10, *(1,10), 4)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_label_IDs_temp = np.array([self.labels[k] for k in indexes])

        # Generate data
        X_user, X_item_meta, X_item_rev, X_item_img = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y = list_label_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_meta, X_item_rev, X_item_img, list_user_IDs_temp, list_item_IDs_temp], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_meta = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_img = np.zeros((self.batch_size, 64, 64, 1))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_meta[i] = self.item_meta_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_rev[i] = self.item_doc_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_img[i] = self.item_img_vec[list_item_IDs_temp[i][0]].copy()

        return X_user, X_item_meta, X_item_rev, X_item_img

class MNARModelDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, ratings, response, category, k_core, 
                  user_doc_vec, item_doc_vec, item_meta_vec, item_img_vec, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.ratings = ratings
        self.response = response
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_doc_vec = item_doc_vec
        self.item_meta_vec = item_meta_vec
        self.item_img_vec = item_img_vec

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_rating_IDs_temp = np.array([self.ratings[k] for k in indexes])
        list_response_IDs_temp = np.array([self.response[k] for k in indexes])

        # Generate data
        X_user, X_item_meta, X_item_rev, X_item_img = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y_rating = list_rating_IDs_temp.copy()
        y_response = list_response_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_meta, X_item_rev, X_item_img, list_user_IDs_temp, list_item_IDs_temp], [y_rating, y_response]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_meta = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_img = np.zeros((self.batch_size, 64, 64, 1))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_meta[i] = self.item_meta_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_rev[i] = self.item_doc_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_img[i] = self.item_img_vec[list_item_IDs_temp[i][0]].copy()

        return X_user, X_item_meta, X_item_rev, X_item_img

class MNARModelDataGenerator_just_review(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, ratings, response, category, k_core, 
                  user_doc_vec, item_doc_vec, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.ratings = ratings
        self.response = response
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_doc_vec = item_doc_vec

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_rating_IDs_temp = np.array([self.ratings[k] for k in indexes])
        list_response_IDs_temp = np.array([self.response[k] for k in indexes])

        # Generate data
        X_user, X_item_rev= self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y_rating = list_rating_IDs_temp.copy()
        y_response = list_response_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_rev, list_user_IDs_temp, list_item_IDs_temp], [y_rating, y_response]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_rev[i] = self.item_doc_vec[list_item_IDs_temp[i][0]].copy()


        return X_user, X_item_rev

class MNARModelDataGenerator_no_item_review(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, ratings, response, category, k_core, 
                  user_doc_vec, item_meta_vec, item_img_vec, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.ratings = ratings
        self.response = response
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_meta_vec = item_meta_vec
        self.item_img_vec = item_img_vec

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_rating_IDs_temp = np.array([self.ratings[k] for k in indexes])
        list_response_IDs_temp = np.array([self.response[k] for k in indexes])

        # Generate data
        X_user, X_item_meta, X_item_img = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y_rating = list_rating_IDs_temp.copy()
        y_response = list_response_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_meta, X_item_img, list_user_IDs_temp, list_item_IDs_temp], [y_rating, y_response]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_meta = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_img = np.zeros((self.batch_size, 64, 64, 1))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_meta[i] = self.item_meta_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_img[i] = self.item_img_vec[list_item_IDs_temp[i][0]].copy()

        return X_user, X_item_meta, X_item_img

class MNARModelDataGenerator_no_item_metadata(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, ratings, response, category, k_core, 
                  user_doc_vec, item_doc_vec, item_img_vec, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.ratings = ratings
        self.response = response
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_doc_vec = item_doc_vec
        self.item_img_vec = item_img_vec

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_rating_IDs_temp = np.array([self.ratings[k] for k in indexes])
        list_response_IDs_temp = np.array([self.response[k] for k in indexes])

        # Generate data
        X_user, X_item_rev, X_item_img = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y_rating = list_rating_IDs_temp.copy()
        y_response = list_response_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_rev, X_item_img, list_user_IDs_temp, list_item_IDs_temp], [y_rating, y_response]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_img = np.zeros((self.batch_size, 64, 64, 1))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_rev[i] = self.item_doc_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_img[i] = self.item_img_vec[list_item_IDs_temp[i][0]].copy()

        return X_user, X_item_rev, X_item_img


class MNARModelDataGenerator_no_item_image(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_user_IDs, list_item_IDs, ratings, response, category, k_core, 
                  user_doc_vec, item_doc_vec, item_meta_vec,  batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.ratings = ratings
        self.response = response
        self.list_user_IDs = list_user_IDs
        self.list_item_IDs = list_item_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.doc_vec_dim = 500
        self.user_doc_vec = user_doc_vec
        self.item_doc_vec = item_doc_vec
        self.item_meta_vec = item_meta_vec

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_user_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_user_IDs_temp = np.array([self.list_user_IDs[k] for k in indexes])
        list_item_IDs_temp = np.array([self.list_item_IDs[k] for k in indexes])
        list_rating_IDs_temp = np.array([self.ratings[k] for k in indexes])
        list_response_IDs_temp = np.array([self.response[k] for k in indexes])

        # Generate data
        X_user, X_item_meta, X_item_rev, = self.__data_generation(list_user_IDs_temp, list_item_IDs_temp)
        y_rating = list_rating_IDs_temp.copy()
        y_response = list_response_IDs_temp.copy()

        list_user_IDs_temp = list_user_IDs_temp.reshape((len(indexes),1))
        list_item_IDs_temp = list_item_IDs_temp.reshape((len(indexes),1))

        return [X_user, X_item_meta, X_item_rev, list_user_IDs_temp, list_item_IDs_temp], [y_rating, y_response]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_user_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_user_IDs_temp, list_item_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_user = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_rev = np.zeros((self.batch_size, 1, self.doc_vec_dim))
        X_item_meta = np.zeros((self.batch_size, 1, self.doc_vec_dim))

        # Generate data
        for i in range(len(list_user_IDs_temp)):
            # Store sample
            X_user[i] = self.user_doc_vec[list_user_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_meta[i] = self.item_meta_vec[list_item_IDs_temp[i][0]].copy()

        for i in range(len(list_item_IDs_temp)):
            # Store sample
            X_item_rev[i] = self.item_doc_vec[list_item_IDs_temp[i][0]].copy()

        return X_user, X_item_meta, X_item_rev

