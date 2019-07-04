import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
import cv2
import itertools
import math

RNASET = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3,
          'N': 'n', 'Y': 'y', 'R': 'r', 'W': 'w',
          'M': 'm', 'K': 'k', 'S': 's', 'B': 'b',
          'H': 'h', 'D': 'd', 'V': 'v', 'X': 'x'
          }


# =========================== one_hot_encoding ========================================================
def encoding_1hot(seq, seq_length=312):
    arr = np.zeros((1, seq_length, 4, 1))
    for i, c in enumerate(seq):
        if i < seq_length:
            if type(RNASET[c]) == int:
                idx = RNASET[c]
                arr[0][i][idx][0] = 1
            else:
                continue

    return arr


def RNAonehot_generator(file, batch_size=64, num_classes=142, seq_length=312, shuffle=True):
    """Using python generator to generate batch of dataset
    """

    df = pd.read_csv(file, sep=',', header=None)
    indexs = [tmp for tmp in range(len(df[0]))]

    while True:
        # select sequences for the csv file
        if shuffle:
            np.random.shuffle(indexs)
        for i in range(0, len(indexs), batch_size):
            ids = indexs[i:i + batch_size]

            seq_np = np.zeros((len(ids), seq_length, 4, 1), dtype=np.float32)
            labels = np.zeros((len(ids), num_classes))

            for n in range(len(ids)):
                seq_np[n] = encoding_1hot(df[1][ids[n]])
                labels[n] = to_categorical(df[0][ids[n]], num_classes)

            yield (seq_np, labels)


class RNA_onehot(keras.utils.Sequence):
    """Generates data for Keras. https://github.com/afshinea/keras-data-generator/blob/master/my_classes.py
    """

    def __init__(self, file, ids, batch_size=64, dim=(312, 4), num_channels=1,
                 num_classes=143, shuffle=True):
        """Initialization
        """
        self.file = file  # csv file (column_0:index, column_1:sequence)
        self.ids = ids  # size of dataset
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = num_channels
        self.n_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.ceil(self.ids / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(self.ids)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples
        """  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        df = pd.read_csv(self.file, sep=',', header=None)
        labels = df[0]
        seqs = df[1]

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X[i, ] = encoding_1hot(seqs[ID], seq_length=self.dim[0])

            # Store class
            y[i] = labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# ================================== image =======================================
class RNA_img(keras.utils.Sequence):
    """Generates data for Keras. https://github.com/afshinea/keras-data-generator/blob/master/my_classes.py
    """

    def __init__(self, list_IDs, labels, batch_size=64, dim=(312, 312),
                 num_channels=1, num_classes=143, shuffle=True):
        """Initialization
        """
        self.dim = dim
        self.n_channels = num_channels
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples
        """  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp_mat = cv2.imread(f"inputdir/{ID}", self.n_channels - 1) / 255
            X[i,] = np.resize(temp_mat, (*self.dim, self.n_channels))

            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
