import keras
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate, BatchNormalization, \
    Activation, AveragePooling2D


# ================= model using to train one-hot dataset ==================
def DeepRfam(seq_length=312, num_c=4, num_filters=256,
             filter_sizes=[24, 36, 48, 60, 72, 84, 96, 108],
             dropout_rate=0.5, num_classes=143, num_hidden=512):
    # initialization
    in_shape = (seq_length, num_c, 1)

    input_shape = Input(shape=in_shape)
    #     input_shape = Input(shape = (312, 4, 1))

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters,
                      (filter_sizes[i], num_c),
                      padding='valid',
                      activation='relu')(input_shape)
        pool = MaxPooling2D((seq_length - filter_sizes[i] + 1, 1),
                            padding='valid')(conv)
        pooled_outputs.append(pool)

    merge = concatenate(pooled_outputs)

    x = Flatten()(merge)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model

    
# =============== model using to train image dataset ======================
def L5CFam(seq_length=312, num_filters=[32, 64], num_channels=1,
           filter_sizes=[2, 2], dropout_rate=0.5, num_classes=42, num_hidden=[128, 64]):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters[0],
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters[1],
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden[0], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden[1], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model
