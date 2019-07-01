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


def DeepRfam_deep(seq_length=312, num_c=4, num_filters=256,
                  filter_sizes=[24, 36, 48, 60, 72, 84, 96, 108],
                  dropout_rate=0.5, num_classes=143, num_hidden=512):
    # initialization
    in_shape = (seq_length, num_c, 1)

    input_shape = Input(shape=in_shape)
    #     input_shape = Input(shape = (312, 4, 1))

    pooled_outputs = []
    # input_shape = Conv2D(num_filters,
    #                 (2, 4),
    #                 padding='valid',
    #                 activation='relu')(input_shape)
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters,
                      (filter_sizes[i], num_c),
                      padding='valid',
                      activation='relu')(input_shape)
        pool = MaxPooling2D((2, 1), strides=2,
                            padding='valid')(conv)

        conv = Conv2D(num_filters*2,
                      (filter_sizes[i], 1),
                      padding='valid',
                      activation='relu')(pool)
        pool = MaxPooling2D((seq_length - filter_sizes[i]*2 + 1, 1),
                            padding='valid')(conv)

        pooled_outputs.append(pool)

    merge = concatenate(pooled_outputs)

    x = Flatten()(merge)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


# use local max pooling
def DeepRfam_lenet(seq_length=312, num_c=4, num_filters=32,
                   filter_sizes=[3, ],
                   dropout_rate=0.5, num_classes=143, num_hidden=512):
    # initialization
    in_shape = (seq_length, num_c, 1)

    input_shape = Input(shape=in_shape)
    # input_shape = Input(shape = (312, 4, 1))

    conv = Conv2D(num_filters,
                  (filter_sizes[0], num_c),
                  padding='valid',
                  activation='relu')(input_shape)
    pool = MaxPooling2D((2, 1), strides=1,
                        padding='valid')(conv)
    conv = Conv2D(num_filters * 2,
                  (filter_sizes[0], 1),
                  padding='valid',
                  activation='relu')(pool)
    pool = MaxPooling2D((2, 1), strides=1,
                        padding='valid')(conv)

    x = Flatten()(pool)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model

    
# =============== model using to train image dataset ======================
def ImgFam(seq_length=312, num_filters=32, num_channels=1,
           filter_sizes=[72, 84, 96, 108],
           dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters,
                      (filter_sizes[i], filter_sizes[i]),
                      padding='valid',
                      activation='relu')(input_shape)
        pool = MaxPooling2D((seq_length - filter_sizes[i] + 1, seq_length - filter_sizes[i] + 1),
                            padding='valid')(conv)
        pooled_outputs.append(pool)

    merge = concatenate(pooled_outputs)

    x = Flatten()(merge)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L4Fam(seq_length=312, num_filters=16, num_channels=1,
          filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L4BNFam(seq_length=312, num_filters=16, num_channels=1,
            filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid')(input_shape)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D((2, 2), padding='valid')(act1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid')(pool1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D((2, 2), padding='valid')(act2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L5CFam(seq_length=312, num_filters=16, num_channels=1,
           filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden * 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L5CFam_ave(seq_length=312, num_filters=16, num_channels=1,
           filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = AveragePooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = AveragePooling2D((2, 2), padding='valid')(conv2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden * 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L5CFam_temp(seq_length=312, num_filters=16, num_channels=1,
           filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = AveragePooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = AveragePooling2D((2, 2), padding='valid')(conv2)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden * 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L5CFam_dilation(seq_length=312, num_filters=16, num_channels=1,
           filter_sizes=[3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu',
                   dilation_rate=5)(input_shape)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu',
                   dilation_rate=5)(conv1)

    x = Dropout(dropout_rate)(conv2)
    x = Flatten()(x)
    x = Dense(num_hidden * 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L5Fam(seq_length=312, num_filters=16, num_channels=1,
          filter_sizes=[3, 3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv1)

    conv2 = Conv2D(num_filters * 2,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(pool1)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv2)

    conv3 = Conv2D(num_filters * 4,
                   (filter_sizes[2], filter_sizes[2]),
                   padding='valid',
                   activation='relu')(pool2)

    pool3 = MaxPooling2D((2, 2), padding='valid')(conv3)

    x = Dropout(dropout_rate)(pool3)
    x = Flatten()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L6Fam(seq_length=312, num_filters=16, num_channels=1,
          filter_sizes=[3, 3, 3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    conv2 = Conv2D(num_filters,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(conv1)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv2)

    conv3 = Conv2D(num_filters * 2,
                   (filter_sizes[2], filter_sizes[2]),
                   padding='valid',
                   activation='relu')(pool1)

    conv4 = Conv2D(num_filters * 2,
                   (filter_sizes[3], filter_sizes[3]),
                   padding='valid',
                   activation='relu')(conv3)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv4)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def L7CFam(seq_length=312, num_filters=16, num_channels=1,
           filter_sizes=[3, 3, 3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    conv1 = Conv2D(num_filters,
                   (filter_sizes[0], filter_sizes[0]),
                   padding='valid',
                   activation='relu')(input_shape)

    conv2 = Conv2D(num_filters,
                   (filter_sizes[1], filter_sizes[1]),
                   padding='valid',
                   activation='relu')(conv1)

    pool1 = MaxPooling2D((2, 2), padding='valid')(conv2)

    conv3 = Conv2D(num_filters * 2,
                   (filter_sizes[2], filter_sizes[2]),
                   padding='valid',
                   activation='relu')(pool1)

    conv4 = Conv2D(num_filters * 2,
                   (filter_sizes[3], filter_sizes[3]),
                   padding='valid',
                   activation='relu')(conv3)

    pool2 = MaxPooling2D((2, 2), padding='valid')(conv4)

    x = Dropout(dropout_rate)(pool2)
    x = Flatten()(x)
    x = Dense(num_hidden * 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def Github_scnn(seq_length=312, num_filters=16, num_channels=1,
                filter_sizes=[3, 3, 3, 3], dropout_rate=0.5, num_classes=42, num_hidden=64):
    """https://github.com/zhulanyun/Chinese-character-recognition-using-a-shallow-CNN/blob/master/cnnmodel.py
    """
    # initialization
    in_shape = (seq_length, seq_length, num_channels)

    input_shape = Input(shape=in_shape)

    X = Conv2D(num_filters, (filter_sizes[0], filter_sizes[0]), strides=(1, 1), activation='relu', padding='same')(
        input_shape)
    X = Conv2D(num_filters, (filter_sizes[0], filter_sizes[0]), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=-1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one',
                           mode=0)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = Conv2D(num_filters * 2, (filter_sizes[1], filter_sizes[1]), strides=(1, 1), activation='relu', padding='same')(
        X)
    X = Conv2D(num_filters * 2, (filter_sizes[1], filter_sizes[1]), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=-1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one',
                           mode=0)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = Conv2D(num_filters * 4, (filter_sizes[2], filter_sizes[2]), strides=(1, 1), activation='relu', padding='same')(
        X)
    X = Conv2D(num_filters * 4, (filter_sizes[2], filter_sizes[2]), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=-1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one',
                           mode=0)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(dropout_rate)(X)

    X = Flatten()(X)

    X = Dense(num_hidden * 2)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    X = Dropout(dropout_rate)(X)
    X = Dense(num_hidden)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    X = Dropout(dropout_rate)(X)
    Y = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=input_shape, outputs=Y)

    return model
