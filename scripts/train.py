import configparser
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import History, EarlyStopping

try:
    config = configparser.ConfigParser()
    config.read('conf.ini')

    img_cols = config.getint('DEFAULT', 'img_cols')
    img_rows = config.getint('DEFAULT', 'img_rows')
    batch_size = config.getint('DEFAULT', 'batch_size')
    num_classes = config.getint('DEFAULT', 'num_classes')
    epochs = config.getint('DEFAULT', 'epochs')
except FileNotFoundError:
    print('not found "conf.ini"')
    sys.exit()

train = np.load('../data/train_data11.npz')
x_train = train['x']
y_train = train['y']

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# 入力データ
x_train = x_train.astype('float32')
x_train /= 255

# ラベルはone-hot encodingを施す
y_train = y_train.astype('int32')
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)

# コールバックを定義
early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')

# 学習モデルを定義
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 学習モデルをコンパイル
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 学習を実行
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    validation_split=0.1,
                    callbacks=[early_stopping])

number_of_epochs_it_ran = len(history.history['loss'])

print('acc')
print(history.history['acc'])
print('val_acc')
print(history.history['val_acc'])

plt.plot(range(1, number_of_epochs_it_ran+1), history.history['acc'], label="training")
plt.plot(range(1, number_of_epochs_it_ran+1), history.history['val_acc'], label="validation")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# モデルと重みをファイルに書き出す
open('../models/temp2.json', "w").write(model.to_json())
model.save_weights('../models/temp2.h5')
