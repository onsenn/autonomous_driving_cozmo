import sys
import configparser
import datetime

from keras .preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np


def shuffle_samples(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x_result, y_result = zip(*zipped)
    return np.asarray(x_result), np.asarray(y_result)


def make_dataset():
    # 画像サイズ
    try:
        config = configparser.ConfigParser()
        config.read('conf.ini')

        img_cols = config.getint('DEFAULT', 'img_cols')
        img_rows = config.getint('DEFAULT', 'img_rows')
        img_size = (img_rows, img_cols)
    except FileNotFoundError:
        print('not found "conf.ini"')
        sys.exit()

    x_train = []
    y_train = []
    # 学習データ
    for i in range(1, 165):
        img = load_img('../data/train2/left_obtuse/s' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(3)

        print(i)

    for i in range(1, 120):
        img = load_img('../data/train2/right_obtuse/s' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(1)

        print(i)

    for i in range(1, 125):
        img = load_img('../data/train2/left_acute/b' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(4)

        print(i)

    for i in range(1, 81):
        img = load_img('../data/train2/right_acute/b' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(5)

        print(i)

    for i in range(1, 27):
        img = load_img('../data/train2/stop/b' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(2)

        print(i)

    for i in range(1, 89):
        img = load_img('../data/train2/straight/s' + str(i) + '.png', grayscale=False, target_size=img_size)
        img = np.asarray(img)
        x_train.append(img)
        y_train.append(0)

        print(i)

    x_train, y_train = shuffle_samples(x_train, y_train)

    np.savez('../data/train_data11.npz', x=x_train, y=y_train)


if __name__ == '__main__':
    make_dataset()
