import configparser
import sys
import time

from PIL import Image, ImageTk
import cozmo
import cv2
from keras import backend as K
from keras.models import model_from_json
import numpy as np


def init(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.enable_auto_exposure()

    try:
        config = configparser.ConfigParser()
        config.read('conf.ini')

        img_cols = config.getint('DEFAULT', 'img_cols')
        img_rows = config.getint('DEFAULT', 'img_rows')
    except FileNotFoundError:
        print('not found "conf.ini"')
        sys.exit()

    # Initializing prev_label
    prev_label = [None, None, None, None, None, None, None, None, None, None,
                  None, None, None, None, None, None, None, None, None, None,
                  None, None, None, None, None, None, None, None, None, None]

    return img_cols, img_rows, prev_label


# モデルと重みのpathを引数として受け取り、モデルを返す
def set_model(model_path, weight_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weight_path)
    model.summary()

    return model


# 画像を学習時と同じ形にする
def get_binary_image(image):
    threshold = 60
    image_bool = image > threshold
    binary_image = image_bool * 255
    binary_image = binary_image[80:160, 0:320]

    return binary_image


# テスト用画像を実際に走行しながら取得する関数
def image_saver(image, i):
    image_name = "test_image" + str(i) + ".png"

    cv2.imwrite(image_name, image)


def run2(img_cols, img_rows, prev_label, model, robot: cozmo.robot.Robot):
    """
    まっすぐ, 右の緩急, 左の緩急, バックの６パターン
    TODO: 右に行くときの認識が甘く、まっすぐと誤認識している。要改善。
    TODO: 学習画像の枚数が不揃いなので、揃える。
    TODO: Cozmoの画像上に進む方向をオーバーレイで表示する。
    TODO: 何もないところをバックと認識することがある。
    """

    for _ in range(100000):
        start = time.time()

        new_image = robot.world.wait_for(cozmo.world.EvtNewCameraImage)
        new_image.image.raw_image.save('image.png')
        image = cv2.imread('image.png', cv2.IMREAD_COLOR)
        image = get_binary_image(image)

        if K.image_data_format() == 'channels_first':
            image = image.reshape(1, 3, img_rows, img_cols)
        else:
            image = image.reshape(1, img_rows, img_cols, 3)

        pred = model.predict(image, batch_size=1, verbose=0)
        pred_label = np.argmax(pred)

        prev_label = np.roll(prev_label, 1)
        prev_label[0] = pred_label

        print(prev_label, end=", ")

        # 緩やかに右に動く
        if pred_label == 1:
            print('rightゆっくり→')
            robot.drive_wheel_motors(25, 10)
        # 緩やかに左に動く
        elif pred_label == 0:
            print('←leftゆっくり')
            robot.drive_wheel_motors(10, 25)
        # 急に左に動く
        elif pred_label == 4:
            print('←leftきゅうカーブ')
            robot.drive_wheel_motors(5, 25)
        # 急に右に動く
        elif pred_label == 5:
            print('rightきゅうカーブ→')
            robot.drive_wheel_motors(25, 5)
        # まっすぐ進む
        elif pred_label == 3:
            print('まっすぐ')
            robot.drive_wheel_motors(25, 25)
        # 戻ってやり直し
        elif pred_label == 2:
            print("------------------")
            robot.drive_wheel_motors(0, 0)
            time.sleep(3)
            robot.drive_wheel_motors(-10, -10)
            time.sleep(5)
            robot.drive_wheel_motors(0, 0)
            time.sleep(3)

            count0 = count1 = 0
            for i in range(30):
                if prev_label[i] == 0 or prev_label[i] == 4:
                    count0 += 1
                elif prev_label[i] == 1 or prev_label[i] == 5:
                    count1 += 1

            print(prev_label)
            print("Loop1 → count0: " + str(count0) + ", " + "count1: " + str(count1))
            if count1 < count0:
                print("左に行きます！")
                robot.drive_wheel_motors(5, 25)
            elif count0 < count1:
                print("右に行きます！")
                robot.drive_wheel_motors(25, 5)
            else:
                count0 = count1 = 0
                for j in range(29):
                    if prev_label[j] == 0 or prev_label[j] == 4:
                        count0 += 1
                    elif prev_label[j] == 1 or prev_label[j] == 5:
                        count1 += 1

                print("Loop2 → count0: " + str(count0) + ", " + "count1: " + str(count1))
                if count1 < count0:
                    print("左に行きます！")
                    robot.drive_wheel_motors(5, 25)
                elif count0 < count1:
                    print("右に行きます！")
                    robot.drive_wheel_motors(25, 5)
                else:
                    print("どれでもない")

            time.sleep(2)

        print(time.time() - start)


def cozmo_program(robot: cozmo.robot.Robot):
    img_cols, img_rows, prev_label = init(robot)
    model = set_model('../models/and9.json',
                      '../models/and9.h5')
    run2(img_cols, img_rows, prev_label, model, robot)


if __name__ == '__main__':
    cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=False)
