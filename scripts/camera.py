import cozmo
import random
import time
import cv2


def init(robot):
    robot.camera.image_stream_enabled = True
    robot.camera.enable_auto_exposure()


def binary_images(image):
    threshold = 60
    image_bool = image > threshold
    binary_image = image_bool * 255
    binary_image = binary_image[80:160, 0:320]

    return binary_image


def get_images(count_image, image_path, robot):
    time.sleep(1)
    for i in range(count_image):
        try:
            new_image = robot.world.wait_for(cozmo.world.EvtNewCameraImage)

            num = random.randrange(10000)
            image_name1 = image_path[1] + str(num) + '.png'
            new_image.image.raw_image.save(image_name1)

            image = cv2.imread(image_name1)
            image = binary_images(image)
            image_name2 = image_path[0] + str(num) + '.png'
            cv2.imwrite(image_name2, image)

            print(str(i + 1) + '枚目')

            time.sleep(2)

        except AttributeError:
            print("AttributeError")


def cozmo_program(robot: cozmo.robot.Robot):
    init(robot)

    count_image = 1

    image_path1 = ['../data/train2/straight/', '../data/train2/original_image/straight']
    image_path2 = ['../data/train2/left_obtuse/', '../data/train2/original_image/left_obtuse']
    image_path3 = ['../data/train2/left_acute/', '../data/train2/original_image/left_acute']
    image_path4 = ['../data/train2/right_obtuse/', '../data/train2/original_image/right_obtuse']
    image_path5 = ['../data/train2/right_acute/', '../data/train2/original_image/right_acute']
    image_path6 = ['../data/train2/stop/', '../data/train2/original_image/stop']

    get_images(count_image, image_path6, robot)


if __name__ == '__main__':
    cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
