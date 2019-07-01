import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
import time


model = model_from_json(open("../models/and9.json").read())
model.load_weights("../models/and9.h5")
model.summary()

img_rows, img_cols = 80, 320
image = cv2.imread('../data/train2/straight/s69.png')
plt.imshow(image)
print('image_shape: ' + str(image.shape))
if K.image_data_format() == 'channels_first':
    image = image.reshape(1, 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    image = image.reshape(1, img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

middle_model = Model(inputs=model.input, output=model.get_layer('dense_1').output)
middle_output = middle_model.predict(image)
print('middle_output_shape: ' + str(middle_output.shape))

#middle_output = middle_output.transpose(1, 0)
# plt.axis()
# plt.figure(facecolor="b", edgecolor="b", linewidth=2)
plt.tick_params(labelcolor='w', color='w')
plt.box(False)
plt.imshow(middle_output)
plt.savefig("./data/dense_1/98.png", bbox_inches='tight', transparent=True)
plt.imsave("./data/dense_1/99.png", middle_output)
print(middle_output)
# middle_output = middle_output.transpose(3, 0, 1, 2)
# nb_filter, nb_channel, nb_row, nb_col = middle_output.shape
#
# print(middle_output.shape)
#
# plt.figure()
# i = 0
# for i in range(nb_filter):
#     im = middle_output[i, 0]
#     #scaler = MinMaxScaler(feature_range=(0, 255))
#     #im = scaler.fit_transform(im)
#
#     plt.imshow(im, cmap='gray')
#     plt.imsave('./data/max_pooling2d_1/' + str(i) + '.png', im)
#
#     i = i + 1
#
# plt.show()