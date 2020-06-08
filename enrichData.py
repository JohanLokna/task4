import numpy as np
from numpy.random import uniform, binomial
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator

# Transformer
transformer = ImageDataGenerator()

for photoId in range(10000):

  # Load photo
  x = img_to_array(load_img('food/' + str(photoId).zfill(5) + '.jpg'))

  for marker in list(map(chr, range(97, 110))):

    transformArgs = {
      'theta': uniform(-30, 30),
      'tx': uniform(-30, 30),
      'ty': uniform(-30, 30),
      'brightness': uniform(0.1, 2.0),
      'flip_horizontal': binomial(1, 0.5),
      'flip_vertical': binomial(1, 0.5)
    }
    transformedX = transformer.apply_transform(x, transformArgs)
    save_img('food/' + str(photoId).zfill(5) + '_' + marker + '.jpg', transformedX)
