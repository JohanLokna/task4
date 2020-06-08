# GLOBAL DEFINES
T_G_WIDTH = 224
T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337
T_G_BATCHSIZE = 50
T_G_VAL_RATIO = 0.02
# Misc. Necessities
import sys
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
np.random.seed(T_G_SEED)

# TensorFlow Includes
import tensorflow as tf
tf.random.set_seed(T_G_SEED)

# tensorflow.keras Imports & Defines
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.layers as kl

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocessResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions as decodeResNet50

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocessMobileNetV2


from tensorflow.keras.applications.xception import preprocess_input as preprocessXception


T_G_PREPROCESS = preprocessXception

# Pandas for reading triplets
import pandas as pd

# Utiles
import os
from itertools import product

############# Model definition #############

# PRE:
# POST: Triplet loss function
def triplet_loss(y_true, y_pred):
  margin = K.constant(0.5)
  return K.mean(K.maximum(K.constant(0), y_pred[:,0] - y_pred[:,1] + margin))

# PRE:
# POST: Returns squared eucledian distance
def distanceSquared(x):
  x1, x2 = x
  return K.sum(K.square(x1 - x2), axis=1)

# PRE:
# POST: Costum accuracy
def accuracy(y_true, y_pred):
    tf.print(y_pred)
    return K.mean(y_pred[:,0] < y_pred[:,1])

# PRE:
# POST: Costum accuracy
def posDist(y_true, y_pred):
    return y_pred[:,0]

# PRE:
# POST: Costum accuracy
def negDist(y_true, y_pred):
    return y_pred[:,1]


# PRE:
# POST: 
def makeTriplet(baseModel, combineModel=None, name='tripletSiamese'):

  # Triplet framework, shared weights
  inputShape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
  inputAnchor = kl.Input(shape=inputShape, name='inputAnchor')
  inputPositive = kl.Input(shape=inputShape, name='inputPos')
  inputNegative = kl.Input(shape=inputShape, name='inputNeg')

  netAnchor = baseModel(inputAnchor)
  netPositive = baseModel(inputPositive)
  netNegative = baseModel(inputNegative)

  if combineModel != None:
    # Freely chosen modell for distance
    positiveDist = combineModel(kl.Subtract()([netAnchor, netPositive]))
    negativeDist = combineModel(kl.Subtract()([netAnchor, netNegative]))
  
  else:
    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positiveDist = kl.Lambda(distanceSquared, name='posDist')([netAnchor, netPositive])
    negativeDist = kl.Lambda(distanceSquared, name='negDist')([netAnchor, netNegative])

  # This lambda layer simply stacks outputs so both distances are available to the objective
  stackedDists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stackedDists')([positiveDist, negativeDist])

  model = Model(inputs=[inputAnchor, inputPositive, inputNegative], outputs=stackedDists, name=name)

  model.compile(optimizer='adam', loss=triplet_loss, metrics=[accuracy, posDist, negDist])

  return model


# PRE:
# POST: 
def createSimilarityModel(feature_model):
  img_a_in = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
  img_b_in = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
  img_a_feat = feature_model(img_a_in)
  img_b_feat = feature_model(img_b_in)
  combined_features = kl.concatenate([img_a_feat, img_b_feat], name = 'merge_features')
  combined_features = kl.Dense(16, activation = 'linear')(combined_features)
  combined_features = kl.BatchNormalization()(combined_features)
  combined_features = kl.Activation('relu')(combined_features)
  combined_features = kl.Dense(4, activation = 'linear')(combined_features)
  combined_features = kl.BatchNormalization()(combined_features)
  combined_features = kl.Activation('relu')(combined_features)
  combined_features = kl.Dense(1, activation = 'sigmoid')(combined_features)
  similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'SimilarityModel')

  similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
  
  return similarity_model


# PRE:
# POST: Returns a resnet base model
def createModelResnet():

  # Initialize a ResNet50_ImageNet Model
  pretrained = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                       weights='imagenet', pooling='avg',
                                                       input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))

  # freeze the body layers
  for layer in pretrained.layers:
      layer.trainable = False
  
  # Add top to make base model
  top = pretrained.output
  top = kl.Flatten()(top)
  top = kl.Dropout(0.25)(top)
  top = kl.Dense(1024, activation='relu')(top)

  baseModel = Model(inputs=pretrained.inputs, outputs=top, name='baseModel')

  return baseModel
  
def createMobileNetV2Top():
  baseModel = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
  # print(baseModel._name)
  # baseModel._name = 'baseModel'
  
  # Freeze MobileNetV2
  for layer in baseModel.layers:
    layer.trainable = False
  top = baseModel.output
  top = kl.GlobalAveragePooling2D()(top)
  top = kl.Flatten()(top)
  top = kl.Dense(256, activation='relu')(top)
  top = kl.Dense(128, activation='softmax')(top)
  return Model(inputs=baseModel.inputs, outputs=top, name='baseModel')

# PRE:
# POST: Returns a resnet base model
def createModelXception():

  pretrained = tf.keras.applications.xception.Xception(include_top=False,
                                                    weights='imagenet', pooling='avg',
                                                    input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
  pretrained._name = 'baseModel'
  # Add top to make base model
  top = pretrained.output
  # top = kl.MaxPooling2D(pool_size=(2,2), padding='same')(top)
  # top = kl.Flatten(name='flatten')(top)
  # top = kl.Dense(128, activation='tanh')(top)
  
  top = kl.Flatten()(top)
  top = kl.Dense(1024, activation='relu')(top)

  baseModel = Model(inputs=pretrained.inputs, outputs=top, name='baseModel')

  # freeze the body layers
  for layer in pretrained.layers:
      layer.trainable = False

  return baseModel




# PRE:
# POST: Returns a resnet base model
def createModelResnetFull():

  # Initialize a ResNet50_ImageNet Model
  baseModel = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
  baseModel.name = 'baseModel'

  # Freeze ResNet50
  for layer in baseModel.layers[:-2]:
    layer.trainable = False

  return baseModel


# PRE:
# POST: Returns a tensorflow.keras siamese model with triplet loss
def createModelShort():

  # Initialize base model
  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(filters=32, kernel_size=(7,7), activation='relu', padding='same', name='conv1'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp1'))
  baseModel.add(kl.Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same', name='conv2'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp2'))
  baseModel.add(kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv3'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp3'))
  baseModel.add(kl.Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same', name='conv4'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp4'))
  baseModel.add(kl.Conv2D(filters=28, kernel_size=(1,1), activation=None, padding='same', name='conv5'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp5'))
  baseModel.add(kl.Flatten(name='flatten'))

  return baseModel


# PRE:
# POST: Returns a tensorflow.keras siamese model with triplet loss
def createModelSmall():

  # Initialize base model
  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(filters=16, kernel_size=(7,7), activation='relu', padding='same', name='conv1'))
  baseModel.add(kl.MaxPooling2D(pool_size=(4,4), padding='same', name='mp1'))
  baseModel.add(kl.Conv2D(filters=2, kernel_size=(3,3), activation='relu', padding='same', name='conv2'))
  baseModel.add(kl.MaxPooling2D(pool_size=(5,5), padding='same', name='mp2'))
  baseModel.add(kl.Flatten(name='flatten'))
  baseModel.add(kl.Dense(12, activation='relu', name='embedded'))

  return baseModel


# PRE:
# POST: Returns a tensorflow.keras siamese model with triplet loss
def createModelSmallRegularized():

  # Initialize base model
  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(filters=16, kernel_size=(7,7), padding='same', name='conv1'))
  baseModel.add(kl.BatchNormalization())
  baseModel.add(kl.Activation('relu'))
  baseModel.add(kl.MaxPooling2D(pool_size=(4,4), padding='same', name='mp1'))
  baseModel.add(kl.Conv2D(filters=2, kernel_size=(3,3), padding='same', name='conv2'))
  baseModel.add(kl.BatchNormalization())
  baseModel.add(kl.Activation('relu'))
  baseModel.add(kl.MaxPooling2D(pool_size=(5,5), padding='same', name='mp2'))
  baseModel.add(kl.Flatten(name='flatten'))
  baseModel.add(kl.Dense(12, activation='relu', name='embedded'))
  baseModel.add(kl.Dropout(0.5, name='dropout'))

  return baseModel


# PRE:
# POST: Returns a tensorflow.keras siamese model with triplet loss
def createModelNormalized():

  # Initialize base model
  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(filters=16, kernel_size=(5,5), padding='same', name='conv1'))
  baseModel.add(kl.BatchNormalization())
  baseModel.add(kl.Activation('relu'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp1'))
  baseModel.add(kl.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', name='conv2'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp2'))
  baseModel.add(kl.Conv2D(filters=64, kernel_size=(1,1), padding='same', name='conv3'))
  baseModel.add(kl.BatchNormalization())
  baseModel.add(kl.Activation('relu'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp3'))
  baseModel.add(kl.Conv2D(filters=8, kernel_size=(1,1), padding='same', name='conv4'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2,2), padding='same', name='mp4'))
  baseModel.add(kl.Flatten(name='flatten'))
  baseModel.add(kl.Dropout(0.5, name='dropout'))

  return baseModel


# PRE:
# POST:
def createModelLecture():

  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Dropout(0.25))
  baseModel.add(kl.Flatten())
  baseModel.add(kl.Dense(128, activation='relu'))
  baseModel.add(kl.Dropout(0.5))
  baseModel.add(kl.Dense(128, activation='softmax'))

  return baseModel


# PRE:
# POST:
def createModelMini():

  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Dropout(0.25))
  baseModel.add(kl.Flatten())
  baseModel.add(kl.Activation('tanh'))

  return baseModel


# PRE:
# POST:
def createModelKaggle():
  img_in = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS), name = 'FeatureNet_ImageInput')
  n_layer = img_in
  for i in range(2):
      n_layer = kl.Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
      n_layer = kl.BatchNormalization()(n_layer)
      n_layer = kl.Activation('relu')(n_layer)
      n_layer = kl.Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
      n_layer = kl.BatchNormalization()(n_layer)
      n_layer = kl.Activation('relu')(n_layer)
      n_layer = kl.MaxPool2D((2,2))(n_layer)
  n_layer = kl.Flatten()(n_layer)
  n_layer = kl.Dense(32, activation = 'linear')(n_layer)
  n_layer = kl.Dropout(0.5)(n_layer)
  n_layer = kl.BatchNormalization()(n_layer)
  n_layer = kl.Activation('relu')(n_layer)
  baseModel = Model(inputs = [img_in], outputs = [n_layer], name = 'baseModel')
  
  return baseModel


# PRE:
# POST:
def createModelTrueLecture():
  baseModel = tf.keras.Sequential(name='baseModel')
  baseModel.add(kl.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  baseModel.add(kl.MaxPooling2D(pool_size=(2, 2)))
  baseModel.add(kl.Dropout(0.25))
  baseModel.add(kl.Flatten())
  baseModel.add(kl.Dense(128, activation='softmax'))
  
  return baseModel


# PRE:
# POST:
def createPostprocess(emb_size):
  combineModel = tf.keras.Sequential(name='combineModel')
  combineModel.add(kl.Dense(emb_size, activation='relu'))
  combineModel.add(kl.Dropout(0.5))
  combineModel.add(kl.Dense(1, activation='softmax'))

  return combineModel


############# Generator definition #############


class TipletGenerator(tf.keras.utils.Sequence) :

  def __init__(self, tiplets, batchSize, preprocess=lambda x: x) :
    self.triplets = tiplets
    self.batchSize = batchSize
    self.preprocess = preprocess


  def __len__(self) :
    return (np.ceil(len(self.triplets) / float(self.batchSize))).astype(np.int)


  def __getitem__(self, idx) :
    batch = self.triplets[idx * self.batchSize : min((idx+1) * self.batchSize, len(self.triplets))]
    
    def getChannelInput(idx):
      
      inputPaths = ['./food/' + triplet[idx] + '.jpg' for triplet in batch]
      
      inputs = list(map(
        lambda x: load_img(x, target_size=(T_G_WIDTH, T_G_HEIGHT)),
        inputPaths
      ))
      
      inputs = np.array(list(map(
        lambda x: img_to_array(x),
        inputs
      )))
      
      inputs = self.preprocess(inputs)

      return inputs

    return [getChannelInput(idx) for idx in range(3)], np.zeros(len(batch))


class SimilarityGenerator(tf.keras.utils.Sequence) :

  def __init__(self, triplets, batchSize, preprocess=lambda x: x, shuffle=False) :
    self.pairs = [[triplet[0], triplet[i]] for triplet, i in product(triplets, [1, 2])]
    self.y = np.array([i for triplet, i in product(triplets, [1, 0])])
    self.batchSize = batchSize
    self.preprocess = preprocess

    if shuffle:

      indices = np.arange(self.y.shape[0])
      np.random.shuffle(indices)
      
      self.pairs = [self.pairs[i] for i in indices]
      self.y = self.y[indices]


  def __len__(self) :
    return (np.ceil(len(self.pairs) / float(self.batchSize))).astype(np.int)


  def __getitem__(self, idx) :

    batch = self.pairs[idx * self.batchSize : (idx+1) * self.batchSize]
    
    def getChannelInput(idx):
      
      inputPaths = ['./food/' + pair[idx] + '.jpg' for pair in batch]
      
      inputs = list(map(
        lambda x: load_img(x, target_size=(T_G_WIDTH, T_G_HEIGHT)),
        inputPaths
      ))
      
      inputs = np.array(list(map(
        lambda x: img_to_array(x),
        inputs
      )))
      
      inputs = self.preprocess(inputs)

      return inputs

    return [getChannelInput(idx) for idx in range(2)], self.y[idx * self.batchSize : (idx+1) * self.batchSize]


############# Utils #############


def getTriplets(filename, subfixes=['']):
  with open(filename) as f:
    return [[idx + subfix for idx in line.split()] for subfix, line in product(subfixes, f)]


def loadModel(filename):
  model = tf.keras.models.load_model(filename, custom_objects={'triplet_loss': triplet_loss, \
                                                            'distanceSquared': distanceSquared, \
                                                            'accuracy': accuracy, \
                                                            'posDist': posDist, 'negDist': negDist})
  return model


def printModel(model):
  print(model.summary())
  print(model.get_layer('baseModel').summary())
    

def train_val_split(triplets, size):
  val_triplets = triplets[:int(len(triplets) * size)]
  mask = np.any(np.isin(triplets, val_triplets), axis=1) != True
  train_triplets = np.array(triplets)[mask]
  return train_triplets, val_triplets


############# Training #############


def train(model, tripletsTrain, tripletsVal, nEpochs, batchSize, outdir, preprocess=lambda x: x, transfer=False):

  trainGen = TipletGenerator(tripletsTrain, batchSize, preprocess)
  valGen = TipletGenerator(tripletsVal, batchSize, preprocess)
  if transfer:
    print('transfer')
    baseModel = model.get_layer('baseModel')
    # transfer learning: after training the classifier layers on top of the convs,
    # fine-tune the conv layers
    for layer in baseModel.layers[:-2]:
      layer.trainable = True
    for layer in baseModel.layers[-2:]:
      layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=triplet_loss, metrics=[accuracy, posDist, negDist])
    printModel(model)
  
  print("training for ", len(tripletsTrain), "triplets")
  for i in range(nEpochs):
    model.fit(trainGen,
                        steps_per_epoch = len(trainGen),
                        epochs = 1,
                        verbose = 1,
                        validation_data = valGen,
                        validation_steps = len(valGen))
    model.save(outdir + '/model' +  str(len(os.listdir(outdir))))


def main(model, nEpochs, outdir, preprocess=lambda x: x, sufixes=[''], transfer=False):
  
  printModel(model)

  if not os.path.exists(outdir):
    os.makedirs(outdir)
 
  triplets = getTriplets('train_triplets.txt', sufixes)
  train_triplets, val_triplets = train_val_split(triplets, T_G_VAL_RATIO)

  train(model, train_triplets, val_triplets, nEpochs, T_G_BATCHSIZE, outdir, preprocess, transfer)


############# Post training #############


def pred(model, filename, distFilename):

  triplets = getTriplets('test_triplets.txt')
  testGen = TipletGenerator(triplets, 100)
  with open(filename, 'w+') as f:
    with open(distFilename, 'w+') as distF:
      for i in range(len(testGen)):
        X, y = testGen[i]
        y_pred = model.predict(X)
        for j in range(y_pred.shape[0]):
          f.write(str(int(y_pred[j,0] < y_pred[j,1])) + '\n')
          distF.write('{} {}\n'.format(y_pred[j,0], y_pred[j,1]))
        print('Completed: {}/{}'.format(i + 1, len(testGen)))


def validate(model, preprocess):
  triplets = getTriplets('train_triplets.txt')
  train_triplets, val_triplets = train_val_split(triplets, T_G_VAL_RATIO)
  batchSize = T_G_BATCHSIZE
  valGen = TipletGenerator(val_triplets, batchSize, preprocess)

  def getValAccuracy():
        res = 0.0
        n = 0
        for i in range(len(valGen)):
          X, y = valGen[i]
          y_pred = model.predict(X)
          res += np.sum(y_pred[:,0] < y_pred[:,1])
          n += y.shape[0]
          print(y_pred)
          print('Temp ' + str(i) + '/' + str(len(valGen)) + ': ', res / n)
        return res / n

  print('\n\nValidation accuracy for model: ', getValAccuracy(), '\n\n')


############# Calling #############

if __name__ == '__main__':
  model = makeTriplet(baseModel=createModelXception(), combineModel=None, name='mobilenetv2')
  # model = loadModel('mobilenetv2/model5')
  main(model, 'xception', T_G_PREPROCESS, transfer=True)
  # printModel(model)
  # pred(model, T_G_PREPROCESS, 'resultsmobilenetv2_3epochs.txt')
  # validate(model, T_G_PREPROCESS)
