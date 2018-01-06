import time

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
import keras.callbacks
from keras.applications.imagenet_utils import get_file
from keras.optimizers import adam

from keras_resnet.models import ResNet50

from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import WEIGHTS_PATH_NO_TOP_50
from keras_retinanet import losses
from keras_retinanet.models.retinanet import retinanet_bbox

batch_size = 1
epochs = 20
steps = 500

def create_generator():
    # horizontal flip for preprocessing augmentation
    train_generator = ImageDataGenerator(horizontal_flip=True)
    # wrap in CSVGenerators
    csv_train_generator = CSVGenerator('./train.csv', './classes.csv', 
                                       train_generator, batch_size=batch_size)
    # no flip for val
    val_generator = ImageDataGenerator()
    csv_val_generator = CSVGenerator('./train.csv', './classes.csv', 
                                       val_generator, batch_size=batch_size)
    return csv_train_generator, csv_val_generator

def create_model(num_classes):
    input_shape = Input((None, None, 3))
    
    # first create the backbone ResNet50 model
    weights_path = get_file('ResNet-50-model.keras.h5', WEIGHTS_PATH_NO_TOP_50, 
                            cache_subdir='models', md5_hash='3e9f4e4f77bbe2c9bec13b53ee1c2319')
    backbone_model = ResNet50(input_shape, include_top=False, freeze_bn=True)
    # add feature pyramid network and subnets for box regression and object classification
    model = retinanet_bbox(inputs=input_shape, num_classes=num_classes, backbone=backbone_model)
    model.load_weights(weights_path, by_name=True)
    # compile model
    model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=adam(lr=1e-5, clipnorm=0.001)
    )
    
    return model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def create_callbacks():
    checkpoint = keras.callbacks.ModelCheckpoint('resnet50_csv_best.h5', verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    loss_history = LossHistory()
    
    return [checkpoint, lr_scheduler, loss_history]

train_generator, val_generator = create_generator()
model = create_model(train_generator.num_classes())
print(model.summary())

callbacks = create_callbacks()

start_time = time.time()

history = model.fit_generator(generator=train_generator, 
                              steps_per_epoch=steps, epochs=epochs, 
                              validation_data=val_generator, validation_steps=steps, 
                              callbacks=callbacks, verbose=1)

end_time = time.time()

print('Training took {} seconds'.format(end_time - start_time))

# save the history as a json for chart
import pickle

with open('history.pkl' , 'wb') as history_file:
    pickle.dump(history.history, history_file)

with open('history.pkl', 'rb') as history_file:
    data = pickle.load(history_file)
    print(data)