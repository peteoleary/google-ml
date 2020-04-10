import cats_and_dogs_files
from cats_and_dogs_files import check_flag

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


from tensorflow.keras import layers

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

import os

from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])

from cats_and_dogs_train import Training

trainer = Training(model, 3)

history = trainer.train()

from cats_and_dogs_plot import Plot

plotter = Plot()

plotter.plot(history)

if check_flag('fine_tune'):
    from tensorflow.keras.optimizers import SGD

    unfreeze = False

    # Unfreeze all models after "mixed6"
    for layer in pre_trained_model.layers:
      if unfreeze:
        layer.trainable = True
      if layer.name == 'mixed6':
        unfreeze = True

    # As an optimizer, here we will use SGD 
    # with a very low learning rate (0.00001)
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(
                      lr=0.00001, 
                      momentum=0.9),
                  metrics=['acc'])
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
    
    plotter.plot(history)