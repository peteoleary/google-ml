from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cats_and_dogs_files import check_flag, train_dir, validation_dir

# All images will be rescaled by 1./255

class Training:
    
    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs

    def train(self):
        if check_flag('augment'):
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,)
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)

        val_datagen = ImageDataGenerator(rescale=1./255)

        training_batch_size = 20

        # Flow training images in batches of 20 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
                train_dir,  # This is the source directory for training images
                target_size=(150, 150),  # All images will be resized to 150x150
                batch_size=training_batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

        validation_batch_size = 20

        # Flow validation images in batches of 20 using val_datagen generator
        validation_generator = val_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=validation_batch_size,
                class_mode='binary')

        training_steps = 100
        validation_steps = 50

        print("training the model total_training_size=%d total_validation_size=%d..." % (training_batch_size * training_steps, validation_batch_size * validation_steps), flush=True)

        history = self.model.fit_generator(
              train_generator,
              steps_per_epoch=training_steps,  # 2000 images = batch_size * steps
              epochs=self.epochs,
              validation_data=validation_generator,
              validation_steps=validation_steps,  # 1000 images = batch_size * steps
              verbose=1)

        print("finished training...", flush=True)

        return history