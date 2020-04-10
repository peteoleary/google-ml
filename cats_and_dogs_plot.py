import matplotlib.pyplot as plt

class Plot:

    def plot(self, history):
        # Retrieve a list of accuracy results on training and validation data
        # sets for each training epoch
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        # Retrieve a list of list results on training and validation data
        # sets for each training epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Get number of epochs
        epochs = range(len(acc))

        # Plot training and validation accuracy per epoch
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')

        plt.figure()

        # Plot training and validation loss per epoch
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')

        plt.show()