import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Install Keras Tuner if not already installed
!pip install keras-tuner
import keras_tuner as kt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess: Reshape to (28, 28, 1) and scale pixels
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def build_model(hp):
    model = models.Sequential()

    # --- DYNAMIC CONVOLUTION STAGE ---
    # Choose between 1 to 3 Convolutional Blocks
    for i in range(hp.Int("conv_layers", 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Choice(f"filters_{i}", [32, 64, 128]),
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=(28, 28, 1) if i == 0 else None
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    # --- DYNAMIC DENSE STAGE ---
    # Choose between 1 to 2 Dense Layers
    for j in range(hp.Int("dense_layers", 1, 2)):
        model.add(layers.Dense(
            units=hp.Choice(f"units_{j}", [256, 512]),
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float(f"dropout_{j}", 0.3, 0.5, step=0.1)))

    model.add(layers.Dense(10, activation='softmax'))

    # Tune the learning rate
    hp_lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

   tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='tuning_results',
    project_name='mnist_optimized'
)

# Early stopping to prevent overfitting during tuning
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
Best Conv Layers: {best_hps.get('conv_layers')}
Best Dense Layers: {best_hps.get('dense_layers')}
Best Learning Rate: {best_hps.get('learning_rate')}
""") 
