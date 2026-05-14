
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and Reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build the Champion Model (Based on Trial #8)
model = models.Sequential([
    # Convolutional Base
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Dense Head
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(10, activation='softmax')
])

# 3. Compile and Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00018453),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Starting final training run...")
history = model.fit(
    x_train, y_train, 
    epochs=20, 
    validation_data=(x_test, y_test),
    batch_size=40,
    verbose=1
)

# 4. Save for GitHub
model.save('mnist_champion_model.h5')
