import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Utilize GPU instead of CPU
physical_devices = tf.config.experimental.list_logical_devices('GPU')
print('Available GPUSs: ', len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initialize sequential model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='relu')
])

# Sumarize model
#model.summary()

# Prepare model for training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=true, verbose=2)