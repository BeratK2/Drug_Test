# Background: An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial.
# The trial had 2100 participants. Half were younger than 65, half were older.
# Around 95% of patients 65 or older experienced side effects.
# Around 95% of patients under 65 has no side effects.

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Initialize train labels and samples
train_labels = []
train_samples = []
test_labels = []
test_samples = []


# Create sample dataset
# Outliers (5% of younger with side effects and 5% of older without side effects)
for i in range(50):
    # 5% of younger patients with side effects
    random_younger = randint(13,64) # Generate to random age
    train_samples.append(random_younger) # Append age to sample dataset
    train_labels.append(1) # Append 1 (true for side effects) for the associated label

    # 5% of older individuals who did not experience side effects
    random_older = randint(65,100) # Generate random age
    train_samples.append(random_older) # Append age to sample dataset
    train_labels.append(0) # Append 0 (false for side effects) for the associated label

# Normal samples (95% younger without side effects and 95% older with side effects)
for i in range(1000):
    # 95% of younger patients with no side effects
    random_younger = randint(13,64) # Generate to random age
    train_samples.append(random_younger) # Append age to sample dataset
    train_labels.append(0) # Append 0 (false for side effects) for the associated label

     # 95% of older individuals who did experience side effects
    random_older = randint(65,100) # Generate random age
    train_samples.append(random_older) # Append age to sample dataset
    train_labels.append(1) # Append 1 (true for side effects) for the associated label

# Create test dataset
# Outliers (5% of younger with side effects and 5% of older without side effects)
for i in range(10):
    # 5% of younger patients with side effects
    random_younger = randint(13,64) # Generate to random age
    test_samples.append(random_younger) # Append age to sample dataset
    test_labels.append(1) # Append 1 (true for side effects) for the associated label

    # 5% of older individuals who did not experience side effects
    random_older = randint(65,100) # Generate random age
    test_samples.append(random_older) # Append age to sample dataset
    test_labels.append(0) # Append 0 (false for side effects) for the associated label

# Normal samples (95% younger without side effects and 95% older with side effects)
for i in range(200):
    # 95% of younger patients with no side effects
    random_younger = randint(13,64) # Generate to random age
    test_samples.append(random_younger) # Append age to sample dataset
    test_labels.append(0) # Append 0 (false for side effects) for the associated label

     # 95% of older individuals who did experience side effects
    random_older = randint(65,100) # Generate random age
    test_samples.append(random_older) # Append age to sample dataset
    test_labels.append(1) # Append 1 (true for side effects) for the associated label


# Transform data to optimize the model
# Convert to numpy arrays
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
train_labels, train_samples = shuffle(train_labels, train_samples) # Shuffle arrays respective to each other to get rid of imposed order
test_labels, test_samples = shuffle(test_labels, test_samples) # Shuffle arrays respective to each other to get rid of imposed order


# Normalize data and scale down (13-64 = 0, 65+ = 1)
scaler = MinMaxScaler(feature_range=(0,1)) # Generates percentage (0.x) of 13-64. Returns 1 if 65+ 
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))


# Utilize GPU instead of CPU
physical_devices = tf.config.experimental.list_logical_devices('GPU')
print('Available GPUSs: ', len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

 

# Initialize sequential model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# Sumarize model
#model.summary()

# Prepare model for training (Use 10% for validatiom set)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2) # This is probably overfitting



#  Predictions based on test data
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

#for i in rounded_predictions:
 #   print(i)

