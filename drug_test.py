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
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os.path

# ---INITIALIZE LABELS AND SAMPLES---
train_labels = []
train_samples = []
test_labels = []
test_samples = []



# ---CREATE SAMPLE DATASET---
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



# ---TRANSFORM DATA TO OPTIMIZE THE MODEL---
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



# ---UTILIZE GPU INSTEAD OF CPU---
physical_devices = tf.config.experimental.list_logical_devices('GPU')
print('Available GPUSs: ', len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

 

# ---INITIALIZE SEQUENTIAL MODEL---
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# Sumarize model
#model.summary()

# Prepare model for training (Use 10% for validatiom set)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=0) # This is probably overfitting



#  ---PREDICTIONS BASED ON TEST DATA---
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=2)
rounded_predictions = np.argmax(predictions, axis=-1) # Get index of predictions with highest probability



# ---CONFUSION MATRIX---
# Visualizes prediction results
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# Plot prediction labels
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')



# ---SAVE MODEL---
if os.path.isfile('models/medical_trial_model.keras') is False:
    model.save('models/medical_trial_model.keras')

# Load model and summarize it
new_model = load_model('models/medical_trial_model.keras') # Completely new model loaded from what's on disk
#new_model.summary()



# ---MODEL ARCHITECTURE AS JSON---
# Save model as JSON 
json_string = model.to_json()

# Create new model from JSON architecture
model_architecture = model_from_json(json_string)
model_architecture.summary()



# ---SAVE WEIGHTS OF A MODEL---
if os.path.isfile('models/my_model_weights.keras') is False:
    model.save_weights('models/my_model_weights.keras')

# Load saved weights in a new neural network
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model2.load_weights('models/my_model_weights.keras')
model2.get_weights()