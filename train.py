import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from visualize import visualize_activations
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
import app

# Load data
input_data, labels = load_galaxy_data()
print(input_data.shape) # RGB data
print(labels.shape) # 4 Classification labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size = 0.20, shuffle = True, random_state = 222, stratify = labels)

# Preprocess input
data_generator = ImageDataGenerator(rescale = 1.0/255)

training_iterator = data_generator.flow(X_train, y_train, batch_size = 5)

validation_iterator = data_generator.flow(X_test, y_test, batch_size = 5)

# Build Convolutional Neural Network model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2, 2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(4, activation = 'softmax'))

# Compile model
model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
  loss = tf.keras.losses.CategoricalCrossentropy(),
  metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

model.summary()

# Train Model
model.fit(training_iterator,
          steps_per_epoch = len(X_train)/5,
          epochs = 8,
          validation_data = validation_iterator,
          validation_steps = len(X_test)/5)

# Visualize CNN Images
visualize_activations(model, validation_iterator)
