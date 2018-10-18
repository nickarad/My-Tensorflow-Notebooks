# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# import tensorflow.keras as keras
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import h5py
from tensorflow import keras

from keras.models import model_from_json


# print(tf.__version__)

# -- dataset of hand-written digits, 0 through 9. It's 28x28 images of these hand-written digits
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()
# -- the x_train data is the "features."
# In this case, the features are pixel values of the 28x28 images of these digits 0-9.
# -- The y_train is the label (is it a 0,1,2,3,4,5,6,7,8 or a 9?)

# print(x_train[0])

# -- visualise data
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
#-- print(y_train[0])

# -- Normalise data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_train = x_train.reshape(-1, 28*28)

x_test = tf.keras.utils.normalize(x_test, axis = 1)
x_test = x_test.reshape(-1, 28*28)
# print(x_train[0])

# -- Building the model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(784,)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# # --  output layer
# model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
# # --  "compile" the model

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# -- fit model
model = create_model()
model.fit(x_train, y_train, epochs=3)

# -- model accurancy
val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy


# -- Make predictions
predictions = model.predict(x_test)
print("prediction:", np.argmax(predictions[8]))
print("real value:", y_test[8])

# print(np.argmax(predictions[8]))
plt.imshow(x_test[8], cmap=plt.cm.binary)
plt.show()

model.summary()
# Save entire model
model.save('my_model.h5')