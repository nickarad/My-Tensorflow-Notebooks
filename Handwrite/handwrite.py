# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# import tensorflow.keras as keras
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import createmodel
from tensorflow import keras

from keras.models import model_from_json

# print(tf.__version__)

# -- dataset of hand-written digits, 0 through 9. It's 28x28 images of these hand-written digits
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

# -- visualise data
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
#-- print(y_train[0])

# -- Normalise data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_train_reshaped = x_train.reshape(-1, 28*28)

x_test = tf.keras.utils.normalize(x_test, axis = 1)
x_test_reshaped = x_test.reshape(-1, 28*28)

# print(x_train[0])


# -- fit model
model = createmodel.create_model()
model.fit(x_train_reshaped, y_train, epochs=3)

# -- model accurancy
val_loss, val_acc = model.evaluate(x_test_reshaped, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy


# -- Make predictions
predictions = model.predict(x_test_reshaped)
print("prediction:", np.argmax(predictions[8]))
print("real value:", y_test[8])

# print(np.argmax(predictions[8]))
plt.imshow(x_test[8], cmap=plt.cm.binary)
plt.show()

model.summary()
# Save entire model
model.save('my_model.h5')