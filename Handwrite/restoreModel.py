from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras

# ===================== Datasets ============================================================
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
# print(x_train[0])

# ============================================================================================

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model



# ================== Load weights from checkpoint and re-evaluate ===========================

# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy {:5.2f}%".format(100*acc))
# new_model = create_model()
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# ===========================================================================================