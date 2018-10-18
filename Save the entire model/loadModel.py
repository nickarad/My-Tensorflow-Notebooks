from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras

# ===================== Datasets ============================================================
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
#
# train_images = train_images[:1000].reshape(-1, 28*28)/255.0
# test_images = test_images[:1000].reshape(-1, 28*28)/255.0

# ============================================================================================

def create_model():
    model=tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model



# ================== Load weights from checkpoint and re-evaluate ===========================

# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy {:5.2f}%".format(100*acc))

new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# ===========================================================================================
