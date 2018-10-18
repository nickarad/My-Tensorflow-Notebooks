import tensorflow as tf
from tensorflow import keras

# ===================== Datasets ============================================================
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

x_train_reshaped = x_train.reshape(-1, 28*28)
x_test_reshaped = x_test.reshape(-1, 28*28)

# print(x_train[0])

# ============================================================================================

# ================== Load weights from checkpoint and re-evaluate ===========================

new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(x_test_reshaped, y_test)
print("Restored model, accuracy {:5.2f}%".format(100*acc))

# ===========================================================================================