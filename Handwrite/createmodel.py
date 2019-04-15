import tensorflow as tf


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
