import tensorflow as tf


def build_neural_network(n_features):
    # layer input
    inputs = tf.keras.layers.Input(name="input", shape=(n_features,))

    # hidden layer 1
    h1 = tf.keras.layers.Dense(name="h1", units=int(round((n_features + 1) / 2)), activation='relu')(inputs)
    h1 = tf.keras.layers.Dropout(name="drop1", rate=0.2)(h1)

    # hidden layer 2
    h2 = tf.keras.layers.Dense(name="h2", units=int(round((n_features + 1) / 4)), activation='relu')(h1)
    h2 = tf.keras.layers.Dropout(name="drop2", rate=0.2)(h2)

    # layer output
    outputs = tf.keras.layers.Dense(name="output", units=1, activation='sigmoid')(h2)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
    model.summary()

    return model


build_neural_network(n_features=10)
