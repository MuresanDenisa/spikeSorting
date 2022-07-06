import tensorflow as tf
import simulations_dataset as ds
import superlets as slt
import numpy as np
from sklearn.model_selection import train_test_split


def build_cnn(data_size, n_features):
    model = tf.keras.models.Sequential()
    # input layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(data_size, n_features)))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(1, activation='softmax'))

    return model


def apply_cnn(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    slt_features = slt.slt(spikes, ord, ncyc)

    slt_features = np.asarray(slt_features)
    x_train, x_valid, y_train, y_valid = train_test_split(slt_features, labels, test_size=0.33, shuffle=True)

    # aici am incercat sa fac reshape la date, dar momentan tot nu e bie
    x_train = np.reshape(x_train, [1, len(x_train), 500])
    x_valid = np.reshape(x_valid, [1, len(x_train), 500])

    model = build_cnn(len(x_train), n_features=500)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_valid, y_valid))

    # plot_result ii o functie prin care imi plotez graficu, poti gasi pe net
    # plt = plot_results(history, simNr, ord, ncyc, mode=0)
    # plt.show()