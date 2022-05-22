import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import simulations_dataset as ds
import superlets as slt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# ----------------------- VISUALIZATION METHODS --------------------------------------
# Extract info for each layer in a keras model.
def utils_nn_config(model):
    lst_layers = []
    if "Sequential" in str(model):  # -> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({"name": "input", "in": int(layer.input.shape[-1]), "neurons": 0,
                           "out": int(layer.input.shape[-1]), "activation": None,
                           "params": 0, "bias": 0})
    for layer in model.layers:
        try:
            dic_layer = {"name": layer.name, "in": int(layer.input.shape[-1]), "neurons": layer.units,
                         "out": int(layer.output.shape[-1]), "activation": layer.get_config()["activation"],
                         "params": layer.get_weights()[0], "bias": layer.get_weights()[1]}
        except:
            dic_layer = {"name": layer.name, "in": int(layer.input.shape[-1]), "neurons": 0,
                         "out": int(layer.output.shape[-1]), "activation": None,
                         "params": 0, "bias": 0}
        lst_layers.append(dic_layer)
    return lst_layers


# Plot the structure of a keras neural network.
def visualize_nn(model, description=False, figsize=(10, 8)):
    # get layers info
    lst_layers = utils_nn_config(model)
    layer_sizes = [layer["out"] for layer in lst_layers]

    # fig setup
    fig = plt.figure()
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right - left) / float(len(layer_sizes) - 1)
    y_space = (top - bottom) / float(max(layer_sizes))
    p = 0.025

    # nodes
    for i, n in enumerate(layer_sizes):
        top_on_layer = y_space * (n - 1) / 2.0 + (top + bottom) / 2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes) - 1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color

        # add description
        if description is True:
            d = i if i == 0 else i - 0.5
            if layer['activation'] is None:
                plt.text(x=left + d * x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left + d * x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left + d * x_space, y=top - p, fontsize=10, color=color, s=layer['activation'] + " (")
                plt.text(x=left + d * x_space, y=top - 2 * p, fontsize=10, color=color,
                         s="Σ" + str(layer['in']) + "[X*w]+b")
                out = " Y" if i == len(layer_sizes) - 1 else " out"
                plt.text(x=left + d * x_space, y=top - 3 * p, fontsize=10, color=color,
                         s=") = " + str(layer['neurons']) + out)

        # circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left + i * x_space, top_on_layer - m * y_space - 4 * p), radius=y_space / 4.0,
                                color=color, ec='k', zorder=4)
            ax.add_artist(circle)

            # add text
            if i == 0:
                plt.text(x=left - 4 * p, y=top_on_layer - m * y_space - 4 * p, fontsize=10,
                         s=r'$X_{' + str(m + 1) + '}$')
            elif i == len(layer_sizes) - 1:
                plt.text(x=right + 4 * p, y=top_on_layer - m * y_space - 4 * p, fontsize=10,
                         s=r'$y_{' + str(m + 1) + '}$')
            else:
                plt.text(x=left + i * x_space + p,
                         y=top_on_layer - m * y_space + (y_space / 8. + 0.01 * y_space) - 4 * p, fontsize=10,
                         s=r'$H_{' + str(m + 1) + '}$')

    # links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i + 1]
        color = "green" if i == len(layer_sizes) - 2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space * (n_a - 1) / 2. + (top + bottom) / 2. - 4 * p
        layer_top_b = y_space * (n_b - 1) / 2. + (top + bottom) / 2. - 4 * p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i * x_space + left, (i + 1) * x_space + left],
                                  [layer_top_a - m * y_space, layer_top_b - o * y_space],
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()


# ------------------------------------ PLOT RESULTS
def plot_results(training, simNr, ord, ncyc, mode):
    # plot
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.tight_layout(pad=3.0)

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    ax22.legend()

    if mode:
        plot_path = f'./figures/neuralNetworks/'
    else:
        plot_path = f'./figures/convolutionalNeuralNetworks/'

    filename = "sim" + str(simNr) + "_superlet_ord" + str(ord) + "_ncyc" + str(ncyc)

    plt.savefig(plot_path + filename + ".png")

    return plt


# ------------------------------------ define metrics
def Recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def Precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def build_neural_network(n_features):
    # layer input
    inputs = tf.keras.layers.Input(name="input", shape=(n_features,))

    # hidden layer 1
    h1 = tf.keras.layers.Dense(name="h1", units=int(round((n_features + 1) / 2)), activation='elu')(inputs)

    # hidden layer 2
    h2 = tf.keras.layers.Dense(name="h2", units=int(round((n_features + 1) / 4)), activation='elu')(h1)

    # layer output
    outputs = tf.keras.layers.Dense(name="output", units=1, activation='softmax')(h2)
    # outputs = tf.keras.layers.Dense(name="output", units=1, activation='softmax')(h2)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
    # model.summary()

    return model


def build_cnn(data_size, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(data_size,n_features)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='softmax'))

    return model


def apply_neural_network(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    slt_features = slt.slt(spikes, ord, ncyc)
    list(np.repeat(labels, 500))

    # flatten the result from the superlets
    slt_features = np.asarray(slt_features)

    # scale data
    scaler = MinMaxScaler()
    scaled_slt_features = scaler.fit_transform(slt_features)

    # split data intro training and validation data
    x_train, x_valid, y_train, y_valid = train_test_split(scaled_slt_features, labels, test_size=0.33, shuffle=True)

    model = build_neural_network(n_features=500)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', F1])
    # train/validation
    training = model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, shuffle=True, verbose=0,
                         validation_data=(x_valid, y_valid))

    # training = model.fit(x=slt_features, y=labels, batch_size=32, epochs=100, shuffle=True, verbose=0,
    #                      validation_split=0.33)

    # model.evaluate(X, y, verbose=0)

    plt = plot_results(training, simNr, ord, ncyc, mode=1)
    plt.show()


def apply_cnn(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    slt_features = slt.slt(spikes, ord, ncyc)

    slt_features = np.asarray(slt_features)
    x_train, x_valid, y_train, y_valid = train_test_split(slt_features, labels, test_size=0.33, shuffle=True)
    x_train = np.reshape(x_train, [1, len(x_train), 500])
    x_valid = np.reshape(x_valid, [1, len(x_train), 500])

    model = build_cnn(len(x_train), n_features=500)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_valid, y_valid))

    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    plt = plot_results(history, simNr, ord, ncyc, mode=0)
    plt.show()


def visualize_data(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    slt_features = slt.slt(spikes, ord, ncyc)
    slt_features = np.asarray(slt_features)
    # print(slt_features)
    print('no of slt features:' + str(len(slt_features)))
    print('length of each feature: ' + str(len(slt_features[0])))
    print('no of labels:' + str(len(labels)))
    slt_features = np.asarray(slt_features)
    x_train, x_valid, y_train, y_valid = train_test_split(slt_features, labels, test_size=0.33, shuffle=True)
    print('no of slt features in x_train:' + str(len(x_train)))
    print('length of each feature in x_train: ' + str(len(x_train[0])))
    print('no of labels in y_train:' + str(len(y_train)))
    print('no of slt features in x_valid:' + str(len(x_valid)))
    print('length of each feature in x_valid: ' + str(len(x_valid[0])))
    print('no of labels in y_valid:' + str(len(y_valid)))


apply_neural_network(simNr=8, ord=2, ncyc=2)
# apply_cnn(simNr=8, ord=2, ncyc=2)
# visualize_nn(build_neural_network(5))
# visualize_data(8,2,2)