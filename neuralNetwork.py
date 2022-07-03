from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.backend import sum
from tensorflow.python.keras.backend import round
from tensorflow.python.keras.backend import clip
from tensorflow.python.keras.backend import epsilon
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

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
def plot_results(training, simNr, ord, ncyc):
    # plot
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.tight_layout(pad=3.0)

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black', label="loss")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[0].legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black', label="loss")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    ax22.legend()
    ax[1].legend()

    plot_path = f'./figures/neuralNetworks/'
    filename = "sim" + str(simNr) + "_superlet_ord" + str(ord) + "_ncyc" + str(ncyc)

    plt.savefig(plot_path + filename + ".png")

    return plt


# ------------------------------------ define F1 score
def F1(y_true, y_pred):
    true_positives = sum(round(clip(y_true * y_pred, 0, 1)))
    predicted_positives = sum(round(clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())

    possible_positives = sum(round(clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())

    return 2 * ((precision * recall) / (precision + recall + epsilon()))


def build_neural_network(n_features, output_size):
    inputLayer = Input(name="input", shape=(n_features,))
    hiddenLayer = Dense(name="hidden", units=int(round((n_features + 1) / 2)),
                        activation='relu')(inputLayer)
    outputLayer = Dense(name="output", units=output_size, activation='sigmoid')(hiddenLayer)

    model = Model(inputs=inputLayer, outputs=outputLayer, name="neuralNetwork")
    model.summary()

    return model


def apply_neural_network(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    slt_features = slt.slt(spikes, ord, ncyc)

    # flatten the result from the superlet
    slt_features = np.asarray(slt_features)

    # scale data
    scaler = MinMaxScaler()
    scaled_slt_features = scaler.fit_transform(slt_features)

    # labels encoding
    encoded_labels = to_categorical(labels.reshape(-1, 1).astype(int))
    encoded_labels = np.array(encoded_labels)

    visualize_labels_after_preprocessing(labels, encoded_labels)

    # Separate the test data
    features, features_test, encoded_labels, encoded_labels_test = train_test_split(slt_features,
                                                                                    encoded_labels, test_size=0.15,
                                                                                    shuffle=True)

    # Split the remaining data to train and validation
    features_train, features_validation, labels_train, labels_validation = train_test_split(features,
                                                                                            encoded_labels,
                                                                                            test_size=0.15,
                                                                                            shuffle=True)
    visualize_data_after_split(slt_features, labels, features_test, encoded_labels_test,
                               features_train, labels_train, features_validation, labels_validation)

    model = build_neural_network(n_features=500, output_size=encoded_labels.shape[-1])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', F1])

    # train/validation
    training = model.fit(x=features_train, y=labels_train, batch_size=32, epochs=100, shuffle=True, verbose=1,
                         validation_data=(features_validation, labels_validation))

    result = model.evaluate(features_test, encoded_labels_test, verbose=1)
    print("Test loss function: ", result[0])
    print("Test accuracy: ", result[1])
    print("Test F1 score: ", result[2])

    plt = plot_results(training, simNr, ord, ncyc)
    plt.show()


def visualize_labels_after_preprocessing(labels, encoded_labels):
    print('Labels shape before preprocessing: ' + str(labels.shape))
    print('Unique labels: ' + str(np.unique(labels)))
    print('First three labels shape: ' + str(labels[0:3]))
    print('---------------------------------------------------------------')
    print('Labels shape after preprocessing: ' + str(encoded_labels.shape))
    print('Unique labels: ' + str(np.unique(encoded_labels)))
    print('First three labels shape: ' + str(encoded_labels[0:3]))


def visualize_data_after_split(slt_features, labels, features_test, labels_test,
                               features_train, labels_train, features_validation, labels_validation):
    print('No of slt features:' + str(len(slt_features)))
    print('Length of each feature: ' + str(len(slt_features[0])))
    print('No of labels:' + str(len(labels)))
    print('---------------------------------------------------------------')
    print('No of slt features for training:' + str(len(features_train)))
    print('No of labels for training:' + str(len(labels_train)))
    print('---------------------------------------------------------------')
    print('No of slt features for validation:' + str(len(features_validation)))
    print('No of labels for validation:' + str(len(labels_validation)))
    print('---------------------------------------------------------------')
    print('No of slt features for testing:' + str(len(features_test)))
    print('No of labels for testing:' + str(len(labels_test)))


# apply_neural_network(simNr=8, ord=2, ncyc=1.5)
