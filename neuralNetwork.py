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


# functie care ploteaza graficul cu rezultatele antrenarii
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


# functie care defineste modul de calcul al scorului F1
def F1(y_true, y_pred):
    true_positives = sum(round(clip(y_true * y_pred, 0, 1)))
    predicted_positives = sum(round(clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())

    possible_positives = sum(round(clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())

    return 2 * ((precision * recall) / (precision + recall + epsilon()))


# functie care construieste reteaua neuronala
def build_neural_network(n_features, output_size):
    inputLayer = Input(name="input", shape=(n_features,))
    hiddenLayer = Dense(name="hidden", units=int(round((n_features + 1) / 2)),
                        activation='relu')(inputLayer)
    outputLayer = Dense(name="output", units=output_size, activation='sigmoid')(hiddenLayer)

    model = Model(inputs=inputLayer, outputs=outputLayer, name="neuralNetwork")
    model.summary()

    return model


# functie care importa setul de date, aplica Superlet si imparte rezultatele
# in training set, validation set si testing set
# antreneaza reteaua neuronala si o testeaza pe testing set
def apply_neural_network(simNr, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    slt_features = slt.slt(spikes, ord, ncyc)

    # flatten the result from the superlet
    slt_features = np.asarray(slt_features)

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

# functie pentru vizualizarea etichetelor inainte si dupa codificare
# pentru a asigura integritatea datelor
def visualize_labels_after_preprocessing(labels, encoded_labels):
    print('Labels shape before preprocessing: ' + str(labels.shape))
    print('Unique labels: ' + str(np.unique(labels)))
    print('First three labels shape: ' + str(labels[0:3]))
    print('---------------------------------------------------------------')
    print('Labels shape after preprocessing: ' + str(encoded_labels.shape))
    print('Unique labels: ' + str(np.unique(encoded_labels)))
    print('First three labels shape: ' + str(encoded_labels[0:3]))

# functie pentru a vizualiza statistici despre cum a fost impartit setul de date
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
