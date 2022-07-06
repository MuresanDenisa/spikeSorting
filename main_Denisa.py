import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import simulations_dataset
import scatter_plot
import numpy as np
import superlets as slt
import sklearn
from constants import LABEL_COLOR_MAP
import main_spectrum as ms
import neuralNetwork


# functie care ploteaza individual spike-ul mediu pentru fiecare cluster
def show_average_spike_per_cluster(simNr, spike, unique_label):
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("gray")
    plt.title("Average spike for cluster " + str(unique_label) + "in simulation" + str(simNr))

    plt.plot(range(len(spike)), spike, color=LABEL_COLOR_MAP[unique_label],
             label='cluster {unique_label}'.format(unique_label=unique_label))

    plt.legend(loc='best')
    plt.show()

# functie ce genereaza spike-urile medii per clustere
def average_spike_per_cluster(simNr, onePlot=True):
    spikes, labels = simulations_dataset.get_dataset_simulation(simNr=simNr)

    if onePlot:
        plt.figure()
        ax = plt.axes()
        ax.set_facecolor("gray")
        plt.title("Average spikes for simulation " + str(simNr))

    for unique_label in np.unique(labels):
        selected_spikes = spikes[labels == unique_label]
        average_spike = np.mean(selected_spikes, axis=0)
        if onePlot:
            plt.plot(range(len(average_spike)), average_spike, color=LABEL_COLOR_MAP[unique_label],
                     label='cluster {unique_label}'.format(unique_label=unique_label))
        else:
            show_average_spike_per_cluster(simNr, average_spike, unique_label)

    if onePlot:
        plt.legend(loc='best')
        plt.show()


# functie pentru aplicarea reducerii dimensionalitatii prin PCA
def apply_dim_reduction(results, no_of_components):
    scaler = StandardScaler()
    features = scaler.fit_transform(results)
    pca = PCA(n_components=no_of_components)

    return pca.fit_transform(features)


# functie pentru calculul scorurilor metricilor de performanta
def compute_metrics_scores(slt_features, labels):
    db_score = sklearn.metrics.davies_bouldin_score(slt_features, labels)
    ch_score = sklearn.metrics.calinski_harabasz_score(slt_features, labels)
    s_score = sklearn.metrics.silhouette_score(slt_features, labels)
    print("Score Davies-Bouldin: ", db_score)
    print("Score Calinsky-Harabasz: ", ch_score)
    print("Score Silhouette: ", s_score)


# functie care calculeaza valoarea minima si maxima in urma aplicarii
# transformatei Superlet/Faslt (in functie de variabila booleana adaptive)
# folosita pentru normalizarea spectrelor
def find_min_max(spikes, ord_min, ord_max, ncyc, adaptive):
    results = []
    for spike in spikes:
        if adaptive:
            result = faslt_per_spike(spike, ord_min, ord_max, ncyc)
        else:
            result = slt.slt_1spike_without_plot(spike, ord_min, ncyc, 1, 200)
        results.append((np.ndarray.flatten(np.array(result))))

    return np.min(results), np.max(results)


# functie pentru generarea spectrului timp-frecventa per cluster in urma aplicarii
# transfromatei Superlet pe fiecare spike in parte
def spectrum_with_superlet(simNr, spikes, label, ord, ncyc, normalized):
    ampl_min = 0
    ampl_max = 0
    if normalized:
        ampl_min, ampl_max = find_min_max(spikes, ord, ord, ncyc, False)

    results = []

    for spike in spikes:
        result = slt.slt_1spike_without_plot(spike, ord, ncyc, 1, 200)
        results.append(result)

    average_result = np.mean(results, axis=0)

    scales_freq = np.arange(1, 200)
    df = scales_freq[-1] / scales_freq[-2]
    ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
    im = plt.pcolormesh(np.arange(79), ymesh, average_result, cmap="jet")
    plt.colorbar(im)

    if normalized:
        plot_path = f'./figures/' + str(simNr) + '/normalized/'
        plt.clim(ampl_min, ampl_max)
    else:
        plot_path = f'./figures/' + str(simNr)
    filename = "cluster_" + str(label) + "_ord" + str(ord) + "_ncyc" + str(ncyc) + "_superlet"

    plt.title(filename)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(plot_path + filename + ".png")
    plt.show()


# functie care aplica FASLT pe fiecare spike
def faslt_per_spike(spike, order_min, order_max, ncyc):
    foi = np.linspace(1, 250)
    scales = ms.scale_from_period(1 / foi)

    spec = ms.superlet(
        spike,
        samplerate=1000,
        scales=scales,
        order_max=order_max,
        order_min=order_min,
        c_1=ncyc,
        adaptive=True,
    )

    return np.abs(spec)

# functie pentru generarea spectrului timp-frecventa per cluster in urma aplicarii
# transfromatei FASLT pe fiecare spike in parte
def spectrum_with_faslt(simNr, spikes, label, order_min, order_max, ncyc, normalized):
    # frequencies of interest in Hz
    foi = np.linspace(1, 250)

    ampl_min = 0
    ampl_max = 0
    if normalized:
        ampl_min, ampl_max = find_min_max(spikes, order_min, order_max, ncyc, True)

    results = []
    for spike in spikes:
        result = faslt_per_spike(spike, order_min, order_max, ncyc)
        results.append(result)

    average_result = np.mean(results, axis=0)

    extent = [0, 79, foi[0], foi[-1]]
    im = plt.imshow(average_result, cmap="jet", aspect="auto", extent=extent, origin='lower')
    plt.colorbar(im)

    if normalized:
        plot_path = f'./figures/' + str(simNr) + '/adaptive/normalized/'
        plt.clim(ampl_min, ampl_max)
    else:
        plot_path = f'./figures/' + str(simNr) + '/adaptive/'

    filename = "cluster" + str(label) + "_superlets_ord[" + str(order_min) + "," + str(order_max) + "]_ncyc" + str(ncyc)
    plt.title(filename)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(plot_path + filename + ".png")

    plt.show()


# functie pentru generarea spectrului timp frecventa prin care se alege metoda
# de extragere a caracteristicilor (FASLT sau Superlet) si selecteaza toate
# spike-urile dintr-un cluter pt a generat spectru per cluster
def spectrum(simNr, ord_min, ord_max, ncyc, normalized, adaptive=False):
    spikes, labels = simulations_dataset.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    for unique_label in np.unique(labels):
        selected_spikes = spikes[labels == unique_label]
        if adaptive:
            spectrum_with_faslt(simNr, selected_spikes, unique_label, ord_min, ord_max, ncyc, normalized)
        else:
            spectrum_with_superlet(simNr, selected_spikes, unique_label, ord_min, ncyc, normalized)


# functie pentru aplicarea pipeline-ului pentru transformata Superlet
# cu ord si ncyc dati ca parametrii de intrare, aplicarea reducerea
# dimensionalitatii in functie de no_of_components, evaluarea performantei
# prin scorurile metricilor, generarea unui grafic de vizualizare si
# antrenarea unei retele neuronale pe trasaturile extrase de Superlet
def pipeline_superlet(simNr, ord, ncyc, no_of_components):
    spikes, labels = simulations_dataset.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    slt_features = slt.slt(spikes, ord, ncyc)
    features_after_pca = apply_dim_reduction(slt_features, no_of_components)
    compute_metrics_scores(slt_features, labels)

    plot_path = f'./figures/' + str(simNr) + '/'
    filename = 'sim_' + str(simNr) + '_ord' + str(ord) + '_ncyc' + str(ncyc) + 'in'\
               + str(no_of_components) + 'D'

    scatter_plot.plot(filename, features_after_pca, labels, marker='o')
    plt.savefig(plot_path + filename + ".png")
    plt.show()

    neuralNetwork.apply_neural_network(slt_features, labels, simNr, ord, ncyc)


# functie pentru aplicarea pipeline-ului pentru transformata FASLT
# cu ord si ncyc dati ca parametrii de intrare, aplicarea reducerea
# dimensionalitatii in functie de no_of_components, evaluarea performantei
# prin scorurile metricilor si generarea unui grafic de vizualizare
def pipeline_faslt(simNr, ord_min, ord_max, ncyc, no_of_components):
    spikes, labels = simulations_dataset.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    results = []
    for spike in spikes:
        result = faslt_per_spike(spike, ord_min, ord_max, ncyc)
        results.append((np.ndarray.flatten(np.array(result))))

    features_pca = apply_dim_reduction(results, no_of_components)

    file_name = "sim_" + str(simNr) + "_ord[" + str(ord_min) + "," + str(ord_max) + \
                "]_ncyc" + str(ncyc) + "_FASLT_" + str(no_of_components) + "D"
    plot_path = f'./figures/' + str(simNr) + '/'

    scatter_plot.plot(file_name, features_pca, labels, marker='o')

    plt.savefig(plot_path + file_name + ".png")
    plt.show()

    compute_metrics_scores(results, labels)


# functie ce genereaza un grafic cu statistici
# despre numarul de spike-uri din fiecare cluster dintr-o simmulare
def data_per_cluster(simNr):
    spikes, labels = simulations_dataset.get_dataset_simulation(simNr=simNr)

    results = []
    for unique_label in np.unique(labels):
        result = (labels == unique_label).sum()
        results.append(result)

    plt.bar(np.unique(labels), results)
    plt.title('Features per Cluster in Sim' + str(simNr))
    plt.ylabel('Features')
    plt.xlabel('Clusters')
    plt.show()


def main():
    # average_spike_per_cluster(simNr=33, onePlot=False)
    # data_per_cluster(simNr=84)
     pipeline_superlet(simNr=15, ord=15, ncyc=1.5, no_of_components=2)
    # spectrum(simNr=4, ord_min=2, ord_max=2, ncyc=1.5, normalized=False, adaptive=False)
    #  pipeline_faslt(simNr=84, ord_min=1.5, ord_max=3, ncyc=3, no_of_components=2)
    #  neuralNetwork.apply_neural_network(simNr=33, ord=2, ncyc=1.5)


main()
