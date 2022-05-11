import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import simulations_dataset as ds
import scatter_plot
import numpy as np
import superlets as slt
import sklearn
from constants import LABEL_COLOR_MAP
import main_spectrum as ms


def superlet(simNr, ord, ncyc, dimension):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    # def slt(spikes, ord, ncyc, derivatives=True)
    slt_features = slt.slt(spikes, ord, ncyc)

    # scaler = StandardScaler()
    # features = scaler.fit_transform(spikes)
    # pca = PCA(n_components=dimension)
    # features_after_pca = pca.fit_transform(features)
    # scatter_plot.plot('spikes',
    #                   features_after_pca, labels, marker='o')
    # plt.show()

    scaler = StandardScaler()
    features = scaler.fit_transform(slt_features)
    pca = PCA(n_components=dimension)

    features_after_pca = pca.fit_transform(features)

    scatter_plot.plot('Sim_' + str(simNr) + '_ord' + str(ord) + '_ncyc' + str(ncyc) + 'in' + str(dimension) + 'D',
                      features_after_pca, labels, marker='o')
    plt.show()


def average_spike_per_cluster(simNr):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    for unique_label in np.unique(labels):
        selected_spikes = spikes[labels == unique_label]
        average_spike = np.mean(selected_spikes, axis=0)
        plt.plot(range(len(average_spike)), average_spike, color=LABEL_COLOR_MAP[unique_label],
                 label='cluster {unique_label}'.format(unique_label=unique_label))

    ax = plt.axes()
    ax.set_facecolor("gray")
    plt.title("Average spike for simulation " + str(simNr))
    plt.legend(loc='best')
    plt.show()


def plot_data_with_score(simulation, n_components, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation, align_to_peak=False)
    # def slt(spikes, ord, ncyc, derivatives=True)
    slt_features = slt.slt(spikes, ord, ncyc)
    clusters = max(labels)
    scaler = StandardScaler()
    features = scaler.fit_transform(slt_features)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    scatter_plot.plot("Sim_%d (%d clusters)_%dD" % (simulation, clusters, n_components), features_pca, labels,
                      marker='o')
    plt.show()
    score_davies1 = sklearn.metrics.davies_bouldin_score(slt_features, labels)
    score_calinski = sklearn.metrics.calinski_harabasz_score(slt_features, labels)
    score_silhouette = sklearn.metrics.silhouette_score(slt_features, labels)
    print("Score Davies-Bouldin: ", score_davies1)
    print("Score Calinsky-Harabasz: ", score_calinski)
    print("Score Silhouette: ", score_silhouette)


# def spectrum_help(simNr, spikes, label, ord, ncyc, freq_start, freq_end):
#     results = []
#
#     for spike in spikes:
#         result = slt.slt_1spike_without_plot(spike, ord, ncyc, freq_start, freq_end)
#         results.append(result)
#
#     average_result = np.mean(results, axis=0)
#     # magnitude = ms.fourier_power(average_result)
#
#     plot_path = f'./figures/' + str(simNr) + '/'
#     values = np.abs(average_result)
#     scales_freq = np.arange(freq_start, freq_end)
#     df = scales_freq[-1] / scales_freq[-2]
#     ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
#     im = plt.pcolormesh(np.arange(79), ymesh, values, cmap="jet")
#     plt.colorbar(im)
#     plt.title("cluster_" + str(label) + "_ord" + str(ord) + "_ncyc" + str(ncyc) + "_superlets")
#     plt.xlabel("Time")
#     plt.ylabel("Frequency - linear")
#     plt.savefig(plot_path + "cluster_" + str(label) + "_ord" + str(ord) + "_ncyc" +  str(ncyc) + "_superlets.png")
#     plt.show()

def find_min_max(spikes, ord_min, ord_max, ncyc):
    results = []
    for spike in spikes:
        result = faslt_per_spike(spike, ord_min, ord_max, ncyc)
        results.append((np.ndarray.flatten(np.array(result))))

    return np.min(results), np.max(results)


def spectrum(simNr, ord_min, ord_max, ncyc, normalized, adaptive=False):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    ampl_min = 0
    ampl_max = 0
    if normalized:
        ampl_min, ampl_max = find_min_max(spikes, ord_min, ord_max, ncyc)

    for unique_label in np.unique(labels):
        selected_spikes = spikes[labels == unique_label]
        if adaptive:
            spectrum_with_faslt(simNr, selected_spikes, unique_label, ord_min, ord_max, ncyc,
                                ampl_min, ampl_max, normalized)
        else:
            spectrum_help(simNr, selected_spikes, unique_label, ord_min, ord_max, ncyc)


def spectrum_help(simNr, spikes, label, ord_min, ord_max, ncyc):
    results = slt.slt_spectrum(spikes, ord_min, ord_max, ncyc)
    values = np.abs(results)

    scales_freq = np.arange(len(results) - 1)
    print(scales_freq)

    plot_path = f'./figures/' + str(simNr) + '/simple'
    df = scales_freq[-1] / scales_freq[-2]
    ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
    im = plt.pcolormesh(np.arange(79), ymesh, values, cmap="jet")
    plt.colorbar(im)
    plt.title("cluster" + str(label) + "_superlets_ord[" + str(ord_min) + "," + str(ord_max) + "]_ncyc" + str(ncyc))
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(
        plot_path + "cluster" + str(label) + "_superlets_ord[" + str(ord_min) + "," + str(ord_max) + "]_ncyc" + str(
            ncyc) + ".png")
    plt.show()


def faslt_per_spike(spike, order_min, order_max, ncyc):
    fs = 1000  # sampling frequency
    # frequencies of interest in Hz
    foi = np.linspace(1, 250)
    scales = ms.scale_from_period(1 / foi)

    spec = ms.superlet(
        spike,
        samplerate=fs,
        scales=scales,
        order_max=order_max,
        order_min=order_min,
        c_1=ncyc,
        adaptive=True,
    )

    return np.abs(spec)


def spectrum_with_faslt(simNr, spikes, label, order_min, order_max, ncyc, ampl_min, ampl_max, normalized):
    # frequencies of interest in Hz
    foi = np.linspace(1, 250)

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


def faslt_scores(simNr, ord_min, ord_max, ncyc, no_of_components):
    plot_path = f'./figures/' + str(simNr) + '/'
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)

    results = []
    for spike in spikes:
        result = faslt_per_spike(spike, ord_min, ord_max, ncyc)
        results.append((np.ndarray.flatten(np.array(result))))

    scaler = StandardScaler()
    features = scaler.fit_transform(results)
    pca = PCA(n_components=no_of_components)
    features_pca = pca.fit_transform(features)

    file_name = "Sim_" + str(simNr) + "_ord[" + str(ord_min) + "," + str(ord_max) + \
                "]_ncyc" + str(ncyc) + "_FASLT_" + str(no_of_components) + "D"

    scatter_plot.plot(file_name, features_pca, labels, marker='o')

    plt.savefig(plot_path + file_name + ".png")
    plt.show()

    score_davies = sklearn.metrics.davies_bouldin_score(results, labels)
    score_calinski = sklearn.metrics.calinski_harabasz_score(results, labels)
    score_silhouette = sklearn.metrics.silhouette_score(results, labels)
    print("Score Davies-Bouldin: ", score_davies)
    print("Score Calinsky-Harabasz: ", score_calinski)
    print("Score Silhouette: ", score_silhouette)


def main():
    # average_spike_per_cluster(4)
    superlet(8, 3, 1.5, 2)
    # superlet_feature_extraction(4, 2, 1.5)
    # plot_data_with_score(64, 3, 2, 1.5)
    # spectrum(simNr=64, ord_min=1, ord_max=5, ncyc=1.5, normalized=True, adaptive=True)
    # faslt_scores(simNr=64, ord_min=1, ord_max=1, ncyc=3, no_of_components=2)


main()
