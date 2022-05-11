import matplotlib.pyplot as plt
import sklearn
from numpy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import simulations_dataset as ds
import scatter_plot
import superlets as slt
import matplotlib.pyplot as ppl
import numpy as np
from scipy.signal import fftconvolve
from constants import LABEL_COLOR_MAP


def superlet(
        data_arr,
        samplerate,
        scales,
        order_max,
        order_min=1,
        c_1=3,
        adaptive=False,
):
    """
    Performs Superlet Transform (SLT) according to Moca et al. [1]_
    Both multiplicative SLT and fractional adaptive SLT are available.
    The former is recommended for a narrow frequency band of interest,
    whereas the  is better suited for the analysis of a broad range
    of frequencies.
    A superlet (SL) is a set of Morlet wavelets with increasing number
    of cycles within the Gaussian envelope. Hence the bandwith
    is constrained more and more with more cycles yielding a sharper
    frequency resolution. Complementary the low cycle numbers will give a
    high time resolution. The SLT then is the geometric mean
    of the set of individual wavelet transforms, combining both wide
    and narrow-bandwidth wavelets into a super-resolution estimate.
    Parameters
    ----------
    data_arr : nD :class:`numpy.ndarray`
        Uniformly sampled time-series data
        The 1st dimension is interpreted as the time axis
    samplerate : float
        Samplerate of the time-series in Hz
    scales : 1D :class:`numpy.ndarray`
        Set of scales to use in wavelet transform.
        Note that for the SL Morlet the relationship
        between scale and frequency simply is s(f) = 1/(2*pi*f)
        Need to be ordered high to low for `adaptive=True`
    order_max : int
        Maximal order of the superlet set. Controls the maximum
        number of cycles within a SL together
        with the `c_1` parameter: c_max = c_1 * order_max
    order_min : Minimal order of the superlet set. Controls
        the minimal number of cycles within a SL together
        with the `c_1` parameter: c_min = c_1 * order_min
        Note that for admissability reasons c_min should be at least 3!
    c_1 : int
        Number of cycles of the base Morlet wavelet. If set to lower
        than 3 increase `order_min` as to never have less than 3 cycles
        in a wavelet!
    adaptive : bool
        Wether to perform multiplicative SLT or fractional adaptive SLT.
        If set to True, the order of the wavelet set will increase
        linearly with the frequencies of interest from `order_min`
        to `order_max`. If set to False the same SL will be used for
        all frequencies.

    Returns
    -------
    gmean_spec : :class:`numpy.ndarray`
        Complex time-frequency representation of the input data.
        Shape is (len(scales), data_arr.shape[0], data_arr.shape[1]).
    Notes
    -----
    .. [1] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
       Nature communications 12.1 (2021): 1-18.


    """

    # adaptive SLT
    if adaptive:

        gmean_spec = FASLT(data_arr, samplerate, scales, order_max, order_min, c_1)

    # multiplicative SLT
    else:

        gmean_spec = multiplicativeSLT(
            data_arr, samplerate, scales, order_max, order_min, c_1
        )

    return gmean_spec


def multiplicativeSLT(data_arr, samplerate, scales, order_max, order_min=1, c_1=3):
    dt = 1 / samplerate
    # create the complete multiplicative set spanning
    # order_min - order_max
    cycles = c_1 * np.arange(order_min, order_max + 1)
    order_num = order_max + 1 - order_min  # number of different orders
    SL = [MorletSL(c) for c in cycles]

    # lowest order
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)
    gmean_spec = np.power(gmean_spec, 1 / order_num)

    for wavelet in SL[1:]:
        spec = cwtSL(data_arr, wavelet, scales, dt)
        gmean_spec *= np.power(spec, 1 / order_num)

    return gmean_spec


def FASLT(data_arr, samplerate, scales, order_max, order_min=1, c_1=3):
    """ Fractional adaptive SL transform
    For non-integer orders fractional SLTs are
    calculated in the interval [order, order+1) via:

    R(o_f) = R_1 * R_2 * ... * R_i * R_i+1 ** alpha
    with o_f = o_i + alpha
    """

    dt = 1 / samplerate
    # frequencies of interest
    # from the scales for the SL Morlet
    fois = 1 / (2 * np.pi * scales)
    orders = compute_adaptive_order(fois, order_min, order_max)

    # create the complete superlet set from
    # all enclosed integer orders
    orders_int = np.int32(np.floor(orders))
    cycles = c_1 * np.unique(orders_int)
    SL = [MorletSL(c) for c in cycles]

    # every scale needs a different exponent
    # for the geometric mean
    exponents = 1 / (orders - order_min + 1)

    # which frequencies/scales use the same integer orders SL
    order_jumps = np.where(np.diff(orders_int))[0]
    # each frequency/scale will have its own multiplicative SL
    # which overlap -> higher orders have all the lower orders

    # the fractions
    alphas = orders % orders_int

    # 1st order
    # the lowest order is needed for all scales/frequencies
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)  # 1st order <-> order_min
    # Geometric normalization according to scale dependent order
    gmean_spec = np.power(gmean_spec.T, exponents).T

    # we go to the next scale and order in any case..
    # but for order_max == 1 for which order_jumps is empty
    last_jump = 1

    for i, jump in enumerate(order_jumps):
        # relevant scales for the next order
        scales_o = scales[last_jump:]
        # order + 1 spec
        next_spec = cwtSL(data_arr, SL[i + 1], scales_o, dt)

        # which fractions for the current next_spec
        # in the interval [order, order+1)
        scale_span = slice(last_jump, jump + 1)
        gmean_spec[scale_span, :] *= np.power(
            next_spec[: jump - last_jump + 1].T,
            alphas[scale_span] * exponents[scale_span],
        ).T

        # multiply non-fractional next_spec for
        # all remaining scales/frequencies
        gmean_spec[jump + 1:] *= np.power(
            next_spec[jump - last_jump + 1:].T, exponents[jump + 1:]
        ).T

        # go to the next [order, order+1) interval
        last_jump = jump + 1

    return gmean_spec


class MorletSL:
    def __init__(self, c_i=3, k_sd=5):
        """ The Morlet formulation according to
        Moca et al. shifts the admissability criterion from
        the central frequency to the number of cycles c_i
        within the Gaussian envelope which has a constant
        standard deviation of k_sd.
        """

        self.c_i = c_i
        self.k_sd = k_sd

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Complex Morlet wavelet in the SL formulation.

        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time

        """

        ts = t / s
        # scaled time spread parameter
        # also includes scale normalisation!
        B_c = self.k_sd / (s * self.c_i * (2 * np.pi) ** 1.5)

        output = B_c * np.exp(1j * ts)
        output *= np.exp(-0.5 * (self.k_sd * ts / (2 * np.pi * self.c_i)) ** 2)

        return output


def fourier_period(scale):
    """
    This is the approximate Morlet fourier period
    as used in the source publication of Moca et al. 2021
    Note that w0 (central frequency) is always 1 in this
    Morlet formulation, hence the scales are not compatible
    to the standard Wavelet definitions!
    """

    return 2 * np.pi * scale


def scale_from_period(period):
    return period / (2 * np.pi)


def cwtSL(data, wavelet, scales, dt):
    """
    The continuous Wavelet transform specifically
    for Morlets with the Superlet formulation
    of Moca et al. 2021.
    - Morlet support gets adjusted by number of cycles
    - normalisation is with 1/(scale * 4pi)
    - this way the norm of the spectrum (modulus)
      at the corresponding harmonic frequency is the
      harmonic signal's amplitude
    Notes
    -----

    The time axis is expected to be along the 1st dimension.
    """

    # wavelets can be complex so output is complex
    output = np.zeros((len(scales),) + data.shape, dtype=np.complex64)

    # this checks if really a Superlet Wavelet is being used
    if not isinstance(wavelet, MorletSL):
        raise ValueError("Wavelet is not of MorletSL type!")

    # 1st axis is time
    slices = [None for _ in data.shape]
    slices[0] = slice(None)

    # compute in time
    for ind, scale in enumerate(scales):
        t = _get_superlet_support(scale, dt, wavelet.c_i)
        # sample wavelet and normalise
        norm = dt ** 0.5 / (4 * np.pi)
        wavelet_data = norm * wavelet(t, scale)  # this is an 1d array for sure!
        output[ind, :] = fftconvolve(data, wavelet_data[tuple(slices)], mode="same")

    return output


def _get_superlet_support(scale, dt, cycles):
    """
    Effective support for the convolution is here not only
    scale but also cycle dependent.
    """

    # number of points needed to capture wavelet
    M = 10 * scale * cycles / dt
    # times to use, centred at zero
    t = np.arange((-M + 1) / 2.0, (M + 1) / 2.0) * dt

    return t


def compute_adaptive_order(freq, order_min, order_max):
    """
    Computes the superlet order for a given frequency of interest
    in the fractional adaptive SLT (FASLT) according to
    equation 7 of Moca et al. 2021.

    This is a simple linear mapping between the minimal
    and maximal order onto the respective minimal and maximal
    frequencies.
    Note that `freq` should be ordered low to high.
    """

    f_min, f_max = freq[0], freq[-1]

    assert f_min < f_max

    order = (order_max - order_min) * (freq - f_min) / (f_max - f_min)

    # return np.int32(order_min + np.rint(order))
    return order_min + order


# ---------------------------------------------------------
# Some test data akin to figure 3 of the source publication
# ---------------------------------------------------------


def gen_superlet_testdata(freqs=None, cycles=11, fs=1000, eps=0):
    """
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 3 in Moca et al. 2021 NatComm
    """

    if freqs is None:
        freqs = [20, 40, 60]

    signal = []
    pad = []
    for freq in freqs:
        # 10 cycles of f1
        tvec = np.arange(cycles / freq, step=1 / fs)

        harmonic = np.cos(2 * np.pi * freq * tvec)
        f_neighbor = np.cos(2 * np.pi * (freq + 10) * tvec)
        packet = harmonic + f_neighbor

        # 2 cycles time neighbor
        delta_t = np.zeros(int(2 / freq * fs))

        # 5 cycles break
        pad = np.zeros(int(5 / freq * fs))

        signal.extend([pad, packet, delta_t, harmonic])

    # stack the packets together with some padding
    signal.append(pad)
    signal = np.concatenate(signal)

    # additive white noise
    if eps > 0:
        signal = np.random.randn(len(signal)) * eps + signal

    return signal


def plot_ground_truth_with_score(sim_nr):
    pca = PCA(n_components=3)
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    signal_pca = pca.fit_transform(spikes)
    score_davies1 = sklearn.metrics.davies_bouldin_score(signal_pca, labels)
    score_silhouette = sklearn.metrics.silhouette_score(signal_pca, labels)
    score_calinski = sklearn.metrics.calinski_harabasz_score(signal_pca, labels)
    scatter_plot.plot(title="GT with PCA Sim_%d" % sim_nr, X=signal_pca, labels=labels, marker='o')
    plt.show()
    print("Score Davies-Bouldin: ", score_davies1)
    print("Score Calinsky-Harabasz: ", score_calinski)
    print("Score Silhouette: ", score_silhouette)


def plot_data_with_score(simulation, n_components, ord, ncyc):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation, align_to_peak=False)
    # def slt(spikes, ord, ncyc, derivatives=True)
    slt_features = slt.slt(spikes, ord, ncyc)

    print(slt_features)
    clusters = max(labels)
    scaler = StandardScaler()
    features = scaler.fit_transform(slt_features)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    scatter_plot.plot("Sim_%d (%d clusters)_%dD" % (simulation, clusters, n_components), features_pca, labels,
                      marker='o')
    plt.show()
    score_davies = sklearn.metrics.davies_bouldin_score(slt_features, labels)
    score_calinski = sklearn.metrics.calinski_harabasz_score(slt_features, labels)
    score_silhouette = sklearn.metrics.silhouette_score(slt_features, labels)
    print("Score Davies-Bouldin: ", score_davies)
    print("Score Calinsky-Harabasz: ", score_calinski)
    print("Score Silhouette: ", score_silhouette)


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


def spikes_per_cluster(simNr, label):
    spikes, labels = ds.get_dataset_simulation(simNr=simNr, align_to_peak=False)
    selected_spikes = spikes[labels == label]
    average_spike = np.mean(selected_spikes, axis=0)
    return average_spike


def fourier_magnitude(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:100]]
    Y = [x.imag for x in fft_signal[:, 0:100]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    return amplitude


def fourier_phase(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:100]]
    Y = [x.imag for x in fft_signal[:, 0:100]]
    phase = np.arctan2(Y, X)
    return phase


def fourier_power(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:100]]
    Y = [x.imag for x in fft_signal[:, 0:100]]
    magnitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    power = magnitude * magnitude
    return power


def plot_spectrum(spikes, label, order_max, order_min, c, mode=0):
    ampls = []
    fs = 1000  # sampling frequency
    signal = spikes

    # frequencies of interest in Hz
    foi = np.linspace(1, 250)
    scales = scale_from_period(1 / foi)

    spec = superlet(
        signal,
        samplerate=fs,
        scales=scales,
        order_max=order_max,
        order_min=order_min,
        c_1=c,
        adaptive=True,
    )

    # amplitude scalogram
    if mode == 0:
        ampls = np.abs(spec)
    if mode == 1:
        ampls = fourier_magnitude(spec)
    if mode == 2:
        ampls = fourier_magnitude(spec)
    if mode == 3:
        ampls = fourier_magnitude(spec)

    fig, (ax1, ax2) = ppl.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))

    ax1.plot(np.arange(signal.size) / fs, signal, c=LABEL_COLOR_MAP[label])
    ax1.set_ylabel('average spike')
    ax1.set_facecolor("gray")

    extent = [0, len(signal) / fs, foi[0], foi[-1]]
    im = ax2.imshow(ampls, aspect="auto", extent=extent, origin='lower')

    ppl.colorbar(im, ax=ax2, orientation='horizontal',
                 shrink=0.7, pad=0.2, label='amplitude')

    ax2.plot([0, len(signal) / fs], [20, 20], "--", c='0.5')
    ax2.plot([0, len(signal) / fs], [40, 40], "--", c='0.5')
    ax2.plot([0, len(signal) / fs], [60, 60], "--", c='0.5')

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")

    fig.tight_layout()
    plt.show()


def plot_spectrum_for_clusters(simNr, order_max, order_min, c):
    _, labels = ds.get_dataset_simulation(simNr, align_to_peak=False)
    for label in np.unique(labels):
        plot_spectrum(spikes_per_cluster(4, label), label, order_max, order_min, c)

# ---------------------------- TESTING ------------------------------------------

# slt_1spike(spike, spike_pos, label, ord, ncyc, freq_start, freq_end)
# spikes, labels = ds.get_dataset_simulation(simNr=4, align_to_peak=False)
# print(len(np.unique(labels)))
# print(len(spikes_per_cluster(8,0)))
# slt.slt_1spike(spikes[0], 0, 0, 5, 1.5, 100, 500)

# plot_spectrum_for_clusters(4, order_max=5, order_min=1, c=1.5)

# plot_data_with_score(simulation=4, n_components=3, ord=2, ncyc=1.5)
# plot_ground_truth_with_score(sim_nr=4)
