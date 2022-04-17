from re import S
import numpy as np
from scipy import signal
from fatiguepy import *
from rfcnt import rfcnt
from math import ceil
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

import pkg_resources
pkg_resources.require(['matplotlib>=3.4'])  # Need "stairs" function

fs      = 1024       # Sample rate in [-]
NFFT    = 1024       # Number of spectral bins
df      = fs / NFFT  # Spectral density in [Hz]
dt      = 1 / fs     # Time step in [s]
#xf      = 10000        # Signal duration in [s]
nbins   = 100        # Number of bins for rainflow counting

# Basquin Formula
# N = C * La**-k
# with sigma = sigmaf * (2N)**b
sigmaf  = 1020             # Stress coefficient in [MPa]
b       = -0.2             # Stress exponent in [-]
k       = -1/b             # "Wöhler" exponent (positive)
A       = (2**b)*sigmaf    # Stress coefficient for one cycle (2N) in [MPa]
C       = A ** -(1/b)      # Axis intercept for cycles on sigma==1 in [cycles]



def set_xf(xf_new):
    global xf
    xf = xf_new


class RainflowTest(Rainflow.rainflowD):
    def __init__(self, *args, class_width=None, class_offset=None, auto_resize=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not class_width:
            rng = np.ptp(self.y)
            if not class_width:
                class_width = rng / self.nbins
            if not class_offset:
                class_offset = np.min(self.y) - class_width / 2
                
        self.class_width = class_width
        self.class_offset = class_offset
        self.auto_resize = auto_resize

        nd = 1e7
        sd = self.A * nd**(-1/self.k)  # (s/sd)**-k = n/nd
        self.wl = dict(nd=nd, sd=sd, k=-1/self.b)
                
    def rainflow_histogram(self, residual_method=7):
        "Get rp counts with class boundary at 0 and residue weighted as 'repeated'"
        # residual_method = 7  # "Repeated"
        res = rfcnt.rfc(self.y, 
                        class_count=self.nbins, 
                        class_offset=self.class_offset, 
                        class_width=self.class_width,
                        hysteresis=0, 
                        auto_resize=self.auto_resize,
                        wl=self.wl,
                        residual_method=residual_method)
        
        r = res["rp"][1:,0]
        n = res["rp"][1:,1]
            
        self.D = res["damage"]

        # Calculate damage with 2 different parameter sets
        D_wl = D_bq = 0
        sigmaf = self.A / (2**self.b)
        for i in range(len(r)):
            # Wöhler
            Si = r[i] / 2
            Nf = self.wl["nd"] * (Si / self.wl["sd"])**-self.k   # (s/sd)**-k = n/nd
            D_wl += n[i] / Nf
            # Basquin
            Nf = 0.5 * (Si / self.sigmaf)**(1/self.b)
            #  = 0.5 * (Si*(2**b)/A)**(1/b)
            #  = 0.5 * (Si/A)**(1/b) * (2**b)**(1/b)
            #  = (Si/A)**(1/b)
            D_bq += n[i] / Nf

        # Should give the same results
        assert np.isclose(D_wl, D_bq)
        assert np.isclose(D_wl, self.D)
        
        # Rest is taken from fatiguepy.rainflowD.rainflow_histogram()...
        S = (r/2)/(1-np.mean(self.y)/self.sigmaf)
        rangemax = max(S) - min(S)
        nclass = self.nbins
        tns = sum(n)
        bw = (min(S) + (rangemax/nclass)*(nclass+1)) - (min(S) + (rangemax/nclass)*(nclass))
        #  = min(S) + (rangemax/nclass)*(nclass+1) - min(S) - (rangemax/nclass)*(nclass)
        #  = (rangemax/nclass) * (nclass+1-nclass)
        #  = rangemax/nclass
        p = []
        appendp = p.append
        summ = 0
        for i in range(len(n)):
            appendp((n[i]/tns)/bw)
            summ += p[i]*bw
            
        return S, n, p
    
    def Damage(self):
        D = super().Damage() * max(self.x)
        assert np.isclose(D, self.D)
        return D


def get_psd_target():
    "Return the PSD target"
    f = np.array([5, 10, 60, 210], dtype="f8")
    Gyy = np.array([0.05, 0.06, 0.06, 0.0008])
    
    return f, Gyy


def create_noise(which="psd", blend_method="cos"):
    t = np.arange(0, xf, dt)
    if which == "band":
        # Calculate random noise from 50 to 220 Hz
        y = np.random.randn(len(t))
        nyquist = fs / 2  # Nyquist frequency

        left_pass  = 1.1*50/nyquist
        left_stop  = 0.9*50/nyquist
        right_pass = 0.9*220/nyquist
        right_stop = 1.1*220/nyquist

        (N, Wn) = signal.buttord(wp=[left_pass, right_pass],
                                 ws=[left_stop, right_stop],
                                 gpass=2, gstop=30, analog=0)

        (b, a) = signal.butter(N, Wn, btype='band', analog=0, output='ba')

        y = signal.filtfilt(b, a, y)
        class_width = y.ptp() / 100
        stats = dict(psdnoise=None)
    elif which == "psd":
        try:
            import pyftc
        except ModuleNotFoundError:
            y = stats = None
        else:
            # Generate random noise from PSD shape with its just-in-time statistics
            f, Gyy = get_psd_target()
            _, y, stats = pyftc.psdnoise(psdFreq=f, psdMagn=Gyy, fs=fs, nfft=NFFT, nOut=len(t),
                                         minFreq=np.min(f), maxFreq=np.max(f), windowing=True)
            class_width = stats["class_width"]
            stats = dict(psdnoise=stats)
    else:
        assert False
        
    if not blend_method:
        pass
    elif blend_method == "zero":
        y -= np.mean(y[1:-1])
        y[[0, -1]] = 0
    elif blend_method == "cos":
        ramp_duration = 0.5;
        ramp_t = np.arange(0, ramp_duration, dt)
        ramp_cos = np.cos(2 * np.pi * ramp_t / ramp_duration / 2) * -0.5 + 0.5
        assert 2 * len(ramp_cos) <= len(y)
        y[:len(ramp_cos)] *= ramp_cos
        y[-len(ramp_cos):] *= ramp_cos[::-1]
    else:
        assert False

    class_edges, n, ncum = get_rp(y, class_width)
    stats["signal"] = dict(class_edges=class_edges,
                           class_counts=n,
                           class_counts_cum=ncum)

    return t, y, stats


def get_rp(y, class_width):
    "Calculate range pair counting as should be done by psdnoise"
    rainflowTest = RainflowTest(C, k, y, x=1)  # For wl params only
    res = rfcnt.rfc(y, 
                    class_count=2, 
                    class_offset=0, 
                    class_width=class_width,
                    hysteresis=class_width, 
                    auto_resize=True,
                    wl=rainflowTest.wl,
                    residual_method=7)  # "repeated"

    n = res["rp"][:,1]
    ncum = np.cumsum(n[::-1])[::-1]
    class_edges = np.arange(len(n) + 1) * class_width / 2
    
    return class_edges, n, ncum


def check_rp_stats(y, stats):
    """
    Test if rp counts match with results from psdnoise.
    A closed cycle from class 4 to 6 have an effective range of 2 (6-4) and an amplitude of 1.
    At least that range could have been between 1 and 3!
    """
    if stats:
        # Check RP (Amplitudes)
        fig = plt.figure("Check range pair count results (on-the-fly vs. post-process)")
        if "psdnoise" in stats:
            class_count = stats["psdnoise"]["class_count"]
            class_width = stats["psdnoise"]["class_width"]
            class_edges = np.arange(class_count + 1) * class_width / 2
            class_counts = stats["psdnoise"]["bpz"]
            class_counts = np.insert(class_counts, 0, class_counts[0])
            class_edges = np.insert(class_edges, 0, -np.inf)
            plt.stairs(class_counts, class_edges, lw=4, ls="--", orientation="horizontal", label="psdnoise (jit-stats))")
        # Add rainflow counting results
        if "signal" in stats:
            class_edges = stats["signal"]["class_edges"]
            class_counts_cum = stats["signal"]["class_counts_cum"]
            class_counts_cum = np.insert(class_counts_cum, 0, class_counts_cum[0])
            class_edges = np.insert(class_edges, 0, -np.inf)
            plt.stairs(class_counts_cum, class_edges, lw=2, ls="-", orientation="horizontal", label="rfcnt")
        plt.xscale("log")
        #plt.yscale("log")
        plt.ylim([0.0, None])
        plt.grid(True, which="both")
        plt.legend()

        
def get_spectrum(method="target", y=None, show_check=False):
    if method == "welch":
        window = signal.hann(NFFT, True)
        f, Gyy = signal.welch(y, fs, return_onesided=True, window=window, average='median')
        Gyy = np.abs(Gyy)
        f_welch, Gyy_welch = f, Gyy
    elif method == "pyftc":
        try:
            import pyftc
        except ModuleNotFoundError:
            f = Gyy = None
        else:
            Gyy, f, _ = pyftc.spectrum(y, fs, spec_type="PSD", wnd_type="hann")
            Gyy = Gyy.ravel()
        f_pyftc, Gyy_pyftc = f, Gyy
    elif method == "target":
        f, Gyy = get_psd_target()
    else:
        assert False
        
    if show_check:
        if not "f_welch" in locals():
            f_welch, Gyy_welch = get_spectrum("welch", y)
        if not "f_pyftc" in locals():
            f_pyftc, Gyy_pyftc = get_spectrum("pyftc", y)
        if not "f_target" in locals():
            f_target, Gyy_target = get_spectrum("target")
        # Check PSD
        fig = plt.figure("Comparison of given noise PSD to expectation")
        plt.plot(f_welch, Gyy_welch, label="welch")
        plt.plot(f_pyftc, Gyy_pyftc, label="pyftc")
        plt.plot(f_target, Gyy_target, label="target")
        plt.yscale("log")
        plt.ylim([1e-4, 1e-1])
        plt.xlim([0, 250])
        plt.grid(True, which="both")
        plt.legend()

    return f, Gyy


def model_results(model, y, f, Gyy, s):
    if model == "NB":
        return Narrow_Band.NB(k, C, Gyy, f, xf, s)
    elif model == "RC":
        return Rice.RC(k, C, Gyy, f, xf, s)
    elif model == "WL":
        return Wirsching_Light.WL(k, C, Gyy, f, xf, s)
    elif model == "TB":
        return Tovo_Benasciutti.TB(k, C, Gyy, f, xf, s)
    elif model == "AL":
        return alpha075.AL(k, C, Gyy, f, xf, s)
    elif model == "DK":
        return Dirlik.DK(k, C, Gyy, f, xf, s)
    elif model == "ZB":
        return Zhao_Baker.ZB(k, C, Gyy, f, xf, s)
    elif model == "RF":
        if xf > 1000:
            return None
        #t = np.arange(len(y)) * dt
        t = [len(y) * dt]  # Only maximum needed by "Rainflow"
        return Rainflow.rainflowD(C, k, y, t, nbins=70)
    elif model == "RFC":
        t = [len(y) * dt]  # Only maximum needed by "Rainflow"
        return RainflowTest(C, k, y, t, nbins=70)


def model_get_peak(f, Gyy, start=100, step=1.05, density=64):
    sigma = start
    itermax = 1000
    f_ = np.arange(max(f) + 1)
    Gyy = np.interp(f_, f, Gyy, 0, 0)
    for j in range(itermax):
        s = np.linspace(0, sigma, density)
        model = Dirlik.DK(k, C, Gyy, f_, xf, s)
        n = model.counting_cycles()
        for jj in range(itermax):
            i = np.diff(n)
            if np.any(np.argwhere(i > 0)):
                n = np.delete(n, np.argwhere(i > 0))
            else:
                i = np.digitize(1, n)
                break
        if i == len(n):
            sigma *= step
        else:
            sigma = s[i]
            if n[i-1] < 1.1:
                return sigma

    return None



def main():
    global xf
    xf = 1000
    f, Gyy = get_psd_target()
    sigma_max = model_get_peak(f, Gyy)
    t, y, stats = create_noise(which="psd", blend_method="cos")
    check_rp_stats(y, stats)
    f, Gyy = get_spectrum("welch", y, show_check=True)

    moments = prob_moment.Probability_Moment(Gyy, f)

    m0 = moments.momentn(0)
    m1 = moments.momentn(1)
    m2 = moments.momentn(2)
    m4 = moments.momentn(4)
    m75 = moments.momentn(0.75)
    m15 = moments.momentn(1.5)

    E0 = moments.E0()
    EP = moments.EP()
    gamma = moments.alphan(2)
    
    si = 0.0
    sf = y.ptp()  # Peak to peak value
    sf = sigma_max  # Use heuristic instead
    ds = sf / 128   # 128 classes
    s = np.arange(si, sf, ds)


    models = dict()
    for model in ["NB", "RC", "WL", "TB", "AL", "DK", "ZB", "RF", "RFC"]:
        res = model_results(model, y, f, Gyy, s)
        if not res:
            models[model] = None
            continue
        models[model] = dict()
        models[model]["life"] = res.Life()
        models[model]["lifes"] = res.Lifes()
        models[model]["S"] = s
        if model in ("RF", "RFC"):
            S, nRF, pRF = res.rainflow_histogram()
            models[model]["hist"] = dict(S= S, n = nRF, p = pRF)
            models[model]["pdf"] = pRF
            models[model]["damage"] = res.Damage()
            models[model]["counts"] = nRF
            models[model]["S"] = S
        elif model == "WL":
            models[model]["pdf"] = None
            models[model]["damage"] = None
            models[model]["counts"] = None
        else:
            models[model]["pdf"] = res.PDF()
            models[model]["damage"] = res.Damage()
            models[model]["counts"] = res.counting_cycles()

    def plot_counts(ds, fmt=None, *, cumulated=False, turned=False, **kwargs):
        s = ds["S"]
        n = ds["counts"]
        if cumulated:
            n = np.cumsum(n[::-1])[::-1]
        i = np.argwhere(n > 1).item(-1)
        s = s[:i+2]
        n = n[:i+2]
        if turned:
            args = [n, s]
        else:
            args = [s, n]

        if fmt:
            args.append(fmt)

        plt.plot(*args, **kwargs)

    if 1:
        plt.figure("Comparison in same plot")
        #plt.title(rf"$\alpha_{2} = {round(gammanum, 2)}$")
        if models["RF"]:
            plt.stem(models["RF"]["S"], models["RF"]["counts"], linefmt="C0-", markerfmt="o", label="fatpack")
        plt.stem(models["RFC"]["S"], models["RFC"]["counts"], linefmt="C1-", markerfmt="x", label="rfcnt")
        plot_counts(models["NB"], linestyle=':', label="Rayleigh")
        plot_counts(models["AL"], linestyle='--', label = r"$\alpha_{0.75}$")
        plot_counts(models["TB"], linestyle='-.', label="Tovo-Benasciutti")
        plot_counts(models["DK"], linestyle='-', label="Dirlik")
        plot_counts(models["ZB"], linestyle='--', label="Zhao-Baker")

        plt.legend()

        plt.xlabel(r'$S_{a}$ [MPa]')
        plt.ylabel(r'n [cycles]')
        plt.grid(True)

    if 1:
        plt.figure("Comparison of cumulative results in same plot")
        if models["RF"]:
            plot_counts(models["RF"], 'k-', cumulated=True, turned=True, label="fatpack")
        plot_counts(models["RFC"], cumulated=True, turned=True, label="rfcnt")
        plot_counts(models["NB"], cumulated=True, turned=True, label="Rayleigh")
        plot_counts(models["AL"], cumulated=True, turned=True, label=r"$\alpha_{0.75}$")
        plot_counts(models["TB"], cumulated=True, turned=True, label="Tovo-Benasciutti")
        plot_counts(models["DK"], cumulated=True, turned=True, label="Dirlik")
        plot_counts(models["ZB"], cumulated=True, turned=True, label="Zhao-Baker")

        plt.legend()

        plt.xlabel(r'N (log) [cycles]')
        plt.ylabel(r'$S_{a}$ [MPa]')
        plt.grid(True)
        plt.xscale('log')
        plt.xlim([1, None])
        
    plt.show()


if __name__ == "__main__":
    main()