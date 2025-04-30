'''
Creates a light curve object with operations to work on it
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy import units as u
from astropy import timeseries
from pdb import set_trace

class LC:
    def __init__(self, t, y, yerr=None):
        self.t = t
        self.y = y
        self.yerr = yerr

    def correlated_noise(self, maxbin, interval=50, plots=False, \
        verbose=False):
        '''
        Compute correlated noise level for a given time bin, following
        Pont+2006.

        t and maxbin must in the same units.
        '''

        if verbose:
            print('Computing red noise curve...')

        resolution = abs(np.diff(self.t)).min()
        l_win = int(maxbin/resolution)
        V_n, white = [], []
        nbins =  np.arange(1, l_win + 1, interval)
        for n in nbins:
            F_j = signal.medfilt(self.y, kernel_size=n)
            V_n.append(np.std(F_j))
            # White noise
            try:
                white.append(np.std(self.y)/np.sqrt(n))
            except FloatingPointError:
                white.append(np.sqrt(np.sum((self.y - np.mean(self.y))**2) \
                    /len(self.y))/np.sqrt(n))

        if plots:
            def bin2dt(x):
                return x*np.diff(self.t).min()*24.

            def dt2bin(x):
                return x/np.diff(self.t).min()*24.

            fig, ax = plt.subplots()
            ax.plot(nbins, np.array(V_n)*1e6, label='red')
            ax.plot(nbins, np.array(white)*1e6, label='white')
            ax.set_xlabel('Bin size [data points]', fontsize=14)
            ax.set_ylabel('Noise level [ppm]', fontsize=14)
            secax = ax.secondary_xaxis('top', functions=(bin2dt, dt2bin))
            secax.set_xlabel('Time [hours]', fontsize=14)
            plt.legend()
            #plt.show()
            #set_trace()

        return np.array(nbins), np.array(V_n), np.array(white)

    def rebin(self, binning_factor):

        tnew = self.t[::binning_factor]

        yerrtot_binned = np.zeros(len(tnew))
        ytot_binned = np.zeros(len(tnew))
        for i in np.arange(len(yerrtot_binned)):
            bin_i = i*binning_factor
            bin_fin = i*binning_factor + binning_factor
            if self.yerr is not None:
                weights = self.yerr[bin_i : bin_fin]**-2
            else:
                weights = np.ones(len(self.y))
            yerrtot_binned[i] += np.sqrt(1./np.sum(weights))
            ytot_binned[i] += np.average(self.y[bin_i : bin_fin], \
                        weights=weights)

        self.tbinned = np.copy(tnew)
        self.ybinned = np.copy(ytot_binned)
        self.yerrbinned = np.copy(yerrtot_binned)

        return

    def prewhitening(self, nurange=[1., 100.], npeaks=100, \
                upper_freq=3., plots=False, verbose=False, yscale='linear', \
                saveplot=None, plot_harmonics=None):
        '''
        Remove sinusoidal signals from light curve.

        Parameters
        ----------
        The frequency range can be specified as interval (nurange) or as
        in multiples of the frequency at max power (upper_freq).
        npeakmin (int): used to keep exploring other peaks (used to target a
        specific frequency) - not used anymore

        '''

        yc = np.copy(self.y - np.median(self.y))
        t2 = (self.t*u.day).to(u.s).value
        freq, power = timeseries.LombScargle(t2, self.y).autopower( \
                    minimum_frequency=50e-6, maximum_frequency=1500e-6)
        P = 1./freq[power.argmax()]
        if upper_freq is not None:
            # Lowest and highest possible frequencies are set
            nurange[1] = min([freq[power.argmax()]*upper_freq*1e6, 277.])
        freq0, power0 = np.copy(freq), np.copy(power)
        flag = np.logical_and(freq*1e6 >= nurange[0], freq*1e6 <= nurange[1])
        freq *= u.Hz

        # Save LS-based model
        y_model = []

        for j in np.arange(npeaks):
            nu0 = freq[flag][power[flag].argmax()]
            ls = timeseries.LombScargle(t2, yc)
            xn = ls.model(t2, nu0.value)
            yc -= xn
            y_model.append(xn)
            freq, power = timeseries.LombScargle(t2, yc).autopower( \
                    minimum_frequency=50e-6, maximum_frequency=2000e-6)
            #freq, power = timeseries.LombScargle(t2, yc).autopower( \
            #        minimum_frequency=3./np.ptp(t2)*1e-6, maximum_frequency=2000e-6)
            flag = np.logical_and(freq*1e6 >= nurange[0], freq*1e6 <= nurange[1])
            noiselev = np.std(power[freq*1e6 > 1000.])
            P_nu0 = power[flag].max()
            snr = P_nu0 / noiselev
            if verbose:
                print('Iteration:', j, ' SNR:', snr)
            if snr <= 4.:# and j > npeakmin:
                break
            freq *= u.Hz

        lowpass = np.sum(y_model, axis=0)

        if plots:
            plt.figure()
            plt.plot(t2/60., self.y - np.median(self.y), '.', \
                                            label='Raw - median')
            plt.plot(t2/60., lowpass, linewidth=3, label='Low-pass')
            plt.plot(t2/60., yc, '.', label='Prewhitened')
            plt.xlabel('Time [min]', fontsize=14)
            plt.ylabel('Flux [units?]', fontsize=14)
            plt.legend()
            plt.tight_layout()
            if saveplot is not None:
                plt.savefig(saveplot + '_lc.pdf')

            fig, ax = plt.subplots()
            ax.plot(freq0*1e6, power0, label='Original')
            ax.plot(freq*1e6, power, label='Prewhitened')
            if plot_harmonics is not None:
                for i in range(1, 6):
                    ax.plot([plot_harmonics*1e6*i, plot_harmonics*1e6*i], \
                            [power.min(), power.max()], 'k--', alpha=0.3)
            ax.set_xscale('log')
            if yscale == 'log':
                ax.set_yscale('log')
            ax.set_xlabel(r'Frequency [$\mu$Hz]', fontsize=14)
            ax.set_ylabel('LS power', fontsize=14)
            ax.set_ylim(power0.min()/100., power0.max()*10.)
            plt.legend()
            plt.tight_layout()
            if saveplot is not None:
                plt.savefig(saveplot + '_ls.pdf')
            plt.close('all')

        return yc, P, lowpass + np.median(self.y)
