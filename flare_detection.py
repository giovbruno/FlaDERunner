import os
from scipy import signal, optimize
from scipy.interpolate import splrep, BSpline, CubicSpline, interp1d
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy import timeseries, constants
from hurst import compute_Hc
import numpy as np
import copy
import lc_class
import peakutils.peak as peakut
import smooth
import celerite2
import gp_utilities
from lmfit import Parameters, fit_report, minimize
import flare_class
import pickle
import models
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pdb import set_trace

def find_flares(lcfile, flatten=True, \
        tstar=None, rstar=None, wth=None, fth=None, peaki=[], \
        plot_flat=False, flare_threshold=5., outlier_thresh=8., \
        fit_continuum=0, clip=True, normalise=True, \
        complexity=5, saveplots='', min_datapoints=3, rebin=0,
        filt_kernel_size=11, plot_clip=False, verbose=True):
    '''
    Parameters
    ----------
    peaki: list of peaks can be provided

    rednoise (list of 3 elements): it can be provided here, or is calculated
                                   during smoothing (if smoothmode is not None)
    filt_kernel_size: kernel size for the median filter to be used in the flare
              edges definition

    Data needs to be normalised to 1.
    '''

    if type(lcfile) == list:
        t, y = lcfile[0], lcfile[1]
        if np.shape(lcfile)[0] == 3:
            yerr = lcfile[2]
        else:
            yerr = np.std(y)
        header = None
    elif type(lcfile) == str and \
                (lcfile.endswith('.dat') or lcfile.endswith('.txt')):
        print('\nLC file:', lcfile.split(lcfolder)[1], '\n')
        t, y = np.loadtxt(lcfile, unpack=True)
        yerr = np.std(y)
        header = None
    elif type(lcfile) == str and lcfile.endswith('.fits'):
        try:
            lc = fits.open(lcfile, ignore_missing_simple=True)
        except OSError:
            print('OSError: Empty or corrupt FITS file')
            return [], []

        # For a TESS-like fits file
        header = lc[0].header
        try:
            if header['TEFF'] is not None:
                tstar = header['TEFF']*u.K
            else:
                tstar = None
        except KeyError:
            tstar = None
        try:
            if header['RADIUS'] is not None:
                rstar = header['RADIUS']*constants.R_sun
            else:
                rstar = None
        except KeyError:
            rstar = None
        try:
            t = lc[1].data['TIME']
            y = lc[1].data['PDCSAP_FLUX']
            yerr = lc[1].data['PDCSAP_FLUX_ERR']
        except TypeError: # Problem with the dataset
            return [], []
        except AttributeError:
            return [], []

    # Remove all NaNs
    flag = np.isnan(y)
    t, y, yerr = t[~flag], y[~flag], yerr[~flag]

    if np.diff(t).min() < 0.:
        print('Possibly damaged file')
        return [], []

    if rebin > 1:
        light_curve = lc_class.LC(t, y, yerr=yerr)
        t, y, yerr = light_curve.rebin(rebin)

    # Normalise LC
    if normalise:
        yerr /= abs(np.median(y))
        y /= abs(np.median(y))

    # Remove stellar variability - this is to fit the flare profiles
    if flatten:
        yf, yferr, ls_quiet, rednoise, header, tck = flatten_LC( \
                t, y, yerr, plots=False, mode='smooth', \
                header=header, savefile='', compute_rednoise=True)
    else:
        yf, yferr = np.zeros(len(t)), np.zeros(len(t)) + np.median(yerr)
        freq, power = [], []
        ls_quiet = np.zeros(len(y))
    yflat = y - yf

    # Remove outliers
    tini = np.copy(t)
    if clip:
        if plot_clip:
            plt.close('all')
            plt.plot(t, yflat)
        yflatfilt = signal.medfilt(yflat, kernel_size=3)
        mask = sigma_clip(yflat - yflatfilt, sigma=outlier_thresh)
        t = t[~mask.mask]
        yflat = yflat[~mask.mask]
        yerr = yerr[~mask.mask]
        tini = tini[~mask.mask]
        yf = yf[~mask.mask]
        y = y[~mask.mask]
        if plot_clip:
            plt.plot(t, yflat)
            plt.show()
            set_trace()

    # This is to find the flare peaks
    noiselev = np.median(yferr)
    if noiselev == 0:
        print('Problem with data error bar estimation: keeping original ' \
                + 'error bars')
        noiselev = np.median(yerr)

    if peaki == []:
        peaki = list(peakut.indexes(yflat, thres=flare_threshold*noiselev, \
                        min_dist=10, thres_abs=True))
    print('Detected peaks #1:', len(peaki))

    # If no flares are found, there's no point in repeating these steps
    if flatten and len(peaki) > 0:
        # Two rounds of fitflares: the first one will give flare intervals,
        # to be used to smooth the LC again. Then, second round of fitflares.
        # This result is kept if the scatter on the residuals is improved
        flare_ranges, flarespar = flare_analysis(t, yflat, yerr, peaki, \
                noiselev, ls_quiet, traw=np.copy(t), yraw=np.copy(y), \
                threshold=flare_threshold, \
                rednoise=rednoise, min_datapoints=min_datapoints, \
                filt_kernel_size=filt_kernel_size, validate=False)

        yf, yferr, ls_quiet, _, header, tck = flatten_LC( \
                t, y, yerr, plots=True, mode='prewhitening', \
                compute_rednoise=False, header=header, savefile='', \
                flare_ranges=flare_ranges, upper_freq=3.)

        yflat = y - yf
        noiselev = np.median(yferr)

        # Update peak detection
        peaki = list(peakut.indexes(yflat, thres=flare_threshold*noiselev, \
                        min_dist=10, thres_abs=True))
        print('Detected peaks #2:', len(peaki))

    # Plot flattended LC
    if plot_flat:
        fig, axs = plt.subplots(figsize=(13, 9), nrows=2)
        axs[0].plot(tini, y, '.', alpha=0.5, label='Raw flux')
        axs[0].plot(tini, yf, label='Quiet flux model')
        axs[0].plot(tini, yf + flare_threshold*noiselev, '--', \
                    label='Detection threshold')
        axs[0].set_title('Scatter S/N: {:.1f}'.format(header['scatter_SN']))
        for pp in peaki:
            axs[0].plot([t[pp], t[pp]], [y[pp] + 5*yerr.max(), \
                    y[pp] + 10*yerr.max()], 'c', linewidth=2)
        axs[0].legend()
        axs[0].set_xlabel('Time [days]', fontsize=14)
        axs[0].set_ylabel('Normalised flux', fontsize=14)

        freq, power = ls_quiet.autopower(nyquist_factor=1.)
        axs[1].loglog(freq*1e3, power, 'k')
        axs[1].set_xlabel('Frequency [mHz]', fontsize=14)
        axs[1].set_ylabel('Power', fontsize=14)

        plt.savefig(saveplots.replace('.pdf','_LC.pdf').replace( \
                                                    '.fits', '_LC.pdf'))
        plt.show()
        set_trace()
        plt.close('all')

    flare_ranges, flarespar = flare_analysis(t, yflat, yerr, peaki, noiselev, \
            ls_quiet, traw=np.copy(t), yraw=y, threshold=flare_threshold, \
            validate=True, \
            rednoise=rednoise, tstar=tstar, rstar=rstar, tflare=9000.*u.K, \
            wth=wth, fth=fth, fit_continuum=fit_continuum,
            complexity=complexity, min_datapoints=min_datapoints, \
            plotname=saveplots.replace('.pdf','_flares.pdf').replace('.fits', \
                    '_flares.pdf'), filt_kernel_size=filt_kernel_size, \
            verbose=verbose)

    plt.close('all')
    # Add LC info
    flarespar.append(header)

    # Return an array where data points that belong to flares are flagged as 1
    flareflag = np.zeros(len(t))
    arrflag = [np.arange(fr[0], fr[1] + 1) for fr in flare_ranges]
    if len(arrflag) > 0:
        flareflag[np.hstack(arrflag)] += 1

    return flarespar, flareflag

def flare_analysis(t, yflat, yerr, peaki, noise_level, ls_quiet, \
            validate=False, threshold=4., complexity=5, tstar=None, \
            rstar=None, tflare=None, wth=None, fth=None, rednoise=None, \
            fit_continuum=0, plotname='', traw=None, yraw=None, \
            min_datapoints=3, filt_kernel_size=11, \
            save_flare_ranges=False, verbose=False):
    '''
    Iterative flare fit routine. No need for two filtered versions of the LC.
    Build a light curve model with all fitted flares.

    Parameters
    ----------
    time, fluxes and error bars for flattened light curve.
    peaki (list): flare peaks
    min_datapoints: minimum flare duration in data points
    filt_kernel_size: kernel size for the median filter to be used in the flare
              edges definition
    validate (bool): fit flare profiles

    Returns
    -------
    flarespar: dict with fitted flare parameters
    '''
    # This is used to determine flare length, by removing some noise
    filtflare = signal.medfilt(yflat, kernel_size=filt_kernel_size)

    flarespar = []
    # This is to check if this flare was already fitted on a multi-peak one
    if validate:
        pplot = PdfPages(plotname)

    nflares = 0
    tuntil = 0
    trawcopy = np.copy(traw)
    yrawcopy = np.copy(yraw)

    flare_ranges = []
    for i, peak in enumerate(peaki):
        if verbose:
            print('Flare #', i + 1, '/', len(peaki))

        # Get previous fitted peaks
        if peak <= tuntil:
            if verbose:
                print('Already fitted')
            continue

        if len(t) - peak < 10 or peak < 10:
            if verbose:
                print('Too close to dataset edge')
            continue

        noiselev = np.median(noise_level)

        j = -1
        flagleft = t < t[peak]
        while abs(j) < len(filtflare[flagleft]) - 1 \
                            and filtflare[flagleft][j] > noiselev:
            j -= 1
        tini = abs(t \
            - t[flagleft][abs(t[flagleft] - t[flagleft][j]).argmin()]).argmin()
        # This will be later used for the dip
        tbeg = np.copy(tini)

        j = 1
        flagright = t > t[peak]
        while j < len(filtflare[flagright]) - 1 \
                            and filtflare[flagright][j] > noiselev:
            j += 1
        tend = abs(t - t[flagright][abs(t[flagright] \
                    - t[flagright][j]).argmin()]).argmin()
        tfin = np.copy(tend)

        # Add points to help catch the continuum and dips
        # This happened with CHEOPS data!
        dd = np.diff(t)
        maxleft = int(1./24./dd[dd > 0].min())
        maxright = int(1./24./dd[dd > 0].min())
        indexleft = 0
        while tini - indexleft > 0 and indexleft < maxleft:
            indexleft += 1
            if filtflare[tini - indexleft] > noiselev:
                break

        indexright = 0
        while tend + indexright < len(t) - 1 and indexright < maxright:
            indexright += 1
            if filtflare[tend + indexright] > noiselev:
                break

        tini -= indexleft
        tend += indexright

        tuntil = np.copy(tend)
        tw = t[tini:tend] - t[peak]
        traw = trawcopy[tini:tend] - t[peak]
        yw = yflat[tini:tend]
        yraw = yrawcopy[tini:tend]
        yerrw = yerr[tini:tend]
        j = abs(tw).argmin()

        if len(tw) < min_datapoints:
            if verbose:
                print('Too short: Probably an outlier')
            continue

        # Reject features that are likely outliers or edge effects
        # If the end of the flare is not identified either, this might
        # be some sort of correlated noise
        myw = 0.
        if (yw[j + 1: j + min_datapoints] <= 2.*noiselev).any():
            if verbose:
                print('Too short: probably an outlier')
            continue

        # Check for data gaps
        if (np.diff(tw) > np.median(np.diff(t))*10).any():
            # Only consider the flux within gaps
            gaps = np.where(np.diff(tw) > np.median(np.diff(t))*10)[0]
            # initialize the closest indices as the first and second elements
            # of the array
            if len(gaps) > 1:
                # Go look into the first and last LC segment
                if gaps[0] > j:
                    tw = tw[:gaps[0]]
                    yw = yw[:gaps[0]]
                    traw = traw[:gaps[0]]
                    yraw = yraw[:gaps[0]]
                    yerrw = yerrw[:gaps[0]]
                elif j > gaps[-1]:
                    tw = tw[gaps[-1] + 1:]
                    yw = yw[gaps[-1] + 1:]
                    traw = traw[gaps[-1] + 1:]
                    yraw = yraw[gaps[-1] + 1:]
                    yerrw = yerrw[gaps[-1] + 1:]
                else:
                # loop through the array to find the closest indices
                    for i in range(len(gaps) - 1):
                        if gaps[i] <= j <= gaps[i + 1]:
                            idx1 = i
                            idx2 = i + 1
                            break
                    tw = tw[gaps[idx1] + 1 : gaps[idx2]]
                    yw = yw[gaps[idx1] + 1 : gaps[idx2]]
                    traw = traw[gaps[idx1] + 1 : gaps[idx2]]
                    yraw = yraw[gaps[idx1] + 1 : gaps[idx2]]
                    yerrw = yerrw[gaps[idx1] + 1 : gaps[idx2]]
            else:
                if gaps[0] > j:
                    idx1 = 0
                    idx2 = gaps[0]
                else:
                    idx1 = gaps[0] + 1
                    idx2 = -1
                tw = tw[idx1 : idx2]
                yw = yw[idx1 : idx2]
                traw = traw[idx1 : idx2]
                yraw = yraw[idx1 : idx2]
                yerrw = yerrw[idx1 : idx2]

        # Other check here
        if len(tw) < fit_continuum + min_datapoints:
            if verbose:
                print('Probably an outlier')
            continue

        flare_ranges.append([int(tbeg), int(tfin)])

        if validate:
            # This is necessary because of the data gaps
            aflare = flare_class.flare(tdata=tw, ydata=yw, yerrdata=yerrw, \
                    tbeg=t[tbeg] - t[peak], tend=t[tfin] - t[peak], tref=t[peak], \
                    Tstar=tstar, Rstar=rstar, Tflare=tflare, \
                    tpeak=t[peak], noise_level=noiselev, \
                    rednoise=rednoise, traw=traw, yraw=yraw, wth=wth, fth=fth)

            # First, the case with no flare: if this fit has the best BIC,
            # the flare is not validated
            aflare.fit_line(verbose=verbose)
            aflare.fit_flare_profile(complexity, threshold, verbose=verbose, \
                        fit_continuum=fit_continuum, plots=False)

            if aflare.npeaks > 0:
                nflares += 1
                # Test dip hypothesis
                result_dip, dip_delta_bic, dip_significance = aflare.fit_dip()
                aflare.result_dip = result_dip
                aflare.dip_delta_bic = dip_delta_bic
                aflare.dip_significance = dip_significance
                # Same for CME
                xmin = (t[tfin] - t[peak])
                if xmin == tw.max():
                    xmin -= abs(np.diff(t)[np.diff(t) > 0.]).min()
                xmax = tw.max()
                if xmin == xmax:
                    xmin -= abs(np.diff(t)[np.diff(t) > 0.]).min()
                result_cme, cme_delta_bic, cme_significance \
                                                = aflare.fit_dip(xmin, xmax)
                aflare.result_cme = result_cme
                aflare.cme_delta_bic = cme_delta_bic
                aflare.cme_significance = cme_significance

                aflare.plot_models(plot_instance=pplot, showplot=False)

                flarespar.append(aflare)

    if validate:
        pplot.close()
    if nflares == 0 and validate:
        print('No validated flares.')
        os.system('rm ' + plotname)
    elif nflares > 0 and validate:
        if nflares == 1:
            ll = 'flare'
        else:
            ll = 'flares'
        print(str(nflares) + ' validated ' + ll + '.')

    return flare_ranges, flarespar

def flatten_LC(t, f, ferr, plots=False, mode='smooth', compute_rednoise=True, \
            savefile='', header=None, flare_ranges=[], upper_freq=None):
    '''
    Find a smoothed verision of the LC, removing flares through iterative
    sigma-clipping

    Timewindow: use the number of data points to compute a time window (hours)
    or use it as it is.
    '''

    print('Flattening light curve...')

    if len(flare_ranges) > 0:
        arrflag = [np.arange(fr[0], fr[1] + 1) for fr in flare_ranges]
        arrflag = np.hstack(arrflag)
    else:
        arrflag = []

    if mode == 'smooth' or mode == 'gp':
        ls = timeseries.LombScargle(t*u.day, f)
        freq, power = ls.autopower(maximum_frequency=0.5/np.diff(t*u.day).min())
        Prot = 1./freq[power.argmax()]
        try:
            FAP = ls.false_alarm_probability(power.max()).value
        except FloatingPointError:
            FAP = -1.
        win = freq[power.argmax()]**-1*24./10./u.day # in hours
        smooth_factor = int(win*60*60./(np.median(np.diff(t))*86400.))

    if 'Prot' in locals():
        header['Prot_[days]'] = Prot.to(u.day).value
    if 'FAP' in locals():
        header['FAP'] = FAP

    # At most 1% of the points can be removed before smoothing. Start
    # with a 3-sigma rejection threshold, then increased if too many points
    # are rejected
    f_removed = 10.
    n_removed = 0
    sigma_threshold = 1.
    if mode == 'smooth':
        while f_removed > 0.1:
            if len(flare_ranges) == 0 :
                x = np.copy(t)
                y = np.copy(f)
                yerr = np.copy(ferr)
            # Use previously found flare ranges to refine flare removal.
            else:
                x = np.delete(t, arrflag)
                y = np.delete(f, arrflag)
                yerr = np.delete(ferr, arrflag)
            n_removed = 0
            nmask = 10
            while nmask > 0:
                y_model = smooth.smooth(y, window_len= \
                        min([int(len(y)/5.), smooth_factor]))[:len(x)]
                yc = sigma_clip(y - y_model, sigma=sigma_threshold)
                nmask = np.sum(yc.mask)
                x = x[~yc.mask]
                y = y[~yc.mask]
                yerr = yerr[~yc.mask]
                n_removed += np.sum(nmask)
            f_removed = 1. - (len(t) - n_removed)/len(t)
            sigma_threshold += 1

    if mode == 'spline':
        x = np.delete(t, arrflag)
        y = np.delete(f, arrflag)
        yerr = np.delete(ferr, arrflag)
        tck = splrep(x, y,  w=1./(2.*yerr), s=len(x) - (2.*len(x))**0.5)
        y_model_int = BSpline(*tck)(t)
        y_model = BSpline(*tck)(x)

    elif mode == 'prewhitening':
        x = np.delete(t, arrflag)
        y = np.delete(f, arrflag)
        yerr = np.delete(ferr, arrflag)
        lc = lc_class.LC(x, y, yerr=yerr)
        y_prewhitened, _, y_model_int = lc.prewhitening(npeaks=30, \
                upper_freq=upper_freq, plots=False, verbose=False)
        y_model = np.copy(y_model_int)

    elif mode == 'gp':
        if len(flare_ranges) == 0:
            x = np.copy(t)
            y = np.copy(f)
            yerr = np.copy(ferr)
        # Use previously found flare ranges to refine flare removal.
        else:
            x = np.delete(t, arrflag)
            y = np.delete(f, arrflag)
            yerr = np.delete(ferr, arrflag)
        # Celerite GP
        term1 = celerite2.terms.RotationTerm(sigma=np.std(f), \
                    period=header['Prot_[days]'], Q0=0.1, dQ=10., f=0.5)
        gp = celerite2.GaussianProcess(term1, mean=np.median(f))
        gp.compute(x, yerr=yerr)
        initial_params = [np.std(f), header['Prot_[days]'], \
                        0.1, 10., 0.5, np.median(ferr)]
        stdf = np.std(f)
        bounds = [(stdf, 3.*stdf), \
                (header['Prot_[days]'] - 1., header['Prot_[days]'] + 1.), \
                (0.001, 1.), (1., 2.), (0., 1.), (0., 3.*stdf)]
        soln = optimize.minimize(gp_utilities.neg_log_like, initial_params, \
                    bounds=bounds, method="L-BFGS-B", args=(gp, x, y, yerr))
        opt_gp = gp_utilities.set_params(soln.x, gp, x, yerr)
        y_model_int = opt_gp.predict(y, t, return_var=False)
        y_model = opt_gp.predict(y, x, return_var=False)
        #scatter = var**0.5

    # Interpolate smoothing
    if mode == 'smooth' or mode == 'prewhitening':
        tck = CubicSpline(x, y_model)
        #tck = interp1d(x, y_model, fill_value='extrapolate', bounds_error=None)
        y_model_int = tck(t)

    scatter = np.std(y - y_model)
    scatter_sn = scatter/np.median(yerr)

    if plots:
        plt.plot(t, f, 'b', alpha=0.2)
        plt.plot(x, y, 'b')
        plt.plot(t, y_model_int, 'orange', linewidth=3, \
                    label='Smooth version after interpolation/prediction')
        plt.xlabel('Time [days]', fontsize=14)
        plt.ylabel('Normalised flux', fontsize=14)
        plt.legend()
        plt.show()
        set_trace()
        plt.close('all')

    # Get PSD of bit of flare-free LC (not normalized)
    lcsec = (x - x.min())*u.day.to(u.s)
    ls = timeseries.LombScargle(lcsec, y, normalization='standard', \
        fit_mean=True)

    # Evaluate red noise level on the residuals
    if compute_rednoise:
        lightc = lc_class.LC(x, y - y_model)
        bins, red, white = lightc.correlated_noise(3600./86400., \
                interval=10, plots=False)
    else:
        bins, red, white = [0., 0., 0.]

    if 'scatter_sn' in locals():
        header['scatter_SN'] = scatter_sn

    # Get Hurst exponent - this might happen to fail
    try:
        hurst_exp, cc, val = compute_Hc(y, kind='price')
    except FloatingPointError:
        hurst_exp = -1.
    if 'hurst_exp' in locals():
        header['hurst_exp'] = hurst_exp

    # Save smoothed LC for later inspection
    if savefile != '':
        fout = open(savefile, 'wb')
        hdu = fits.PrimaryHDU(data=[x, y - y_model], header=header)
        hdu.writeto(savefile, overwrite=True)
        pickle.dump([x, y - y_model], fout)
        fout.close()

    return y_model_int, scatter, ls, [bins, red, white], header, tck
