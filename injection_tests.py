'''
Generate a set of light curves with various activity and noise levels,
and run a flare search on them. Then, compare input and output.
'''

import celerite2
from celerite2 import terms
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import loguniform
import itertools
from models import flare_model_mendoza
import glob
import pickle
from fladerunner import analyse_LC, get_results
from astropy import units
import pandas as pd
import os
from pdb import set_trace

def model_LC(sigma, Prot, Q0, dQ, noise, f=0.5, tmax=28., deltat=20./86400., \
            plots=False):
    '''
    By default, assume a 28 days TESS-duration light curve with 20-s cadence.

    Parameters
    ----------
    nsample: number of samples to draw from the generated GP.
    '''
    t_gp = np.arange(0., tmax, deltat)[::100]

    # Quasi-periodic term
    kernel = terms.RotationTerm(sigma=sigma, period=Prot, Q0=Q0, dQ=dQ, f=f)
    gp = celerite2.GaussianProcess(kernel, mean=1.0)
    gp.compute(t_gp)
    mu = gp.sample(size=1)

    # Increase time cadence
    interp = CubicSpline(t_gp, mu[0])
    t = np.arange(0., tmax, deltat)
    y = interp(t)

    y_scatter = y*np.random.normal(loc=1., scale=noise, size=len(t))

    if plots:
        for i in range(np.shape(mu)[0]):
            plt.plot(t, y_scatter, '.')
            plt.plot(t, y, linewidth=3)
        plt.show()
        set_trace()
        plt.close()

    return t, y_scatter

def generate_LCs(folder_out, plots=False, observed=[]):
    '''
    Generate a grid of input parameters for the GP.

    Parameters
    ----------
    observed: if not empty use these light curves as basis for flare addition.
    '''

    def add_flares(y):
        # Flare parameter grid
        flares_per_lc = np.random.randint(low=5, high=30)
        tpeak = np.random.uniform(low=0., high=28., size=flares_per_lc)
        fwhm = loguniform.rvs(a=20./86400., b=1./24., size=flares_per_lc)
        ampl = loguniform.rvs(a=1e-4, b=1.0, size=flares_per_lc)

        y = np.copy(y_quiet)
        for j in range(flares_per_lc):
            par = {}
            par['tpeak0'] = tpeak[j]
            par['fwhm0'] = fwhm[j]
            par['ampl0'] = ampl[j]
            par['D20'] = 1.2151
            y += flare_model_mendoza(t, par)

        return y, [tpeak, fwhm, ampl]

    def save_LC(t, y, flare_pars, fileout, stellar_pars=None):
        '''
        flare_pars: tpeak, fwhm, and ampl
        '''

        LC = {}
        if stellar_pars != None:
            LC['stellar_pars'] = {}
            LC['stellar_pars']['meta'] = ['sigma, Prot, Q0, dQ, noise']
            LC['stellar_pars']['values'] = stellar_pars
        LC['flare_pars'] = {}
        LC['flare_pars']['meta'] = ['tpeak', 'fwhm', 'ampl']
        LC['flare_pars']['values'] = flare_pars
        LC['t'] = [t.min(), t.max(), np.diff(t)[0]]
        LC['y'] = y

        fout = open(fileout, 'wb')
        pickle.dump(LC, fout)
        fout.close()

        return

    if len(observed) == 0:
        # Stellar parameter grid
        sigma = np.logspace(-3, -1.3, 4)
        Prot = np.linspace(0.5, 10., 5)
        Q0 = np.linspace(0.1, 1., 3)
        dQ = np.linspace(0.5, 1.5, 3)
        # Scale TESS's noise level for 1 hr to 20 sec
        noise_range = 1e-6*np.linspace(50., 1000., 5)*(60.*3.)**0.5

        nLC = len(sigma)*len(Prot)*len(Q0)*len(dQ)*len(noise_range)
        print('Generating', nLC, 'light curves...')

        for c, i in enumerate(itertools.product(sigma, Prot, Q0, dQ, noise_range)):
            print(c)
            sigma_i, Prot_i, Q0_i, dQ_i, noise_i = i
            t, y_quiet = model_LC(sigma_i, Prot_i, Q0_i, dQ_i, noise_i)
            y_flare, flare_params = add_flares(y_quiet)
            save_LC(t, y_flare, flare_params, \
                        folder_out + 'LC_' + str(c) + '.pic', stellar_pars=i)
    else:
        from astropy.io import fits

        for c, lcfile in enumerate(observed):
            print(c)
            try:
                lc = fits.open(lcfile + '.fits')
            except FileNotFoundError:
                print(lcfile, ' not found')
                continue
            t = lc[1].data['TIME']
            y_quiet = lc[1].data['PDCSAP_FLUX']
            flag = np.isnan(y_quiet)
            y_quiet = y_quiet[~flag]
            t = t[~flag]
            t -= np.min(t)
            yerr = lc[1].data['PDCSAP_FLUX_ERR'][~flag]/np.median(y_quiet)
            y_quiet /= np.median(y_quiet)
            y_flare, flare_params = add_flares(y_quiet)
            save_LC(t, y_flare, flare_params, \
                    folder_out + lcfile.split('/')[-1] + '_flareadded.pic', \
                    stellar_pars=[0., 0., 0., 0., np.median(yerr)])

            if plots:
                plt.plot(t, y_flare)
                plt.show()
                set_trace()
                plt.close()

    return

def analyse_LCs(folderin, indices=[]):

    plt.ioff()

    saveresfolder = folderin.replace('simulated_LCs', 'simulated_analysis')

    datadir = '/home/giovanni/Projects/data'
    throughput_folder = datadir + '/filters/'
    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)

    LCs = glob.glob(folderin + '*pic')
    LCs = np.sort(LCs)
    print('Analysing', len(LCs), 'simulated light curves...')
    for LC_i, LC in enumerate(LCs[indices]):
        fout_str = saveresfolder + LC.split('/')[-1].replace( \
                        '.pic', '_results.pic')
        # To start from an uncompleted session
        if os.path.isfile(fout_str):
            continue
        print(LC_i, LC)
        data = pickle.load(open(LC, 'rb'))
        t = np.arange(data['t'][0], data['t'][1] + data['t'][2], data['t'][2])
        y = data['y']
        yerr = np.zeros(len(t)) + data['stellar_pars']['values'][-1]
        saveresfile = saveresfolder + LC.split('/')[-1]
        try:
            flarespar, flaresflag = analyse_LC([t, y, yerr], saveresfile, \
                    wth*units.AA, fth)
            fout = open(fout_str, 'wb')
            pickle.dump(flarespar, fout)
            fout.close()
        except TypeError:
            print('Problematic LC:', LC)

    return

def get_simulation_results(folderout, distance_threshold=3):
    '''
    Work one LCs at a time and plot results for matching flares.

    Parameters
    ----------
    distance_threshold: number of data point distance for a flare peak to
    correspond to a true one.
    '''
    import matplotlib
    ampl_in_tot, ampl_out_tot = [], []
    fwhm_in_tot, fwhm_out_tot = [], []

    LCs = glob.glob(folderout + 'LC*pic')
    LCs = sorted(LCs)

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    yerr = []
    det = {}
    det['ampl'] = []
    det['flag'] = []
    det['SNR'] = []
    false_det = {}
    false_det['ampl'] = []
    false_det['flag'] = []
    false_det['SNR'] = []
    for LCi, LC in enumerate(LCs):
        try:
            df = get_results([LC], folderout, \
                    get_lcs_without_flares=False, get_csv=False)
        except FileNotFoundError:
            continue

        LC_in = LC.replace('_analysis', '_LCs').replace('_results.pic', \
                    '.pic')
        data_in = pickle.load(open(LC_in, 'rb'))
        tpeaks_in = data_in['flare_pars']['values'][0]
        fwhm_in = data_in['flare_pars']['values'][1]
        ampl_in = data_in['flare_pars']['values'][2]
        yerr = data_in['stellar_pars']['values'][4]
        flag = df['Duration [min]'] < 1.
        df = df[~flag]
        if len(df['Peak time']) == 0:
            continue
        # This gives detection rates and missed events
        pairs, flag_array_i = find_closest_pair(tpeaks_in, \
                df['Peak time'], distance_threshold*data_in['t'][2])
        det['ampl'].append(ampl_in)
        det['SNR'].append(ampl_in/yerr)
        det['flag'].append(flag_array_i)
        # This gives false positives (False values in flag array)
        pairs_false, flag_array_i_false = find_closest_pair(df['Peak time'], \
            tpeaks_in, 3*data_in['t'][2])
        false_det['ampl'].append(df['Peak amplitude'])
        false_det['flag'].append(flag_array_i_false)
        false_det['SNR'].append(df['Peak amplitude']/yerr)

        index_input = [x[0] for x in pairs]
        index_output = [x[1] for x in pairs]

        if len(pairs) == 0:
            continue

        # Save single LC results to inspect later
        ampl_in_tot.append(ampl_in[index_input])
        ampl_out_tot.append(df.iloc[index_output]['Peak amplitude'])
        fwhm_in_tot.append(fwhm_in[index_input]*24*60.)
        fwhm_out_tot.append(df.iloc[index_output]['FWHM [min]'])

    ampl_in_tot = np.hstack(ampl_in_tot)
    ampl_out_tot = np.hstack(ampl_out_tot)

    fwhm_in_tot = np.hstack(fwhm_in_tot)
    fwhm_out_tot = np.hstack(fwhm_out_tot)

    # Remove too little flares that were detected by chance
    flag = ampl_in_tot > 1e-3
    ampl_in_tot = ampl_in_tot[flag]
    ampl_out_tot = ampl_out_tot[flag]
    fwhm_in_tot = fwhm_in_tot[flag]
    fwhm_out_tot = fwhm_out_tot[flag]

    ax1.plot(ampl_in_tot, ampl_out_tot, 'k.')
    ax2.plot(fwhm_in_tot, fwhm_out_tot, 'k.')

    fit_ampl = np.polyfit(ampl_in_tot, ampl_out_tot, 2, \
            w=(0.1*ampl_out_tot)**-1, full=False, cov=True)
    fit_fwhm = np.polyfit(fwhm_in_tot, fwhm_out_tot, 2, \
            w=(0.1*fwhm_out_tot)**-1, full=False, cov=True)
    print('Fit ampl:', fit_ampl)
    print('fit_fwhm:', fit_fwhm)

    fout = open(folderout + 'corr_coeff.pic', 'wb')
    res = {'fit_ampl': fit_ampl, 'fit_fwhm': fit_fwhm}
    pickle.dump(res, fout)
    fout.close()

    xx = np.logspace(-2.9, 0.1, 1000)
    #ax1.plot(xx, xx, 'r')
    xx2 = np.logspace(-0.5, 1.85, 1000)
    #ax2.plot(xx2, xx2, 'r')

    ax1.plot(xx, np.polyval(fit_ampl[0], xx), 'r')
    ax2.plot(xx2, np.polyval(fit_fwhm[0], xx2), 'r')

    ax1.set_xlabel('Injected peak amplitude', fontsize=14)
    ax1.set_ylabel('Retrieved peak ampl.', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    fig1.tight_layout()
    fig1.savefig(folderout + 'amplitude.pdf')

    ax2.set_xlabel('Injected FWHM [min]', fontsize=14)
    ax2.set_ylabel('Retrieved FWHM [min]', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig2.tight_layout()
    fig2.savefig(folderout + 'fwhm.pdf')
    plt.close('all')

    # Detection rates
    det['ampl'] = np.hstack(det['ampl'])
    det['SNR'] = np.hstack(det['SNR'])
    det['flag'] = np.hstack(det['flag'])
    hTh, bin_edges = np.histogram(det['ampl'], bins=20)
    hRet, _ = np.histogram(det['ampl'][det['flag']], bins=bin_edges)
    hNDe, _ = np.histogram(det['ampl'][~det['flag']], bins=bin_edges)
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = bin_edges[:-1] + 0.5*np.diff(bin_edges)
    ax.plot(bins, hRet/hTh*100., label='True positives')
    ax.plot(bins, hNDe/hTh*100., label='Missed events')
    ax.set_xscale('log')
    ax.set_xlabel('Peak amplitude', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    # False positives
    false_det['ampl'] = np.hstack(false_det['ampl'])
    false_det['flag'] = np.hstack(false_det['flag'])
    hObs, _ = np.histogram(false_det['ampl'], bins=bin_edges)
    hFP, _ = np.histogram(false_det['ampl'][~false_det['flag']], bins=bin_edges)
    ax.plot(bins, hFP/hObs*100., label='False positives')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folderout + 'detection_stats.pdf')
    plt.close('all')

    set_trace()

    return

def find_closest_pair(arr1, arr2, tolerance):
    '''
    Return indices of arr1 and arr2 that are closer than tolerance data ponints.
    Only the closest pairs are provided.
    '''
    if type(arr2) == np.float64:
        arr2 = [arr2]

    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    flag_array = []

    pairs = []
    truepos, nondet = [], []
    for arr1_idx, a in enumerate(arr1):
        diffs = np.abs(arr2 - a)
        arr2_min_idx = np.argmin(diffs)
        if diffs[arr2_min_idx] <= tolerance:
            pairs.append((arr1_idx, arr2_min_idx))
            flag_array.append(True)
        else: #no match within tolerance
            flag_array.append(False)

    return pairs, flag_array

def get_lcs_without_flares(tgs):
    '''
    Input is a file with names of objects without flares detected.
    '''

    tgs_wo_flares = pickle.load(open(tgs, 'rb'))
    tgs = ['s00' + str(h['SECTOR']) + '-' + str(h['TICID']) for h in tgs_wo_flares]
    set_trace()

    return
