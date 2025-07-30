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
        flares_per_lc = np.random.randint(low=10, high=100)
        tpeak = np.random.uniform(low=0., high=28., size=flares_per_lc)
        fwhm = loguniform.rvs(a=20./86400., b=1./24., size=flares_per_lc)
        ampl = loguniform.rvs(a=1e-4, b=0.1, size=flares_per_lc)

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
        LC['t'] = t
        LC['y'] = y

        fout = open(fileout, 'wb')
        pickle.dump(LC, fout)
        fout.close()

        return

    if len(observed) == 0:
        # Stellar parameter grid
        sigma = np.logspace(-3, -1.3, 10)
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
            if c > 100:
                continue
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

    saveresfolder = folderin.replace('observed_LCs', 'observed_analysis')

    datadir = '/home/giovanni/Projects/data'
    throughput_folder = datadir + '/filters/'
    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)

    LCs = glob.glob(folderin + '*pic')
    LCs = np.sort(LCs)
    print('Analysing', len(LCs), 'simulated light curves...')
    for LC_i, LC in enumerate(LCs[indices]):
        print(LC_i, LC)
        data = pickle.load(open(LC, 'rb'))
        t = data['t']
        y = data['y']
        yerr = np.zeros(len(t)) + data['stellar_pars']['values'][-1]
        saveresfile = saveresfolder + LC.split('/')[-1]
        fout_str = saveresfile.replace('.pic', '_results.pic')
        flarespar, flaresflag = analyse_LC([t, y, yerr], saveresfile, \
                    wth*units.AA, fth)

        fout = open(fout_str, 'wb')
        pickle.dump(flarespar, fout)
        fout.close()

    return

def get_simulation_results(folderout):
    '''
    Work one LCs at a time and plot results for matching flares.
    '''
    import matplotlib

    LCs = glob.glob(folderout + '*pic')
    LCs = sorted(LCs)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    residuals = []
    ampl_inj = []
    ampl_out = []
    yerr = []
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
        flag = df['Duration [min]'] <= 1.
        df = df[~flag]
        pairs = find_closest_pair(df['Peak time'], tpeaks_in, \
                3*data_in['t']['values'][2])
        index_output = [x[0] for x in pairs]
        index_input = [x[1] for x in pairs]

        if len(pairs) == 0:
            continue

        ampl_inj.append(ampl_in[index_input])
        ampl_out.append(df.iloc[index_output]['Peak amplitude'])
        yerr.append(df.iloc[index_output]['Peak SNR'])
        ax1.scatter(ampl_in[index_input], \
            df.iloc[index_output]['Peak amplitude'], \
            marker='.', color='k')
        ax2.scatter(fwhm_in[index_input]*24*60., \
            df.iloc[index_output]['FWHM [min]'], marker='.', color='k')

        residuals.append(abs(1. \
            - df.iloc[index_output]['Peak amplitude']/ampl_in[index_input]))

    xx = np.logspace(-2.9, -0.2, 1000)
    ax1.plot(xx, xx, 'r')
    xx2 = np.logspace(-0.7, 2, 1000)
    ax2.plot(xx2, xx2, 'r')

    ax1.set_xlabel('Injected peak amplitude', fontsize=14)
    ax1.set_ylabel('Retrieved peak amplitude', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig(folderout + 'amplitude.pdf')
    ax2.set_xlabel('Injected FWHM [min]', fontsize=14)
    ax2.set_ylabel('Retrieved FWHM [min]', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(folder_out + 'fwhm.pdf')
    plt.close('all')

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

    pairs = []

    for arr1_idx, a in enumerate(arr1):
        diffs = np.abs(arr2 - a)
        arr2_min_idx = np.argmin(diffs)
        if diffs[arr2_min_idx] <= tolerance:
                pairs.append((arr1_idx, arr2_min_idx))
        # else: no match within tolerance
    return pairs

def get_lcs_without_flares(tgs):
    '''
    Input is a file with names of objects without flares detected.
    '''

    tgs_wo_flares = pickle.load(open(tgs, 'rb'))
    tgs = ['s00' + str(h['SECTOR']) + '-' + str(h['TICID']) for h in tgs_wo_flares]
    set_trace()

    return
