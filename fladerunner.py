# Routines to run the code and produce tables of parameter results

import os
import sys
sys.path.append('../fladerunner/')
import copy
import lc_class
import pickle
from astropy.io import fits
from astropy import units, constants
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib
import matplotlib.pyplot as plt
import lmfit
from matplotlib.ticker import ScalarFormatter
import flare_detection as fd
import pandas as pd
import powerlaw
from scipy import stats
import seaborn as sns
from sklearn import linear_model
homedir = os.path.expanduser('~')
from pdb import set_trace
import glob

datadir = '/home/giovanni/Projects/data'
throughput_folder = datadir + '/filters/'

def call_LC(LC, saveresfolder):

    saveresfile = saveresfolder + LC.split('/')[-1]
    fout_str = saveresfile.replace('.fits', '_results.pic')
    if os.path.exists(fout_str):
        print('LC:', fout_str + ' - already analyzed.')
        return

    print('LC:', LC)
    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)
    flarespar, flaresflag = analyse_LC(LC, saveresfolder, wth*units.AA, fth)

    fout = open(fout_str, 'wb')
    pickle.dump(flarespar, fout)
    fout.close()

    return

def analyse_LC(LC, saveresfolder, wth, fth, flare_threshold=4., rebin=0):
    '''
    Look for flares and dips on a single LC.
    '''

    flarespar, flaresflag = fd.find_flares(LC, wth=wth, fth=fth, \
        flare_threshold=flare_threshold, flatten=True, plot_flat=False, \
        saveplots=saveresfolder, normalise=True, \
        fit_continuum=1, complexity=5, clip=True, \
        rebin=rebin, verbose=True, filt_kernel_size=11)

    return flarespar, flaresflag

def get_results(LCsample, result_path, get_lcs_without_flares=False, \
                get_csv=True):
    '''
    Produce a pandas DataFrame with parameter results from a set of result
    files.
    '''

    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)

    results = {}
    parameters = ['LCname', 'Peak amplitude', 'Peak SNR', \
        'Peak luminosity [erg s$^{-1}$]', \
        'Duration [min]', 'FWHM [min]', 'Energy (Shibayama) [erg]', \
        'Energy (Davenport) [erg]', 'ED [s]', 'Peak time', 'n_event', \
        'total_duration', 'Teff [K]', 'logg', 'radius', \
        'phot_var', 'tessmag', 'Prot', 'FAP', 'redchi2', 'ticname', \
        'ra', 'dec', 'scatter_SN']
    for par in parameters:
        results[par] = []

    for LCi, LCfile in enumerate(LCsample):

        print(LCfile)
        try:
            resLC = pickle.load(open(LCfile, 'rb'))
        except EOFError:
            # The file is empty
            continue

        if len(resLC) > 0:
            header = resLC[-1]
            # If header is None, a bunch of -999 will be output
        else:
            # Light curve was discarded
            continue
        if resLC == [None]:
            continue

        if len(resLC) > 1 and get_lcs_without_flares:
            continue
        elif len(resLC) > 1 and not get_lcs_without_flares:
            pass
        elif len(resLC) == 1 and get_lcs_without_flares:
            try:
                filename = LCfile.split('/')[-1].split('_results')[0]
            except IndexError:
                continue
            lcs_without_flares.append(header)
            count_lcs_without_flares += 1

        for fi, flare in enumerate(resLC[:-1]):
            try:
                for n in np.arange(flare.npeaks):
                    duration = flare.durations[n].to(units.min).value
                    if type(duration) == np.ndarray:
                        results['Duration [min]'].append( \
                                flare.durations[n].to(units.min).value[3])
                    else:
                        # Somehow fit failed
                        continue
                    # Identify LC, to later compute flare rates
                    results['LCname'].append( \
                            LCfile.split('/')[-1].split('_results.pic')[0])
                    # Identify flare event number for given LC
                    results['n_event'].append(fi)
                    results['redchi2'].append(flare.result_nodip.redchi)
                    tt, thisflare = flare.get_single_profile(n)
                    results['Peak time'].append(flare.tref \
                        + flare.result_nodip.params['tpeak' + str(n)].value)
                    results['total_duration'].append( \
                            flare.tend - flare.tbeg)
                    results['Peak amplitude'].append(thisflare.max())
                    results['Peak SNR'].append(thisflare.max()/flare.yerrdata[0])
                    results['FWHM [min]'].append(24.*60* \
                        flare.result_nodip.params_LM['fwhm' + str(n)].value)
                    results['ED [s]'].append(flare.EDs[n].value)
                    try:
                        flare.Tstar = header['TEFF']*units.K
                        flare.Rstar = header['RADIUS']*constants.R_sun
                        flare.Tflare = 9000.*units.K
                        flare.wth = wth*units.AA
                        flare.fth = fth
                        results['Energy (Shibayama) [erg]'].append( \
                                    flare.energies[n].value)
                        results['Teff [K]'].append(header['TEFF'])
                        results['radius'].append(header['RADIUS'])
                        #results['phot_var'].append(header['phot_var'].values[0])
                    except TypeError:
                        results['Energy (Shibayama) [erg]'].append(-999.)
                        results['Teff [K]'].append(-999.)
                        results['radius'].append(-999.)
                        results['phot_var'].append(-999.)
                    except IndexError:
                        # This can only happen for the phot_var info
                        results['phot_var'].append(-999.)

                    # Stellar params
                    try:
                        results['logg'].append(header['LOGG'])
                        results['ra'].append(header['RA_OBJ'])
                        results['dec'].append(header['DEC_OBJ'])
                        results['ticname'].append(header['OBJECT'])
                        results['tessmag'].append(header['TESSMAG'])
                    except TypeError:
                        results['logg'].append(-999.)
                        results['ra'].append(-999.)
                        results['dec'].append(-999.)
                        results['ticname'].append(-999.)
                        results['tessmag'].append(-999.)
                    # If possible, compute energy with Davenport's method
                    #tgd = 'TIC ' + df_stassun['TIC'].astype('string') \
                    #        == header['OBJECT']
                    #if np.sum(tgd) > 0:
                    #    distance = df_stassun['Dist'][tgd].values[0] \
                    #            *units.parsec
                    #    mag = header['TESSMAG']
                    #    flare.get_stellar_luminosity('TESS', mag, distance)
                    #    results['Peak luminosity [erg s$^{-1}$]'].append( \
                    #        flare.stellar_luminosity.value*thisflare.max())
                    #    results['Energy (Davenport) [erg]'].append( \
                    #            flare.get_flare_energy(n, \
                    #            method='davenport').value)
                    #else:
                    results['Energy (Davenport) [erg]'].append(-999.)
                    results['Peak luminosity [erg s$^{-1}$]'].append(-999.)
                    try:
                        results['Prot'].append(header['Prot_[days]'])
                        results['FAP'].append(header['FAP'])
                        results['scatter_SN'].append(header['scatter_SN'])
                    except TypeError:
                        results['Prot'].append(-999.)
                        results['FAP'].append(-999.)
                        results['scatter_SN'].append(-999.)

                if flare.npeaks > 1:
                    # See what would happen with a single-flare fit
                    fcopy = copy.deepcopy(flare)
                    fcopy.result_nodip.params = copy.deepcopy( \
                            flare.QPP_candidate['oneflare_model'].params)
                    fcopy.get_single_duration(0)
                    #results_oneflare['']
            except AttributeError as ae:
                errmsg = 'OSError: No SIMPLE card found, this file does ' \
                        + 'not appear to be a valid FITS file.'
                if errmsg in str(ae):
                    continue

    if not get_lcs_without_flares:
        for p in parameters:
            results[p] = np.array(results[p])
        df = pd.DataFrame.from_dict(results)
        if get_csv:
            df.to_csv(result_path + 'results_table.csv', index=False)
        else:
            return df
    else:
        fout = open(resfolder + 'lcs_without_flares.pic', 'wb')
        pickle.dump(lcs_without_flares, fout)
        fout.close()

    return
