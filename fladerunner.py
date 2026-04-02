# Routines to run the code and analyse results

import os
import sys
import copy
import pickle
from astropy.io import fits
from astropy import units, constants
from astropy.io.votable import from_table
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import lmfit
from scipy.optimize import curve_fit
import pandas as pd
import powerlaw
from scipy import stats, integrate, special
import seaborn as sns
homedir = os.path.expanduser('~')
from pdb import set_trace
import glob
from tabulate import tabulate
import seaborn as sns
sys.path.append('../fladerunner/')
import flare_class

plt.ioff()

datadir = '/home/giovanni/Projects/data'
throughput_folder = datadir + '/filters/'
# CHEOPS targets for comparison
cheops_targets = pd.read_csv('/home/giovanni/Projects/CHEOPS/ancillary/' \
        + 'shortterm_variability/tess_mdwarfs/cheopsnames_vs_ticnames.csv')
cheops_tics = 'TIC ' + cheops_targets['ticname'].astype('str')

sun_area = np.pi*constants.R_sun.to(units.cm)**2

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

    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)
    flarespar, flaresflag = fd.find_flares(LC, wth=wth, fth=fth, \
        flare_threshold=flare_threshold, flatten=True, plot_flat=False, \
        saveplots=saveresfolder, normalise=True, \
        fit_continuum=1, complexity=5, clip=True, \
        rebin=rebin, verbose=True, filt_kernel_size=11)

    return flarespar, flaresflag

def get_results_simulations(LCsample, result_path, get_csv=True):
    '''
    To be used for the results of injection-recovery tests: produce a pandas
    DataFrame with parameter results from a set of result files.
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
            except AttributeError as ae:
                errmsg = 'OSError: No SIMPLE card found, this file does ' \
                        + 'not appear to be a valid FITS file.'
                if errmsg in str(ae):
                    continue

    for p in parameters:
        results[p] = np.array(results[p])
    df = pd.DataFrame.from_dict(results)

    if get_csv:
        df.to_csv(result_path + 'results_table.csv', index=False)
        return
    else:
        return df

def get_results(resfolder, min_flares=100, sectors=range(27, 88), \
                only_cheopstargets=False, get_lcs_without_flares=False, \
                maxT=7000., maxR=1.6, maxProt=10., maxFAP=0.01, maxTmag=11.):
    '''
    Parameters
    ----------
    min_flares (int): minimum number of flares for a target to be included
    in the analysis.
    '''

    # Injection test results
    file_inj = resfolder + '../injection_tests/simulated_analysis/corr_coeff.pic'
    corr_coeffs = pickle.load(open(file_inj, 'rb'))
    corr_ampl = corr_coeffs['fit_ampl'][0]
    corr_fwhm = corr_coeffs['fit_fwhm'][0]

    caII_IRT = pd.read_csv(resfolder + '../literature_data/CaII_IRT_GaiaDR3.csv', \
            skipinitialspace=True)

    stellar_pars_file = resfolder + '../literature_data/crossmarch_stassun_xmm.csv'

    good_targets_gaia = resfolder + 'good_targets_gaia_DR3_reddening.csv'
    gtg = pd.read_csv(good_targets_gaia)
    keep_idx = gtg.groupby('ticname')['ang_sep'].idxmin()
    gtg = gtg.loc[keep_idx].reset_index(drop=True)

    if len(glob.glob(stellar_pars_file)) == 0:
        tic_params_file = resfolder + '../literature_data/flarestars_parameters_stassun2019.vot'
        stassun = Table.read(tic_params_file)
        xmm_fits = fits.open(resfolder + '../literature_data/xmmsl2t.fits')
        xmm = Table(xmm_fits[1].data)
        # Crossmatch XMM and Stassun catalogs. idx are the closest match in
        # Stassun to XMM targets. Only rows with coordinates can be used
        c_stassun = SkyCoord(ra=stassun['RAJ2000'], dec=stassun['DEJ2000'])
        flag = np.logical_or(np.isnan(c_stassun.ra), np.isnan(c_stassun.dec))
        stassun = stassun[~flag]
        c_stassun = SkyCoord(ra=stassun['RAJ2000'], dec=stassun['DEJ2000'])
        c_xmm = SkyCoord(ra=xmm['RA']*units.degree, dec=xmm['DEC']*units.degree)
        idx, d2d, d3d = c_stassun.match_to_catalog_sky(c_xmm)
        # Add columns to stassun table with the solutions
        for band in ['6', '7', '8']:
            # Units: W/m**2
            stassun['FLUX_B' + band] = xmm['FLUX_B' + band][idx]*1e-15
            stassun['FLUX_B' + band + '_ERR'] \
                                    = xmm['FLUX_B' + band + '_ERR'][idx]*1e-15
            stassun['DET_ML_B' + band] = xmm['DET_ML_B' + band][idx]
        stassun['dist_from_xmm'] = d2d.to(units.arcsec)

        # It's convenient to work with a pandas dataframe later
        df_stassun = stassun.to_pandas()
        df_stassun.to_csv(stellar_pars_file, index=False)
    else:
        df_stassun = pd.read_csv(stellar_pars_file)

    # Photometric variability
    phvar_folder = resfolder + '../photometric_variability/'
    phvar_data = glob.glob(phvar_folder + '*csv')
    for i, phd in enumerate(phvar_data):
        phdi = pd.read_csv(phd)
        if i == 0:
            phditot = copy.deepcopy(phdi)
        else:
            phditot = pd.concat([phditot, phdi])

    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)

    results = {}
    # !!! Total duration is the duration of the associated complex flare
    parameters = ['LCname', 'Peak amplitude', 'Peak luminosity [erg s$^{-1}$]', \
        'Duration [min]', 'FWHM [min]', 'Energy (Shibayama) [erg]', \
        'Energy [erg]', 'ED [s]', 'Peak time [BTJD]', 'n_event', \
        'impulsive_ED_fraction', 't12', 'total_duration', 'Teff [K]', 'logg', \
        'feh', 'radius', 'phot_var', 'CaII_IRT', 'tessmag', 'Prot', 'FAP', \
        'distance', 'redchi2', 'ticname', 'ra', 'dec', 'scatter_SN', \
        'dist_from_xmm', 'E_B-V']
    for band in ['6', '7', '8']:
        parameters.append('FLUX_B' + band)
        parameters.append('FLUX_B' + band + '_ERR')
        parameters.append('DET_ML_B' + band)
    results_oneflare = {}
    lcs_without_flares = []
    for sector in sectors:

        foutname = resfolder + 'results_sector' + str(sector) + '.pic'
        if not get_lcs_without_flares:
            if os.path.exists(foutname) and os.path.getsize(foutname) > 100:
                print('Sector ' + str(sector) + ': results already collected.')
                file_res = pickle.load(open(foutname, 'rb'))
                results['S' + str(sector)] = file_res
                continue

        else:
            count_lcs_without_flares = 0

        sample = glob.glob(resfolder + 'S' + str(sector) + '/*-s00' \
                        + str(sector) + '*results.pic')
        params_s = {}
        params_s_oneflare = {}
        for p in parameters:
            params_s[p] = []
            params_s_oneflare[p] = []
        for LCi, LCfile in enumerate(sample):

            print(LCfile)

            try:
                resLC = pickle.load(open(LCfile, 'rb'))
            except EOFError:
                # The file is empty
                continue

            if len(resLC) > 0:
                header = resLC[-1]
            else:
                # Light curve was discarded
                continue

            if len(resLC) > 1 and get_lcs_without_flares:
                continue
            elif len(resLC) > 1 and not get_lcs_without_flares:
                pass
            elif len(resLC) == 1 and get_lcs_without_flares:
                try:
                    filename = LCfile.split('/')[-1].split('_results')[0]
                    phvarflag = phditot['lcname'] == filename
                    header['phot_var'] = phditot[phvarflag]['phot_var'].values[0]
                    header['LCfile'] = LCfile.split('/')[-1].split('_results')[0]
                except IndexError:
                    continue
                lcs_without_flares.append(header)
                count_lcs_without_flares += 1

            for fi, flare_uncorr in enumerate(resLC[:-1]):
                try:
                    for n in np.arange(flare_uncorr.npeaks):
                        # Identify LC, to later compute flare rates
                        lcname = LCfile.split('/')[-1].split('_results.pic')[0]
                        params_s['LCname'].append(lcname)

                        # Identify flare event number for given LC
                        params_s['n_event'].append(fi)
                        # Recompute flare profile after injection tests
                        flare = copy.deepcopy(flare_uncorr)
                        params_s['redchi2'].append(flare.result_nodip.redchi)
                        flare.result_nodip.params_LM['ampl' + str(n)].value \
                            = np.polyval(corr_ampl, \
                                flare_uncorr.result_nodip.params_LM['ampl' + str(n)].value)
                        flare.result_nodip.params_LM['fwhm' + str(n)].value \
                            = np.polyval(corr_fwhm, \
                                flare_uncorr.result_nodip.params_LM['fwhm' + str(n)].value)

                        tt, thisflare = flare.get_single_profile(n)
                        params_s['Peak amplitude'].append(thisflare.max())
                        params_s['Peak time [BTJD]'].append(flare.tref \
                            + flare.result_nodip.params_LM['tpeak' + str(n)].value)
                        params_s['Duration [min]'].append( \
                                flare.get_single_duration(n).to(units.min).value)

                        # This is data-driven, so it shouldn't change with
                        # injection tests
                        params_s['total_duration'].append( \
                                flare.tend - flare.tbeg)
                        params_s['t12'].append(flare.t12.value*24*60.)
                        params_s['FWHM [min]'].append(24.*60.
                           *flare.result_nodip.params_LM['fwhm' + str(n)].value)
                        params_s['ED [s]'].append(flare.get_flare_ED(n).value)

                        # Energy emitted during impulsive phase. FWHM on observed data is often wrong
                        # This is also data-driven, so it should be ok without correction from injection tests
                        fl = flare_class.flare(tt.value, thisflare, 0., 0.)
                        try:
                            ini, fin = fl.get_flare_t12()
                            flag = np.logical_and(tt >= ini, tt <= fin)
                            int_impulsive = integrate.trapezoid(thisflare[flag], x=tt[flag])
                            int_all = integrate.trapezoid(thisflare, x=tt)
                            params_s['impulsive_ED_fraction'].append( \
                                        (int_impulsive/int_all).value)
                        except TypeError:
                            # The flare is at some edge
                            params_s['impulsive_ED_fraction'].append(-999.)
                        try:
                            flare.Tstar = header['TEFF']*units.K
                            flare.Rstar = header['RADIUS']*constants.R_sun
                            flare.Tflare = 9000.*units.K
                            flare.wth = wth*units.AA
                            flare.fth = fth
                            params_s['Energy (Shibayama) [erg]'].append( \
                                flare.get_flare_energy(n, method='shibayama').value)
                            params_s['Teff [K]'].append(header['TEFF'])
                            params_s['radius'].append(header['RADIUS'])
                            phvarflag = phditot['lcname'] == params_s['LCname'][-1]
                            params_s['phot_var'].append( \
                                    phditot[phvarflag]['phot_var'].values[0])
                        except TypeError:
                            params_s['Energy (Shibayama) [erg]'].append(-999.)
                            params_s['Teff [K]'].append(-999.)
                            params_s['radius'].append(-999.)
                            params_s['phot_var'].append(-999.)
                        except IndexError:
                            # This can only happen for the phot_var info
                            params_s['phot_var'].append(-999.)

                        # Stellar params
                        params_s['logg'].append(header['LOGG'])
                        if header['MH'] == None:
                            params_s['feh'].append(0.)
                        else:
                            params_s['feh'].append(header['MH'])
                        params_s['ra'].append(header['RA_OBJ'])
                        params_s['dec'].append(header['DEC_OBJ'])
                        params_s['ticname'].append(header['OBJECT'])
                        params_s['tessmag'].append(header['TESSMAG'])
                        # If possible, compute energy with Davenport's method
                        tgd = 'TIC ' + df_stassun['TIC'].astype('string') \
                                == header['OBJECT']
                        if np.sum(tgd) > 0:
                            distance = df_stassun['Dist'][tgd].values[0]*units.parsec
                            params_s['distance'].append(distance.to(units.m).value)
                            params_s['E_B-V'].append( \
                                                df_stassun['E_B-V_'][tgd].values[0])
                            mag = header['TESSMAG']
                            flare.get_stellar_luminosity('TESS', mag, distance)
                            params_s['Peak luminosity [erg s$^{-1}$]'].append( \
                                flare.stellar_luminosity.value*thisflare.max())
                            params_s['Energy [erg]'].append( \
                                flare.get_flare_energy(n, method='davenport').value)
                            for band in ['6', '7', '8']:
                                params_s['FLUX_B' + band].append( \
                                    df_stassun['FLUX_B' + band][tgd].values[0])
                                params_s['FLUX_B' + band + '_ERR'].append( \
                                    df_stassun['FLUX_B' + band + '_ERR'][tgd].values[0])
                                params_s['DET_ML_B' + band].append( \
                                    df_stassun['DET_ML_B' + band][tgd].values[0])
                            params_s['dist_from_xmm'].append( \
                                df_stassun['dist_from_xmm'][tgd].values[0])
                        else:
                            params_s['Energy [erg]'].append(-999.)
                            params_s['Peak luminosity [erg s$^{-1}$]'].append(-999.)
                            params_s['distance'].append(-999.)
                            params_s['E_B-V'].append(-999.)
                            for band in ['6', '7', '8']:
                                params_s['FLUX_B' + band].append(-999.)
                                params_s['FLUX_B' + band + '_ERR'].append(-999.)
                                params_s['DET_ML_B' + band].append(-999.)
                            params_s['dist_from_xmm'].append(-999.)
                        params_s['Prot'].append(header['Prot_[days]'])
                        params_s['FAP'].append(header['FAP'])
                        params_s['scatter_SN'].append(header['scatter_SN'])
                        dr3flag = caII_IRT['target_id'] == header['OBJECT']
                        if np.sum(dr3flag) > 0:
                            params_s['CaII_IRT'].append( \
                              caII_IRT['activityindex_espcs'][dr3flag].values[0])
                        else:
                            params_s['CaII_IRT'].append(-999.)

                except AttributeError as ae:
                    errmsg = 'OSError: No SIMPLE card found, this file does ' \
                            + 'not appear to be a valid FITS file.'
                    if errmsg in str(ae):
                        continue
        if not get_lcs_without_flares:
            for p in parameters:
                params_s[p] = np.array(params_s[p])
            results['S' + str(sector)] = params_s
            fout = open(foutname, 'wb')
            pickle.dump(params_s, fout)
            fout.close()

    if get_lcs_without_flares:
        fout = open(resfolder + 'lcs_without_flares.pic', 'wb')
        pickle.dump(lcs_without_flares, fout)
        fout.close()
        return

    params = {}
    for p in parameters:
        #if 'gaia_DR2' not in p and 'BP-RP' not in p:
        params[p] = np.hstack([results['S' + str(i)][p] for i in sectors])

    # Let's only consider FGKM stars, luminosity class V stars
    flag_bad = np.logical_or.reduce((params['redchi2'] > np.inf, \
                params['ED [s]'] == 0., params['impulsive_ED_fraction'] == -999., \
                params['Energy [erg]'] == -999., np.isnan(params['distance']), \
                params['distance'] == -999., np.isnan(params['Energy [erg]']), \
                params['radius'] == -999., params['logg'] == None, \
                params['Teff [K]'] == -999., params['phot_var'] < 0.))
    print('\nNo stellar params: {:.2f}%'.format( \
            np.sum(flag_bad)/len(flag_bad)*100.))
    for p in parameters:
        params[p] = np.array(params[p][~flag_bad])

    flag_bad = np.logical_or(params['scatter_SN'] > 3., \
                params['Duration [min]'] <= 1.)
    print('\nToo large scatter/too short flares: {:.2f}%'.format( \
            np.sum(flag_bad)/len(flag_bad)*100.))
    for p in parameters:
        params[p] = np.array(params[p][~flag_bad])

    # Get a dataframe
    keys, values = zip(*params.items())
    df = pd.DataFrame(data=np.transpose(values), columns=keys)
    # Restrict to FGKM stars, with detected rotation period, luminosity class V
    flagT = filter_df(df, maxT, maxR, maxProt, maxFAP, maxTmag)
    df = df[flagT]

    # Add Gaia colours and Ro values
    df = pd.merge(df, gtg, on='ticname')
    df['BP-RP_0'] = df['bp_rp'] - df['ebpminrp_gspphot']
    df = df.astype({'logg':'float', 'Teff [K]':'float', 'BP-RP_0':'float', \
        'feh':'float', 'Duration [min]':'float', 'radius':'float', \
        'FWHM [min]':'float', 'Energy [erg]':'float',
        'Peak luminosity [erg s$^{-1}$]':'float', \
        'Peak amplitude':'float', 'ED [s]':'float'}, copy=False)
    df['tau_conv_bonanno'] = compute_tau_conv_bonanno( \
            df['BP-RP_0'], df['Teff [K]'], df['logg'], df['feh'])
    df['Ro_bonanno'] = df['Prot']/df['tau_conv_bonanno']
    df = df.astype({'Ro_bonanno':'float'}, copy=True)
    df.loc[df['Ro_bonanno'] < 1e-3, 'Ro_bonanno'] = 1e-3
    df = df[~np.isnan(df['Ro_bonanno'])]

    # Add estimate for flare area and conversions, and other stats
    Tspot = -3.58e-5*df['Teff [K]']**2 + 1.0188*df['Teff [K]'] + 239.3
    star_area = np.pi*(df['radius']*constants.R_sun.to(units.cm))**2
    Aspot = df['phot_var']*star_area*(1. - (Tspot/df['Teff [K]'])**4)**-1
    df['Aspot [Asun]'] = Aspot/sun_area
    df['Peak luminosity [W]'] = df['Peak luminosity [erg s$^{-1}$]']*units.erg.to(units.J)
    df['Area [m$^2$]'] = (df['Peak luminosity [W]']/(constants.sigma_sb*(9000.*units.K)**4)).values
    df['Peak flux [W m$^{-2}$]'] = (df['Peak luminosity [W]']/(4.*np.pi*df['distance']**2)).values
    df['Peak flux [erg s$^{-1}$ cm$^{-2}$]'] = df['Peak flux [W m$^{-2}$]']*1e11
    df['Energy [J]'] = (df['Energy [erg]']*units.erg.to(units.J)).values
    df['Aspot_mean [Asun]'] = df.groupby('ticname')['Aspot [Asun]'].transform('mean')
    # Convective turnover time [days] following extrapolation of eq. 36 in
    # Cranmer & Saar (2001) (done by Stelzer+2016) - see caveats
    df['tau_conv_cranmer'] = compute_tau_conv_cranmer(df['Teff [K]'])
    df['Ro_cranmer'] = df['Prot']/df['tau_conv_cranmer']
    df['Impulsiveness [min$^{-1}$]'] = df['Peak amplitude']/df['FWHM [min]']
    ro_bins = [0., 0.1, 0.5, np.inf]
    df['Ro_bins'] = pd.cut(df['Ro_bonanno'], bins=ro_bins, \
        labels=[r'$Ro \leq 0.1$', r'$0.1 < Ro \leq 0.5$', r'$Ro > 0.5$'])
    tlim = [0., 4200., 5300., 5950., 7200.]
    sptype = ['M', 'K', 'G', 'F']
    df['Spectral type'] = pd.cut(df['Teff [K]'], bins=tlim, labels=sptype)
    surf_g = (10**df['logg'])/100.*units.m/units.s**2
    mass = surf_g*df['radius']*(constants.R_sun)**2/constants.G
    df['mass'] = mass/constants.M_sun
    # Get log of activity index
    IRTflag = df['CaII_IRT'] > 0.
    lowmet = [-3.3391, -0.1564, -0.1046, 0.0311]
    solarmet = [-3.3467, -0.1989, -0.1020, 0.0349]
    modmet = [0.25, -3.3501, -0.2137, -0.1029, 0.0357]
    highmet = [-3.3527, -0.2219, -0.1056, 0.0353]
    theta = np.log10(df['Teff [K]'])
    # Let's assume all stars have solar metallicity
    df.loc[IRTflag, 'CaII_IRT'] = np.polyval(solarmet[::-1], theta[IRTflag]) \
                + np.log10(df['CaII_IRT'][IRTflag].astype(float))

    # Separate isolated flares from member of complex flares
    # (SFC as "sympathetic flare candidates")
    group_cols = ['LCname', 'n_event']
    df['peaks_per_event'] = df.groupby(group_cols)['LCname'].transform('count')
    df['Flare type'] = ['IF']*len(df)
    df.loc[df['peaks_per_event'] > 1, 'Flare type'] = 'SFC'
    df.to_csv(resfolder + 'flare_params.csv', index=False)
    single = df['Flare type'] == 'IF'

    # Here, CF as "complex flares"
    dfc = condense_flare_cascades(df, resfolder)
    dfc['Flare type'] = ['IF']*len(dfc)
    dfc.loc[dfc['peaks_per_event'] > 1, 'Flare type'] = 'CF'
    complex = dfc['Flare type'] == 'CF'

    # Useful log quantities
    for dd in [df, dfc]:
        dd['log_radius'] = np.log10(dd['radius'])
        dd['log_impulse'] = np.log10(dd['Impulsiveness [min$^{-1}$]'])
        dd['log_energy'] = np.log10(dd['Energy [erg]'])
        dd['log_duration'] = np.log10(dd['Duration [min]'])
        dd['log_amplitude'] = np.log10(dd['Peak amplitude'])
        dd['log_Ro'] = np.log10(dd['Ro_bonanno'])
        dd['log_ED'] = np.log10(dd['ED [s]'])
        dd['log_fwhm'] = np.log10(dd['FWHM [min]'])
        dd.sort_values('Peak time [BTJD]', inplace=True)
        dd['waiting_time'] = dd.groupby(['ticname', 'LCname'])[ \
                            'Peak time [BTJD]'].diff()*24.*60.
            # Flare rate per target

        dd['flares_per_target'] = dd.groupby(['ticname', 'Flare type']).transform( \
                                'size').astype('float')
        dd['time_on_target'] = dd.groupby('ticname')['LCname'].transform( \
                                'nunique').astype('float')*28.
        dd['rate_per_target'] = np.log10(dd['flares_per_target']/dd['time_on_target'])

    nfl = get_non_flaring_stars(resfolder, maxT, maxR, maxTmag, maxProt, maxFAP)

    ### Trends
    #Tstar_vs_Rstar_vs_Ro(df, nfl, resfolder)
    #compare_Ro(df, resfolder)
    #flaring_vs_nonflaring_stars(df, nfl, resfolder)
    #flare_rate_per_target(pd.concat([df, dfc[complex]]), resfolder)
    #energy_rate(df, resfolder)
    #energy_spotarea(df, resfolder)
    #peaks_vs_ro(dfc, resfolder)
    #waiting_time_distribution(df, resfolder, 'IF + SFC', lab='Part of complex')
    #waiting_time_distribution(dfc, resfolder, 'IF + CF', lab='Complex')
    #set_trace()
    ### Fits with stellar parameters (all in log quantities)
    #simple_complex_fit(df[single], df[~single], dfc[complex], \
    #        'log_radius', 'log_energy', \
    #        r'$\log (R_\star/R_\odot)$', r'$\log$ Energy [erg]', resfolder)
    #simple_complex_fit(df[single], df[~single], dfc[complex], \
    #        'logg', 'log_duration', \
    #        r'$\log g$', r'$\log$ Duration [min]', resfolder)
    #simple_complex_fit(df[single], df[~single], dfc[complex], \
    #        'logg', 'log_impulse', \
    #        r'$\log g$', r'$\log$ Impulsiveness [min$^{-1}$]', resfolder)
    #simple_complex_fit(df[single], df[~single], dfc[complex], \
    #        'log_energy', 'log_fwhm', \
    #        r'$\log$ Energy [erg]', r'$\log$ FWHM [min]', resfolder)
    #simple_complex_fit(df[single], df[~single], dfc[complex], \
    #        'log_amplitude', 'log_duration', \
    #        r'$\log$ Amplitude', r'$\log$ Duration [min]', resfolder)
    #consecutive_flare_stats(df, resfolder)
    #segment_regression(pd.concat([df, dfc[complex]]), 'log_ED', resfolder, \
    #    labelx=r'$\log Ro$', labely=r'$\log$ ED [s]')
    #segment_regression(pd.concat([df, dfc[complex]]), 'rate_per_target', \
    #                resfolder, labelx=r'$\log Ro$', labely='Flares (star day)$^{-1}$')
    search_dragonking(dfc, resfolder)
    search_dragonking(df, resfolder, endname='_distributions_allsingle')
    return
    # Separate results by spectral type
    pars = ['Energy [erg]', 'Peak amplitude', 'Duration [min]', 'FWHM [min]']
    stellar_pars = ['Teff [K]', 'Ro_bonanno', 'logg', 'tessmag']
    stpar_labels = [r'$T_\mathrm{eff}$ [K]', r'$\log Ro$', r'$\log g$', 'TESS mag']

    logs = ['IF', 'SFC', 'CF']
    for pi, par in enumerate(pars):
        for di, dd in enumerate([df[single], df[~single], dfc[complex]]):
            if only_cheopstargets:
                dd = dd[dd['ticname'].isin(cheops_tics)]
                plot_end = '_cheopstargets_min{}flares'.format(min_flares)
            else:
                plot_end = '_min{}flares'.format(min_flares)

            print('\nParameter:', par.split('[')[0])
            print('Dataset: ' + logs[di])

            # Print per spectral type - inly consider stars with at least
            # min_flare occurrences
            if pi == 0:
                dg = dd.groupby(['Spectral type', 'ticname'], observed=True).size()
                print('\nTargets with at least {} flares, Nflares and Ro:'.format(min_flares))
                dg = dg[dg >= min_flares]
                for dline in dg.keys():
                    tline = dd[dd['ticname'] == dline[1]].drop_duplicates(subset='ticname')
                    tline2 = np.sum(dd['ticname'] == dline[1])
                    print(dline, tline2, np.round(tline['Ro_bonanno'].values[0], 2))
            # Now take them all within this constraint
            dff = copy.deepcopy(dd)
            try:
                occurrences = dff.groupby('ticname').size()
            except ValueError:
                set_trace()
            targets = occurrences[occurrences >= min_flares]

            if len(targets) < 1:
                print('PL fit: not enough measurements')
                target_pars = {}
                target_dist = {}
            else:
                target_pars, target_dist = fit_target_distributions(dd, par, \
                    stellar_pars, targets.keys())

            if di == 0:
                target_pars_simple = copy.deepcopy(target_pars)
                target_dist_simple = copy.deepcopy(target_dist)
            elif di == 1:
                target_pars_complex = copy.deepcopy(target_pars)
                target_dist_complex = copy.deepcopy(target_dist)
            else:
                target_pars_complex2 = copy.deepcopy(target_pars)
                target_dist_complex2 = copy.deepcopy(target_dist)

        plot_target_results(target_pars_simple, target_pars_complex, \
            target_dist_simple, target_dist_complex, \
            par, sptype, stellar_pars, stpar_labels, min_flares, \
            resfolder, target_dist3=target_dist_complex2, \
            target_pars3=target_pars_complex2)

        plt.close('all')

    set_trace()

    return

def compare_Ro(df, resfolder):
    plt.figure()
    plt.loglog(df['Ro_cranmer'], df['Ro_bonanno'], '.')
    flag = np.logical_and(df['BP-RP_0'] > 0.55, df['BP-RP_0'] < 1.25)
    plt.loglog(df['Ro_cranmer'][flag], df['Ro_bonanno'][flag], '.', \
            label=r'$0.55 < G_{BP} - G_{RP} < 1.25$')
    plt.legend()
    plt.xlabel(r'$Ro$ (Cranmer+)', fontsize=14)
    plt.ylabel(r'$Ro$ (Bonanno et al. 2025)', fontsize=14)
    x = np.linspace(1e-3, 1., 1000)
    plt.loglog(x, x)
    plt.tight_layout()
    plt.savefig(resfolder + 'Ro_cranmer_vs_metcalfe_flaring.pdf')
    plt.close()

    return

def simulate_pl_size(resfolder):
    '''
    What happens when the data sample is too small?
    (Monte Carlo simulation)
    '''
    a = 1.8
    theo = powerlaw.Power_Law(xmin=1e30, parameters=[a])

    fig, ax = plt.subplots()
    for samplesize in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        print('Sample size:', samplesize)
        alpha = []
        sigma = []
        for iter in range(1000):
            simuldata = theo.generate_random(samplesize)
            fit = powerlaw.Fit(simuldata, verbose=False)
            alpha.append(fit.alpha)
            sigma.append(fit.sigma)
        perc = np.percentile(alpha, [15.9, 50., 84.1])
        dperc = np.diff(perc)[::-1]
        ax.errorbar([samplesize], [perc[1]], yerr=[[dperc[0]], [dperc[1]]], \
            marker='o', color='gray', capsize=2)
    ax.plot([30, 10100], [a, a], 'k--', label='Ground truth')
    plt.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Sample size', fontsize=14)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    plt.savefig(resfolder + 'PL_sample_size.pdf')
    plt.close()

    return

def par_vs_npeaks(df, resfolder):
    '''
    Flare pars vs number of peaks
    '''
    fig, axs = plt.subplots(nrows=2, figsize=(7, 10))
    for pi, pc in enumerate(['ED [s]', 'Energy [erg]']):
        pl = axs[pi].scatter(dfc['Ro_bonanno'], dfc[pc], \
            c=dfc['peaks_per_event'], marker='.')
        axs[pi].set_xscale('log')
        axs[pi].set_yscale('log')
        axs[pi].set_ylabel(pc, fontsize=14)
    axs[1].set_xlabel(r'$Ro$', fontsize=14)
    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pl, cax=cbar_ax, ticks=range(1, 6))
    cbar.set_label('Peaks per event', fontsize=14)
    plt.savefig(resfolder + 'par_vs_npeaks.pdf')
    plt.close('all')

    return

def complex_flare_fraction(df, resfolder):
    '''
    Fraction of complex flares per parameter
    '''
    Ebins = np.logspace(31, 37, 7)
    EDbins = np.logspace(-1, 3.2, 7)
    pars = ['Energy [erg]', 'ED [s]']
    for px, x in enumerate([Ebins, EDbins]):
        fig, ax = plt.subplots()
        for j, ro_i in enumerate(dd['Ro_bins'].unique()):
            y, yerr = [], []
            flag_ro = dd['Ro_bins'] == ro_i
            #label = r'${:.1f} \leq \log Ro \leq ${:.1f}'.format( \
            #    np.log10(ro_i.left), np.log10(ro_i.right))
            if np.sum(flag_ro) == 0:
                continue
            for i, xi in enumerate(x):
                if i == 0:
                    flagE = dfc[flag_ro][pars[px]] <= x[i]
                else:
                    flagE = np.logical_and(dfc[flag_ro][pars[px]] > x[i - 1], \
                                dfc[flag_ro][pars[px]] <= x[i])
                flag = dfc[flag_ro]['peaks_per_event'][flagE] > 1
                if np.sum(flag) + np.sum(~flag) > 1:
                    complex = np.sum(flag)
                    total = np.sum(flag) + np.sum(~flag)
                    y.append(complex/total)
                    yerr.append(np.sqrt( (complex*total**0.5/total**2)**2 \
                            + (complex**0.5/total)**2 ))
                else:
                    y.append(-999.)
                    yerr.append(-999.)
            y = np.array(y)
            yerr = np.array(yerr)
            ax.errorbar(x[y > 0.], y[y > 0.], yerr=yerr[y > 0.], fmt='o-', \
                    capsize=2, label=ro_i)
        ax.set_xscale('log')
        ax.set_ylabel('Complex flare fraction', fontsize=14)
        ax.set_xlabel(pars[px], fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(resfolder + 'complex_flare_fraction_' + pars[px] + '.pdf')
        plt.close()

    return

def monte_carlo_permutation(x_obs, y_obs, sigma_y, n_mc=1000, n_perm=1000):
    '''
    Simulate many data sets within uncertainties and for each derive a
    Spearman correlation coefficient with its p-value.
    '''

    rng = np.random.default_rng()
    p_values = []
    print('Monte-Carlo permutation test...')
    for j in range(n_mc):
        if j % 10 == 0:
            print('Iter:', j)
        # Perturb observed data by measurement uncertainties
        y_perturbed = y_obs + rng.normal(0., sigma_y, len(y_obs))
        # Compute observed Spearman correlation on perturbed data
        obs_rho, _ = stats.spearmanr(x_obs, y_perturbed, alternative='less')
        # Build null distribution by permuting y_perturbed
        perm_rhos = []
        for i in range(n_perm):
            y_perm = rng.permutation(y_perturbed)
            rho_perm, _ = stats.spearmanr(x_obs, y_perm)
            perm_rhos.append(rho_perm)
        perm_rhos = np.array(perm_rhos)

        # Two-sided p-value for this MC iteration
        p_val = np.mean(np.abs(perm_rhos) >= np.abs(obs_rho))
        p_values.append(p_val)

    # Aggregate p-values (median)
    final_p_value = np.median(p_values)

    return final_p_value


def residual_line(par, x, y, yerr):
    '''
    Residuals for a linear fit.
    '''
    residuals = (y - np.polyval(par, x))/yerr
    return residuals

def lin_fit_labels(result, initial_label=''):
    '''
    Put the results of an LMfit minimization in a text string.
    '''
    text = initial_label + '\n'
    for p in result.params:
        text += r'$' + str(p) + r'=${:.1f}$\pm${:.1f}'.format( \
            result.params[p].value, result.params[p].stderr) + '\n'
    return text

def divide_by_previous(x):
    '''
    Divide every element of an array by its previous.
    Return size: len(array) - 1
    '''
    x_shifted = np.roll(x[:-1], 1)
    x_shifted[x_shifted == 0] = np.nan
    return x[:-1]/x_shifted

def robust_linear_fit(x, y, uncert=1., sigma_iter=3., niter=10):
    '''
    Perform a sigma_iter-sigma clipping at each iteration.
    '''
    from astropy.stats import sigma_clip

    x_new = np.copy(x)
    y_new = np.copy(y)

    pfit = lmfit.Parameters()
    pfit.add('a', vary=True, value=0.)
    pfit.add('b', vary=True, value=0.)
    for n in range(niter):
        try:
            result = lmfit.minimize(residual_line, pfit, \
                args=(x_new, y_new, uncert))
        except ValueError:
            if n == 0:
                set_trace()
            else:
                break
        line = np.polyval(result.params, x_new)
        flag = sigma_clip(abs(y_new - line), sigma=sigma_iter).mask
        if np.sum(flag) == 0:
            break
        x_new = x_new[~flag]
        y_new = y_new[~flag]

    print('\nRobust linear fit: {} iterations'.format(n))
    print('Rejected points: {} %'.format((1. - len(x_new)/len(x))*100.))

    return result.params, x_new, y_new

def filter_df(df, Teff, radius, Prot, FAP, Tmag):
    '''
    Only works with upper limits by now.
    '''
    return np.logical_and.reduce((df['Teff [K]'] <= Teff, df['radius'] <= radius, \
                df['Prot'] <= Prot, df['FAP'] <= FAP, df['tessmag'] <= Tmag))

def get_Rsign(R):
    if np.sign(R) > 0:
        return 'not supported'
    else:
        return r'supported'

def obs_time_per_target(df, resfolder):
    '''
    Observation time per target
    '''
    fig, ax = plt.subplots()
    ax.hist(df.drop_duplicates(subset='ticname')['time_on_target'], log=True, \
            histtype='step', color='k')
    ax.set_xlabel('Time on target [days]', fontsize=14)
    ax.set_ylabel('Targets', fontsize=14)
    plt.savefig(resfolder + 'time_on_target.pdf')
    plt.close()

    return

def simple_complex_fit(df1, df2, df3, par1, par2, label_par1, label_par2, \
        resfolder, logx=False, logy=False, deg=1):
    '''
    Compare fits of flare properties vs stellar pars for simple and complex
    flares. The dependent variable is assumed to be in log units.
    Confidence intervals are obtained via pair bootstrapping

    deg (int): degree for polynomial fits
    '''

    dftemp = pd.concat([df1, df2, df3])
    sns.jointplot(dftemp, x=par1, y=par2, hue='Flare type', kind='hist', \
            alpha=0.5, palette='colorblind', \
            marginal_kws=dict(fill=False, element='step'))

    labels = ['dataset 1', 'dataset 2', 'dataset 3']
    for i, dd in enumerate([df1, df2, df3]):
        x = dd[par1]
        y = dd[par2]
        pfit = lmfit.Parameters()
        pfit.add('a', vary=True, min=-100., max=100.)
        pfit.add('b', vary=True, min=-100., max=100.)
        rng = np.random.default_rng()
        a, b = [], []
        for j in range(1000):
            xi, yi = rng.choice([x, y], size=len(x), axis=1)
            ai, bi = np.polyfit(xi, yi, deg=1)
            a.append(ai)
            b.append(bi)
        perc = [15.9, 50., 84.1]
        pnames = ['a', 'b']
        meds = []
        print('\n', par2, 'vs', par1, 'for', labels[i], ':')
        for pi, parfit in enumerate([a, b]):
            percs = np.percentile(parfit, perc)
            print(pnames[pi], '{:.3f}+{:.3f}-{:.3f}\n'.format(percs[0], \
                        np.diff(percs)[1], np.diff(percs)[0]))
            meds.append(percs[1])
        xTh = np.linspace(x.min(), x.max(), 100)
        plt.plot(xTh, np.polyval(meds, xTh))

        # Compute correlation coefficient for the subset without outliers
        R, p = stats.spearmanr(x, y)
        print('Spearman R, p:', R, p)

    plt.xlabel(label_par1, fontsize=14)
    plt.ylabel(label_par2, fontsize=14)
    plt.xlim(dftemp[par1].min() - 0.05, dftemp[par1].max() + 0.05)
    plt.ylim(dftemp[par2].min() - 0.05, dftemp[par2].max() + 0.05)
    plt.tight_layout()
    plt.savefig(resfolder + par2.split(' [')[0] + '_vs_' + par1 + '.pdf')
    plt.close()

    return

def fit_target_distributions(df, par, stellar_pars, targets, bootstrap=True):
    '''
    Fit power law for all objects with at least min_flares.
    Bootstrapped estimates are carried out as well, assuming data has
    uncertainties.

    Return
    ----------
    target_distrib: list of target distributions (for plotting purposes)
    '''
    if len(targets) == 0:
        print('No targets with enough flares for individual PDF fit')

    target_pars = {}
    for tgp in ['alpha', 'sigma', 'nobs', 'maxval', 'minval', 'R_tg', 'p_tg', \
                'deltabic_tg']:
        target_pars[tgp] = []
    for tgp in stellar_pars:
        target_pars[tgp] = []
    if bootstrap:
        for tgp in ['alpha_bootstrap', 'sigma_bootstrap']:
            target_pars[tgp] = []

    target_distrib = []
    for tgi, tg in enumerate(targets):
        flag_tg = df['ticname'] == tg
        fit_tg = powerlaw.Fit(df[par][flag_tg], verbose=False)
        x, y = fit_tg.ccdf(original_data=False)
        target_distrib.append([x, y])

        # Fit with a truncated power law
        R_tg, p_tg = fit_tg.distribution_compare('power_law', \
            'truncated_power_law', normalized_ratio=True)
        deltabic_tg = compare_bic(fit_tg, 'power_law', 'truncated_power_law')
        print(tg, ': PL vs Truncated PL Delta BIC =', deltabic_tg)
        target_pars['R_tg'].append(R_tg)
        target_pars['p_tg'].append(p_tg)
        target_pars['alpha'].append(fit_tg.alpha)
        target_pars['sigma'].append(fit_tg.sigma)
        target_pars['nobs'].append(np.sum(flag_tg))
        target_pars['maxval'].append(df[par][flag_tg].max())
        target_pars['minval'].append(fit_tg.xmin)
        target_pars['deltabic_tg'].append(deltabic_tg)
        for pp in stellar_pars:
            target_pars[pp].append(np.median(df[pp][flag_tg]))

        # See impact of data uncertainties via bootstrap
        #if 'Energy' in par:
        if bootstrap:
            alphas, sigmas = [], []
            print('Bootstrap for target:', tg, '...')
            distrib = df[par][flag_tg]
            rng = np.random.default_rng()
            for i in range(1000):
                newdata = rng.choice(distrib, size=len(distrib))
                fit_th = powerlaw.Fit(newdata, verbose=False)
                alphas.append(fit_th.alpha)
                sigmas.append(fit_th.sigma)
            target_pars['alpha_bootstrap'].append(np.mean(alphas))
            target_pars['sigma_bootstrap'].append(np.std(alphas))

    for tgp in target_pars.keys():
        target_pars[tgp] = np.array(target_pars[tgp])
        # Convert to cgs
        if 'FLUX' in tgp:
            target_pars[tgp] *= 1e3

    return target_pars, target_distrib

def plot_target_results(target_pars1, target_pars2, target_dist1, \
            target_dist2, par, sptype, \
            stellar_pars, stpar_labels, min_flares, resfolder, \
            cheops_targets=False, target_dist3={}, target_pars3={}, \
            bootstrap=True):
    '''
    One plot for simple and complex flare cases

    Parameters
    ----------
    bootstrap: plot bootstrapped statistics or n**-1/2 ones
    '''

    palpha = 'alpha'
    psigma = 'sigma'
    if bootstrap:
        palpha += '_bootstrap'
        psigma += '_bootstrap'

    fig, ax = plt.subplots()
    linestyles = ['-', ':', '--']
    labels = ['IF', 'SFC', 'CF']
    if target_dist1 != {} or target_dist2 != {} or target_dist3 != {}:
        for tdi, td in enumerate([target_dist1, target_dist2, target_dist3]):
            for jj, (x, y) in enumerate(td):
                if jj == 0:
                    ax.loglog(x, y, linestyle=linestyles[tdi], color='k', \
                    label=labels[tdi])
                else:
                    ax.loglog(x, y, linestyle=linestyles[tdi], color='k')
        ax.set_xlabel(par, fontsize=14)
        ax.set_ylabel(r'CCDF', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(resfolder + par + '_distributions.pdf')
        plt.close()

    markers = ['o', 's', '^', 'v']
    labels = ['IF', 'SFC', 'CF']
    colors = ['royalblue', 'orange', 'green']

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    #if 'Energy' in par:
    fig5, ax5 = plt.subplots()
    for tpi, target_pars in enumerate([target_pars1, target_pars2, target_pars3]):

        print('Results for distribution:', labels[tpi])

        if target_pars == {} or len(target_pars[palpha]) < 3:
            print('No available targets for individual PL fits')
            continue

        # Sort results by spectral type
        tlim = [0., 4200., 5300., 5950., 7200.]
        ddd = pd.DataFrame.from_dict(target_pars)
        ddd.sort_values('Teff [K]', inplace=True)
        ddd['Spectral type'] = pd.cut(ddd['Teff [K]'], bins=tlim, labels=sptype)
        ddd['q'] = np.log10(ddd['maxval']/ddd['minval'])

        for sti, stpar in enumerate(stellar_pars):
            # Scatter plots + errorbars: see
            # https://how2matplotlib.com/matplotlib-errorbar-color.html
            if stpar != 'Ro_bonanno':
                x = ddd[stpar]
            else:
                x = np.log10(ddd[stpar])
            y = ddd[palpha]
            yerr = ddd[psigma]

            # Pearson R
            R, p = stats.spearmanr(x, y)

            # Fitted flare vs stellar parameters
            pfit = lmfit.Parameters()
            pfit.add('a', vary=True, min=-10., max=10, value=0.)
            pfit.add('b', vary=True, min=-10, max=10, value=0.)
            #pfit.add('c', vary=False, min=-10, max=10, value=0.)
            result = lmfit.minimize(residual_line, pfit, \
                       args=(x, y, yerr), calc_covar=True)

            ax1.plot(np.sort(x), (y - result.residual*yerr)[np.argsort(x)], \
                linestyles[tpi], c=colors[tpi], \
                label=lin_fit_labels(result, labels[tpi]) \
                        + r'$p$-value = {:.2}'.format(p))

            ax1.errorbar(x, y, yerr=yerr, fmt=markers[tpi], c=colors[tpi], \
                        capsize=2)
            ax1.set_xlabel(stpar_labels[sti], fontsize=14)
            ax1.set_ylabel(par.split('[')[0] + r' $\alpha$', fontsize=14)
            ax1.legend()
            if cheops_targets:
                plot_end = '_cheopstargets_min{}flares'.format(min_flares)
            else:
                plot_end = '_min{}flares'.format(min_flares)
            foutname = resfolder.replace('cluster', 'analysis') \
                   + par.split('[')[0] + '_alpha_vs_' \
                   + stpar.split('[')[0] + plot_end + '.pdf'

            # What is the inertial range related to?
            if stpar != 'Ro_bonanno':
                x = ddd[stpar]
            else:
                x = np.log10(ddd[stpar])
            R, pv = stats.spearmanr(x, ddd['q'])
            print('Spearman corrcoeff between ' + stpar \
                    + ' and inertial range: ' + str(R) + ' ' + str(pv))

        newname = foutname.replace(stpar.split('[')[0], 'change')

        # Number of events
        x = ddd['nobs']
        y = ddd[palpha]
        R, p = stats.spearmanr(x, y)
        ax2.scatter(x, y, c=colors[tpi], label=labels[tpi] \
                        + ': $p-$value={:.2f}'.format(p))
        ax2.legend()
        ax2.set_xlabel('Number of flares', fontsize=14)
        ax2.set_ylabel(par.split('[')[0] + r' $\alpha$', fontsize=14)

        # Minimum value of inertial range
        x = ddd['minval']
        y = ddd[palpha]
        ax3.scatter(x, y, c=colors[tpi], label=labels[tpi])
        ax3.set_xscale('log')
        ax3.legend()
        ax3.set_xlabel(r'$x_\mathrm{min}$', fontsize=14)
        ax3.set_ylabel(r'$\alpha$', fontsize=14)
        fig3.tight_layout()

        # Inertial range
        y = ddd[palpha]
        yerr = ddd[psigma]
        R, pv = stats.spearmanr(ddd['q'], y)
        print('Spearman corrcoeff between alpha and inertial range: ' \
                        + str(R) + ' ' + str(pv))
        R, pv = stats.spearmanr(ddd['q'], ddd['nobs'])
        print('Spearman corrcoeff between nobs' + ' and inertial range: ' \
                        + str(R) + ' ' + str(pv))
        #pfit['c'].vary = True
        result = lmfit.minimize(residual_line, pfit, \
            args=(ddd['q'], y, yerr), calc_covar=True)

        for spi, sp in enumerate(sptype):
            flag = ddd['Spectral type'] == sp
            if np.sum(flag) == 0:
                continue
            ax4.errorbar(ddd['q'][flag], y[flag], yerr=yerr[flag], \
                fmt=markers[spi], c=colors[tpi], label=sp + ' ' + labels[tpi], \
                capsize=2)
        fitpars = [result.params['a'], result.params['b']]#, result.params['c']]
        xTh = np.linspace(ddd['q'].min(), ddd['q'].max(), 100)
        print(par, 'fit alpha vs q:', lin_fit_labels(result, labels[tpi]))
        if 'Energy' in par:
            ax4.plot(xTh, np.zeros(len(xTh)) + 2., 'k--')
        ax4.plot(xTh, np.polyval(fitpars, xTh), c=colors[tpi])#, label='$p$' + pv)
        ax4.set_xlabel(r'$q$', fontsize=14)
        ax4.set_ylabel(par.split('[')[0] + r' $\alpha$', fontsize=14)
        fig4.tight_layout()

        # Print estimate
        if sti == 0:
            print(labels[tpi], 'Estimate for alpha at max x12: ', \
                        np.polyval(fitpars, ddd['q'].max()))

        if 'alpha_bootstrap' in ddd.keys() and not bootstrap:# and 'Energy' in par:
            labx = ['$q$'] #Sample size
            if tpi == 0:
                label1 = 'Bootstrapped'
                label2 = 'Formal uncertainties'
            else:
                label1 = ''
                label2 = ''
            from matplotlib import ticker
            for pi, px in enumerate([ddd['q']]):#, ddd['nobs']]):
                ax5.errorbar(px, ddd['alpha_bootstrap'], fmt='bo', \
                    yerr=ddd['sigma_bootstrap'], capsize=2, label=label1)
                ax5.errorbar(px, ddd['alpha'], fmt='o', color='orange', \
                    capsize=2, yerr=ddd['sigma'], label=label2)
                ax5.legend()
                ax5.set_xlabel(labx[pi], fontsize=14)
                ax5.set_ylabel(r'$\alpha$', fontsize=14)
            ax5.xaxis.set_minor_formatter(ticker.ScalarFormatter())

        if tpi == 0:
            dtot = copy.deepcopy(ddd)
        else:
            dtot = pd.concat([dtot, ddd])

    # Compute overall p-value for all par vs q pairs
    spear = monte_carlo_permutation(dtot['q'], dtot['alpha'], dtot['sigma'])
    pv = ' = ' + str(np.round(spear, 3))
    ax4.plot([], [], 'k', label='$p$' + pv)
    ax4.legend()

    if bootstrap:
        endlabel = '_bootstrap'
    else:
        endlabel = ''
    fig1.savefig(foutname)
    fig2.savefig(newname.replace('change', 'nflares' + endlabel))
    fig3.savefig(newname.replace('change', 'xmin' + endlabel))
    fig4.savefig(newname.replace('change', 'range' + endlabel))
    fig5.savefig(newname.replace('change', 'bootstrap'))

    plt.close('all')

    return

def powerlaw_pdf(x, xmin, alpha):
    return (alpha - 1.)*xmin**(alpha - 1.)*x**(-alpha)

def exponential_pdf(x, xmin, lambdap):
    return lambdap*np.exp(lambdap*xmin)*np.exp(-lambdap*x)

def lognormal_pdf(x, xmin, mu, sigma):
    C = np.sqrt(2./(np.pi*sigma**2)) \
            *(special.erfc((np.log(xmin) - mu)/(2**0.5*sigma)))**-1
    F = 1./x*np.exp(-(np.log(x) - mu)**2/(2.*sigma**2))
    return C*F

def truncated_powerlaw_pdf(x, xmin, lambdap):
    C = lambdap**(1. - alpha)/special.gamma(1. - alpha, lambdap*xmin)

def compute_tau_conv_metcalfe(bp_rp):
    '''
    See https://ui.adsabs.harvard.edu/abs/2024RNAAS...8..260M/abstract
    '''
    params_metcalfe = [-26.1, 204.4, -106.6]
    return np.polyval(params_metcalfe, bp_rp)

def compute_tau_conv_cranmer(teff):
    '''
    See https://ui.adsabs.harvard.edu/abs/2011ApJ...741...54C/abstract,
    eq. 36
    '''
    return 314.24*np.exp(-1.*(teff/1952.5) - (teff/6250.)**18) + 0.002

def compute_tau_conv_bonanno(bp_rp_0, teff, logg, feh):
    '''
    See Bonanno+2025, https://iopscience.iop.org/article/10.3847/1538-4357/ae12f2,
    eq. 16. Using Table 2 results for S1 (MS and early subgiant stars)

    bp_rp_0 is corrected for reddening: see Stassun+2019
    (https://iopscience.iop.org/article/10.3847/1538-3881/ab3467), or also
    Casagrande & VandenBerg 2018 for Gaia DR2
    '''
    alpha1, alpha2, alpha3, alpha4, ln_beta = -3.1, 0.55, 0.41, -12.71, 114.
    ln_tau = ln_beta + alpha1*bp_rp_0 + alpha2*logg + alpha3*feh \
                + alpha4*np.log(teff)
    return np.exp(ln_tau)


#def bic_distribution(fitd, distribution_type):
#    '''
#    Computes the Bayesian Information Criterion for a distribution fitted with
#    powerlaw.
#    '''
#    lkf = powerlaw.likelihood_function_generator(distribution_type, False, \
#                    xmin=fitd.xmin, xmax=fitd.xmax)
#    set_trace()
#    if distribution_type == 'power_law':
#        lk = lkf([fitd.power_law.alpha], fitd.data)
#        npars = 1
#    elif distribution_type == 'exponential':
#        lk = lkf([fitd.exponential.alpha], fitd.data)
#        npars = 1
#    elif distribution_type == 'lognormal':
#        lk = lkf([fitd.lognormal.parameter1, fitd.lognormal.parameter2], \
#            fitd.data)
#        npars = 2
#    elif distribution_type == 'truncated_power_law':
#        lk = lkf([fitd.truncated_power_law.parameter1, \
#            fitd.truncated_power_law.parameter2], fitd.data)
#        npars = 2
#
#   bic = npars*np.log(len(fitd.data)) - 2.*np.sum(np.log(lk))
#   return bic

def bic_distribution(fitd, distribution_type):
    '''
    Following the logL definition in appendix B of Clauset et al. (2009).
    '''
    x = fitd.data
    n = len(x)
    xmin = fitd.xmin

    if distribution_type == 'power_law':
        alpha = fitd.power_law.alpha
        npars = 1
    elif distribution_type == 'truncated_power_law':
        alpha = fitd.truncated_power_law.parameter1
        npars = 2

    logL = n*np.log(alpha - 1.) - n*np.log(xmin) - alpha*np.sum(np.log(x/xmin))
    bic = npars*np.log(n) - 2.*logL

    return bic

def compare_bic(distribution, dist1_type, dist2_type):
    '''
    Compare different models on the same distribution.
    '''
    bic1 = bic_distribution(distribution, dist1_type)
    bic2 = bic_distribution(distribution, dist2_type)
    return bic1 - bic2

def compare_distributions(dist_fit):
    '''
    Compare R and p for all distributions supported by powerlaw.
    '''
    Rtr, ptr = dist_fit.distribution_compare('power_law', \
            'truncated_power_law', normalized_ratio=True)
    Rlog, plog = dist_fit.distribution_compare('power_law', 'lognormal', \
            normalized_ratio=True)
    Rexp, pexp = dist_fit.distribution_compare('power_law', 'exponential', \
            normalized_ratio=True)
    Rlogexp, plogexp = dist_fit.distribution_compare('lognormal', \
            'exponential', normalized_ratio=True)
    Rlogtr, plogtr = dist_fit.distribution_compare('lognormal', \
            'truncated_power_law', normalized_ratio=True)
    Rexptr, pexptr = dist_fit.distribution_compare('exponential', \
            'truncated_power_law', normalized_ratio=True)

    table = [['comparison', 'R', 'p'], \
             ['PL vs truncated PL', Rtr, ptr], \
             ['PL vs logn', Rlog, plog], \
             ['PL vs exp', Rexp, pexp], \
             ['logn vs exp', Rlogexp, plogexp], \
             ['logn vs truncated PL', Rlogtr, plogtr], \
             ['exp vs truncated PL', Rexptr, pexptr]]

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    return

def segment_regression(df, par, resfolder, labelx='', labely=''):
    '''
    Segmented regression to ED vs Ro
    '''
    import piecewise_regression as pr

    ftd = ['IF', 'SFC', 'CF']
    fits = []
    for ift, ft in enumerate(ftd):
        flag = df['Flare type'] == ft
        x = df[flag]['log_Ro'].tolist()
        y = df[flag][par].tolist()
        print(par, 'vs', 'log Ro -', ft)
        pw_fit = pr.Fit(x, y, n_breakpoints=1)
        fits.append(pw_fit)
        print(pw_fit.summary())

    sns.jointplot(df, x='log_Ro', y=par, hue='Flare type', alpha=0.5, \
                kind='hist', fill=True, hue_order=ftd, \
                marginal_kws=dict(fill=False, element='step'))
    xx = np.linspace(df['log_Ro'].min(), df['log_Ro'].max(), 1000).tolist()
    colors = ['royalblue', 'orange', 'g']
    for fi, ff in enumerate(fits):
        plt.plot(xx, ff.predict(xx), c=colors[fi])

    if labelx != '':
        plt.xlabel(labelx, fontsize=14)
    if labely != '':
        plt.ylabel(labely, fontsize=14)
    plt.xlim(df['log_Ro'].min() - 0.05, df['log_Ro'].max() + 0.05)
    plt.ylim(df[par].min() - 0.05, df[par].max() + 0.05)
    plt.tight_layout()
    plt.show()
    set_trace()
    plt.savefig(resfolder + par + '_vs_Ro_segmented.pdf')

    return

def search_dragonking(dfc, resfolder, par='Energy [erg]', min_flares=100, \
            endname='_distributions_single+complex'):
    '''
    Plot energy distributions for targets with > 100 events and search for DK
    events. Measurement uncertainties are neglected here.
    '''

    # For info
    dfc.sort_values(by='ticname', inplace=True)
    dg = dfc.groupby(['ticname', 'Spectral type']).size()
    dg = dg[dg >= min_flares]
    print('Targets for DK events search:')
    print(dg)

    dg = dfc.groupby(['ticname']).size()
    dg = dg[dg >= min_flares]
    for i, tg in enumerate(dg.keys()):

        flag = dfc['ticname'] == tg
        print(tg + ' SpType: ' \
                + dfc['Spectral type'][flag].drop_duplicates().values[0])

        fit_nm = powerlaw.Fit(dfc[flag]['Energy [erg]'], verbose=False)
        label = tg # + ' (' + str(dg[tg]) + ')'
        if i == 0 :
            fig_nm = fit_nm.plot_pdf(label=label, original_data=False)
        else:
            fit_nm.plot_pdf(ax=fig_nm, label=label, original_data=False)
        deltabic = compare_bic(fit_nm, 'power_law', 'truncated_power_law')
        print('Delta BIC PL-truncated PL = {:.2f}'.format(deltabic))
        fit_nm.truncated_power_law.plot_pdf(ax=fig_nm, color='k', linestyle='--', \
                alpha=0.5)
    plt.legend()
    plt.xlabel('Energy [erg]', fontsize=14)
    plt.ylabel('CCDF', fontsize=14)
    plt.tight_layout()
    plt.savefig(resfolder + par + endname + '.pdf')
    plt.close()

    pvals = []
    labels = []
    dfc.sort_values('Energy [erg]', inplace=True)
    for i, tg in enumerate(dg.keys()):
        flag = dfc['ticname'] == tg
        fit_nm = powerlaw.Fit(dfc[flag]['Energy [erg]'], verbose=False)
        x, y = fit_nm.ccdf(original_data=False)
        yth = fit_nm.power_law.ccdf()
        part1 = len(y) - 10#int(len(y)*0.9)
        ydiff = y - yth
        #flag = ydiff[part1:] > 0.
        obs_diff, pval = permutation_test_variance_diff(ydiff[:part1], \
                        ydiff[part1:])#[flag])
        pvals.append(pval)
        if pval < 0.05:
            print('DK events candidate with p < 0.05:', tg)
        labels.append(tg)

    dfres = pd.DataFrame.from_dict({'pvals':pvals, 'tgs':labels})
    dfres.sort_values(by='pvals', inplace=True)
    if endname == '_distributions_single+complex':
        label=''
    else:
        label = 'IF + SFC'
    plt.plot(dfres['pvals'], dfres['tgs'], 'ko', mfc='white', \
                label=label)
    plt.xlabel('Permutation test $p$-value', fontsize=14)

    # Save results for later use, or use them
    fout = resfolder + 'distributions_single+complex.csv'
    if endname == '_distributions_single+complex':
        dfres.to_csv(fout, index=False)
    else:
        dfcomp = pd.read_csv(fout)
        plt.plot(dfcomp['pvals'], dfcomp['tgs'], 'kx', label='IF + CF')
        dfres.rename(columns={'pvals':'pvals_IFCF'}, inplace=True)
        dfcomp.rename(columns={'pvals':'pvals_IFSFC'}, inplace=True)
        dftot = pd.merge(dfres, dfcomp, on='tgs', how='outer').dropna()
        for i, t in enumerate(dftot['tgs']):
            flag = dftot['tgs'] == t
            plt.plot([dftot['pvals_IFCF'][flag], dftot['pvals_IFSFC'][flag]], \
                        [t, t], 'k')
    #plt.plot([0.05, 0.05], [0, len(labels)], 'r--', label=r'$p=0.05$')
    plt.legend()#loc='lower right')
    plt.tight_layout()
    plt.savefig(resfolder + par + endname + '_diff.pdf')
    plt.close()

    return

def permutation_test_variance_diff(part1, part2, n_perm=10000):
    '''
    Used to test the p-value of variance differences for DK candidates.
    '''
    rng = np.random.default_rng()
    combined = np.concatenate([part1, part2])
    n1 = len(part1)
    flag = part2 > 0.
    observed_diff = np.mean(part2)/np.std(part1, ddof=1)
    count = 0

    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_part1 = combined[:n1]
        perm_part2 = combined[n1:]
        flag = perm_part2 > 0.
        try:
            perm_diff = np.mean(perm_part2[flag])/np.std(perm_part1, ddof=1)
        except ValueError:
            # no values > 0 in part 2
            continue
        if perm_diff >= observed_diff:
            count += 1

    p_value = count / n_perm
    return observed_diff, p_value

def waiting_time_distribution(df, resfolder, fltypes, lab):
    '''
    Parameters
    ----------
    fltypes: used to save plot
    lab: labels to print in legend percentage
    '''

    ccs = ['b', 'orange']
    fig_wt, ax_wt = plt.subplots()
    ymax = -np.inf
    for i, ro_i in enumerate(df['Ro_bins'].unique()):

        flag = df['Ro_bins'] == ro_i
        if np.sum(flag) < 10:
            continue
        distrib = df[flag]['waiting_time'].dropna()
        if len(distrib) < 10:
            print(ro_i, 'Sample size:', len(distrib))
            continue

        # How many flares have more than one peak?
        ncomplex = np.sum(df[flag]['peaks_per_event'] > 1)
        totflares = np.sum(flag)
        perc_complex = ncomplex/totflares*100.
        label = ' ({:.1f}%'.format(perc_complex) + ' ' + lab + ')'

        # Try two exponentials
        distrib = distrib.tolist()

        fit_ew = stats.exponweib.fit(distrib)
        print(ro_i, ', Fit EW:', fit_ew)
        fit_e = stats.expon.fit(distrib)
        print(ro_i, ', Fit exp:', fit_e)

        figt, axt = plt.subplots()
        counts, bins, _ = plt.hist(distrib, bins=14, density=True)
        plt.close()
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))
        yy = ax_wt.hist(distrib, bins=logbins, log=True, density=True, \
                    histtype='step', color=ccs[i], label=ro_i + label)
        if yy[0].max() > ymax:
            ymax = yy[0].max()
        xth = lambda a, x: stats.exponweib.ppf(a, x[0], x[1], loc=x[2], \
                                          scale=x[3])
        x_ew = np.linspace(xth(0.001, fit_ew), xth(0.999, fit_ew), 300000)
        pdf_ew = stats.exponweib.pdf(x_ew, fit_ew[0], fit_ew[1], \
                loc=fit_ew[2], scale=fit_ew[3])

        xth = lambda a, x: stats.expon.ppf(a, loc=x[0], scale=x[1])
        x_e = np.linspace(xth(0.001, fit_e), xth(0.999, fit_e), 300000)
        pdf_e = stats.expon.pdf(x_e, loc=fit_e[0], scale=fit_e[1])#*scaling

        ax_wt.plot(x_ew, pdf_ew, '--', color=ccs[i])
        ax_wt.plot(x_e, pdf_e, ':', color=ccs[i])

    ax_wt.set_xscale('log')
    ax_wt.set_yscale('log')
    ax_wt.set_ylim(1e-6, ymax*2.5)
    ax_wt.set_xlim(logbins[0]*0.5, logbins[-1]*1.5)
    ax_wt.plot([], [], 'k--', label='Expon. Weibull')
    ax_wt.plot([], [], 'k:', label='Exponential')
    ax_wt.legend()
    ax_wt.set_xlabel('Waiting time [min]', fontsize=14)
    ax_wt.set_ylabel(r'$P(\Delta t)$', fontsize=14)
    plt.tight_layout()
    plt.savefig(resfolder + 'waiting_time_distribution_' + fltypes + '.pdf')
    plt.close('all')

    return

def heatmap(df, resfolder):
    '''
    Spearman correlation matrix
    '''
    # Spearman correlation matrix
    corr = df[['Peak amplitude', 'Impulsiveness [min$^{-1}$]', \
        'Duration [min]', 'ED [s]', 'Energy [erg]', 'Teff [K]', 'radius', \
        'Ro_metcalfe']].corr(method='spearman')
    ticklabels = [ 'Peak amplitude', 'Impulsiveness', 'Duration', 'ED', 'Energy', \
                r'$T_\mathrm{eff}$', r'$R_\star$', r'$Ro$']
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, xticklabels=ticklabels, yticklabels=ticklabels, \
                annot=True, cbar=False, annot_kws={'fontsize':14})
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(resfolder + 'par_correlations_FGKM' + labc[di] + '.pdf')
    plt.close('all')

    return

def HR_diagram(df, resfolder):

    fig, ax = plt.subplots()
    lum = df['Peak luminosity [erg s$^{-1}$]']/df['Peak amplitude'] \
            / constants.L_sun.to(units.erg/units.s)
    pl = ax.scatter(df['Teff [K]'], lum, c=df['Ro_bonanno'], \
            marker='.', norm=matplotlib.colors.LogNorm())
    cbar = fig.colorbar(pl)
    cbar.set_label(r'$Ro$', fontsize=14)
    ax.set_xlabel(r'$T_\mathrm{eff}$ [K]', fontsize=14)
    ax.set_ylabel(r'$L/L_\odot$', fontsize=14)
    ax.set_yscale('log')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(resfolder + 'lum_vs_teff.pdf')
    plt.close()

    return

def sky_positions(df, resfolder):
    '''
    Plot target coordinates
    '''
    import astropy.coordinates as coord
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="mollweide")
    #ax2 = fig.add_subplot(211, projection="mollweide")
    ra = coord.Angle(df['ra'].values*units.degree)
    ra = ra.wrap_at(180*units.degree)
    dec = coord.Angle(df['dec'].values*units.degree)
    ax.scatter(ra.radian, dec.radian, marker='.', alpha=0.5)

    ra = coord.Angle(nfl['RA_OBJ'].values*units.degree)
    ra = ra.wrap_at(180*units.degree)
    dec = coord.Angle(nfl['DEC_OBJ'].values*units.degree)
    ax.scatter(ra.radian, dec.radian, label='Flaring', marker='.', \
            alpha=0.1)
    plt.savefig(resfolder + 'stars_sky.pdf')
    plt.close()

    return

def energy_spotarea(df, resfolder):
    '''
    # Flare energy vs spot filling factor (as per Herbst+ and ref. within).
    # We are assigning all photometric variability to a single spot for every
    # star
    '''

    sun_active = np.genfromtxt(resfolder + '../literature_data/herbst/' \
        + 'Sun_Kepler_flares/active_region_size_flare_class_1996-2006.txt', \
        dtype=float, missing_values="--", filling_values=np.nan)
    sun_size = sun_active[:,0]*1e-6
    sun_energy = (sun_active[:,1]/1e4)*1e32
    # corrected (*1/0.7) accoding to Hudson et al. (2024)
    sun_energy /= 0.7

    # Secondary axis
    def norm_Aspot_to_Mx(x):
        return (x*sun_area.value)*3000.

    def Mx_to_norm_Aspot(x):
        return (x/sun_area.value)/3000.#/1000.

    f = 0.1
    AspotTh = np.logspace(-5., -0.25)*sun_area
    ypos = [10**30.25/3., 10**30.85/3., 10**31.45/3.]
    fig, ax = plt.subplots()
    for Bi, B in enumerate([500, 1000, 2000]): # B in Gauss
        Eflare = 7e32*(f/0.1)*(B/1e3)**2*(AspotTh/(2.*sun_area)/1e-3)**1.5
        ax.loglog(AspotTh/sun_area, Eflare, 'k')
        ax.text(10**-4.5, ypos[Bi], str(B) + ' G', rotation=32)

    # Add Okamoto+2021 data
    oka = fits.open(resfolder + '../literature_data/okamoto2021.fits')
    tspot_coeffs = [-3.58e-5, 1.0188, 239.3]
    newtspot = np.polyval(tspot_coeffs, oka[1].data['Teff'])
    newAspot = oka[1].data['Amp']*1e-6 \
                *oka[1].data['Rstar']**2 \
                *(1. - (newtspot/oka[1].data['Teff'])**4)**-1
                # division by 2*pi*Rsun, where Rsun == 1 is implicit
                #2.*np.pi*oka[1].data['Rstar'] \

    # Secondary axis
    secax = ax.secondary_xaxis('top', \
                functions=(norm_Aspot_to_Mx, Mx_to_norm_Aspot))
    secax.set_xscale('log')
    secax.set_xlabel('Magnetic flux ($B=3000$ G) [Mx]', fontsize=14)

    # Merging dataframes
    label_oka = 'Kepler superflares'
    label_notsu = 'Solar flares'
    dfthis = copy.deepcopy(df)
    dfthis[r'Spotted area [$A_\odot$]'] = df['Aspot [Asun]']
    df_oka = {r'Spotted area [$A_\odot$]':newAspot, 'Energy [erg]':oka[1].data['E']}
    df_oka = pd.DataFrame.from_dict(df_oka)
    df_notsu = {r'Spotted area [$A_\odot$]':sun_size, 'Energy [erg]':sun_energy}
    df_notsu = pd.DataFrame.from_dict(df_notsu)
    vv = ['This study', label_oka, label_notsu]
    for i, dd in enumerate([dfthis, df_oka, df_notsu]):
        dd['Dataset'] = [vv[i]]*len(dd)
    dfplot = pd.concat([dfthis, df_oka, df_notsu])
    sns.histplot(dfplot, x=r'Spotted area [$A_\odot$]', y='Energy [erg]', \
        hue='Dataset', ax=ax, palette='colorblind')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Spotted area [$A_\odot$]', fontsize=14)
    ax.set_ylabel('Energy [erg]', fontsize=14)
    ax.set_ylim(1e28, 2e37)
    #ax.set_xlim(6e-6, 0.5)
    plt.savefig(resfolder + 'energy_vs_ffactor.pdf')
    plt.close('all')

    return

def flaring_vs_nonflaring_stars(df, nfl, resfolder):

    nfl_unique = nfl.drop_duplicates(subset='ticname')
    df_unique = df.drop_duplicates(subset='ticname')

    labels = [r'$T_\mathrm{eff}$ [K]', r'$Ro$', r'Spotted area [$A_\odot$]']

    def forward(x):
        return x/x.max()*100.
    def inverse(x):
        return x*x.max()/100.

    for pi, p in enumerate(['Teff [K]', 'Ro_bonanno', 'Aspot [Asun]']):
        if pi == 0:
            bin_edges = 7
        elif pi == 1:
            bin_edges = np.logspace(-3., 1., 9)
        elif pi == 2:
            bin_edges = np.logspace(-3.5, 0., 10)
        hh, bin_edges = np.histogram(nfl_unique[p], bins=bin_edges)
        hh2, bin_edges = np.histogram(df_unique[p], bins=bin_edges)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist([nfl_unique[p], df_unique[p]], \
                    label=['Not flaring', 'Flaring'], \
                    color=['lightgrey', 'dimgrey'], bins=bin_edges, log=True)
        ax.plot(bin_edges[:-1] + 0.5*np.diff(bin_edges), hh2/(hh + hh2)*100., 'mo-')
        #        label='Flaring percentage')
        ax.set_xlabel(labels[pi], fontsize=14)
        ax.set_ylabel('Stars', fontsize=14)
        secax = ax.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel('Flaring percentage', fontsize=14)
        ax.legend()
        if pi == 1 or pi == 2:
            ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(resfolder + 'flaring_stars_' + str(p) + '.pdf')
        plt.close()

    return

def Tstar_vs_Rstar_vs_Ro(df, nfl, resfolder):

    nfl_unique = nfl.drop_duplicates(subset='ticname')
    df_unique = df.drop_duplicates(subset='ticname')
    nfl_unique.rename(columns={'RADIUS':'radius'}, inplace=True)

    pars = ['radius', 'Teff [K]', 'Ro_bonanno']
    df_tot = pd.concat([df_unique[pars], nfl_unique[pars]])
    ro_bins = [0., 0.1, 0.5, np.inf]
    df_tot['Ro_bins'] = pd.cut(df_tot['Ro_bonanno'], bins=ro_bins, \
        labels=[r'$Ro \leq 0.1$', r'$0.1 < Ro \leq 0.5$', r'$Ro > 0.5$'])
    sns.scatterplot(df_tot, x='radius', y='Teff [K]', s=20, hue='Ro_bins', \
            palette='colorblind')
    plt.xlabel(r'$R_\star [R_\odot]$', fontsize=14)
    plt.ylabel(r'$T_\mathrm{eff}$ [K]', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resfolder + 'Tstar_vs_Rstar.pdf')
    plt.close()

    return

def peaks_vs_ro(df, resfolder):
    # Peaks vs ro bins
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.arange(0.5, 6.5)
    histos = []
    for j, ro_i in enumerate(df['Ro_bins'].unique()):
        flag_ro = df['Ro_bins'] == ro_i
        ax.hist(df['peaks_per_event'][flag_ro], bins=bins, \
                    log=True, histtype='step', label=ro_i, density=True, \
                    linewidth=2)
        histos.append(np.histogram(df['peaks_per_event'][flag_ro], bins=bins, \
                        density=True))
    ax.set_xlabel('Peaks per event', fontsize=14)
    ax.set_ylabel('Events', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(resfolder + 'peaks_vs_ro.pdf')
    plt.close()

    return

def consecutive_flare_stats(df, resfolder):

    df.sort_values('Peak time [BTJD]', inplace=True)
    df.reset_index(inplace=True, drop=True)
    dfc = df[df['Flare type'] == 'SFC']
    dfc['order'] = dfc.groupby(['LCname', 'n_event']).cumcount()

    # Create a subset for only consecutive isolated flares
    dfi = df[df['Flare type'] == 'IF']
    dfi['seq'] = dfi.groupby('LCname')['n_event'].diff()
    dfi = dfi[dfi['seq'] == 1]
    dfi.reset_index(inplace=True, drop=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax = ax.flatten()
    pars = ['Peak luminosity [erg s$^{-1}$]', 'Impulsiveness [min$^{-1}$]', \
            'Energy [erg]', 'Duration [min]']
    logbins = np.logspace(-1, 3, 20)
    # Distributions to compare
    distribs = {}
    for m, par in enumerate(pars):
        # Start with consecutive isolated flares
        dfi['normalized_' + par] \
            = dfi.groupby(['LCname'])[par].shift(1) / dfi[par]
        ax[m].hist(dfi['normalized_' + par], log=True, histtype='step', \
                    label='Consecutive single-peak', \
                    density=True, cumulative=-1, bins=logbins)

        dfc['normalized_' + par] \
            = dfc.groupby(['LCname', 'n_event'])[par].shift(1) / df[par]
        for order in range(1, 5):
            flag = dfc['order'] == order
            ax[m].hist(dfc['normalized_' + par][flag], log=True, histtype='step', \
                    label='Peak no. {} to {}'.format(order + 1, order), \
                    density=True, cumulative=-1, bins=logbins)
        ax[m].set_xscale('log')
        ax[m].set_xlabel(par.split('[')[0] + 'ratio', fontsize=14)
        ax[m].legend(loc='lower left')
    fig.supylabel('CCDF', fontsize=14)
    plt.tight_layout()
    plt.savefig(resfolder + 'consecutive_flare_stats.pdf')
    plt.close()

    for par in pars:
        print('\n', par)
        for order in range(1, dfc['order'].max() + 1):
            flagm = dfc['order'] == order - 1
            flag = dfc['order'] == order
            if order == 1:
                # Compare first vs second in complex with consecutive isolated
                pval = stats.ks_2samp(dfc[par][flagm].dropna(), \
                            dfi[par].dropna()).pvalue
                print(str(order - 1), 'vs consecutive isolated:', pval)
            pval = stats.ks_2samp(dfc[par][flagm].dropna(), \
            dfc[par][flag].dropna()).pvalue
            print(str(order), 'vs', str(order - 1), ': ', pval)

    dfc['Energy_consecutive [erg]'] = dfc.groupby(['LCname', \
            'n_event'])['Energy [erg]'].shift(1).astype(float)
    dfi['Energy_consecutive [erg]'] = dfi.groupby('LCname')[ \
                    'Energy [erg]'].shift(1).astype(float)
    alphas = [0.3, 1.]
    markers = ['o', '.']
    oks = []
    prs = []
    for i, dd in enumerate([dfi, dfc]):
        ok = ~np.isnan(dd['Energy_consecutive [erg]'])
        oks.append(ok)
        pr = stats.pearsonr(dd['Energy [erg]'][ok], \
                    dd['Energy_consecutive [erg]'][ok])
        plt.loglog(dd['Energy [erg]'], dd['Energy_consecutive [erg]'], \
            'k' + markers[i], label=r'$r=${:.2f}'.format(pr[0]), alpha=alphas[i])
        prs.append(pr[0])
    xx = np.logspace(31, 37, 1000)
    plt.loglog(xx, xx, 'r')
    plt.xlabel('Energy [erg]', fontsize=14)
    plt.ylabel('Consecutive flare energy [erg]', fontsize=14)
    plt.legend()
    plt.tight_layout()


    # Do the two pair distributions come from the same one?
    # Compare distance of Pearson correlation coeffcients
    T_obs = abs(prs[0] - prs[1])
    combined = pd.concat([dfi[['Energy [erg]', 'Energy_consecutive [erg]']][oks[0]], \
        dfc[['Energy [erg]', 'Energy_consecutive [erg]']][oks[1]]]).reset_index(drop=True)
    combined = np.array(combined)

    rng = np.random.default_rng()
    count = 0
    n_permutations = 1000
    for _ in range(n_permutations):
        perm_indices = rng.permutation(len(combined))
        # Split permuted data into two groups
        perm_sample1 = combined[perm_indices[:len(dfi)]]
        perm_sample2 = combined[perm_indices[len(dfi):]]

        # Calculate correlation difference in permuted samples
        r1, _ = stats.pearsonr(perm_sample1[:, 0], perm_sample1[:, 1])
        r2, _ = stats.pearsonr(perm_sample2[:, 0], perm_sample2[:, 1])
        T_perm = abs(r1 - r2)

        if T_perm >= T_obs:
            count += 1
    p_value = count / n_permutations
    print(f"Observed correlation difference: {T_obs:.4f}")
    print(f"p-value from permutation test: {p_value:.4f}")

    plt.show()
    set_trace()
    return

def flare_rate_per_target(df, resfolder):
    '''
    It makes more sense to divide isolated from multi-peak flares here, right?
    '''

    df.sort_values(by='Flare type', inplace=True)

    flag = ~np.isnan(df['rate_per_target'])
    # This works both for simple and complex flares
    single = df['Flare type'] == 'IF'
    sfc = df['Flare type'] == 'SFC'
    complex = df['Flare type'] == 'CF'

    ftd = ['IF', 'SFC', 'CF']
    fits = []
    for ift, ft in enumerate([single, sfc, complex]):
        fit = np.polyfit(df[ft][flag]['log_Ro'], \
                df[ft][flag]['rate_per_target'], 1)
        fits.append(fit)
        print('Fit rate,', ftd[ift], ':', fit)

    colors = ['royalblue', 'orange', 'g']
    fig, ax = plt.subplots()
    x = np.linspace(-3, 0, 1000)
    sns.histplot(df, x='log_Ro', y='rate_per_target', hue='Flare type', \
            alpha=0.5, fill=True, palette='colorblind')
    for i, cc in enumerate(colors):
        plt.plot(x, np.polyval(fits[i], x), c=colors[i])
    ax.set_xlabel(r'$\log Ro$', fontsize=14)
    ax.set_ylabel(r'Flares (star day)$^{-1}$', fontsize=14)
    plt.tight_layout()
    plt.ylim(df['rate_per_target'].min() - 0.1, \
                df['rate_per_target'].max() + 0.1)
    plt.savefig(resfolder + 'flare_rate_per_target.pdf')
    plt.close()

    return df

def duration_vs_energy_vs_logg(dd, resfolder, label='', plot_fit=True):
    '''
    Dependence of duration on energy and a third parameter
    '''
    par = ['FWHM [min]', 'Duration [min]']
    if label == '':
        label = dd['Flare type'].drop_duplicates().values[0]

    for j, pi in enumerate(par):
        x1 = dd['log_energy']
        x2 = dd['logg']
        y = np.log10(dd[pi])
        a, b, c = [], [], []
        for iter in range(1000):
            x1i = x1*np.random.normal(loc=1., scale=0.01, size=np.shape(x1))
            X = np.array([x1i, x2])
            yi = y*np.random.normal(loc=1., scale=0.01, size=np.shape(y))
            popt, pcov = curve_fit(multi_var_model, X, y)
            a.append(popt[0])
            b.append(popt[1])
            c.append(popt[2])
        print('\n', pi, 'vs energy and logg,', label, ':')
        a = [np.mean(a), np.std(a)]
        b = [np.mean(b), np.std(b)]
        c = [np.mean(c), np.std(c)]
        print('a:', a)
        print('b:', b)
        print('c:', c)

        # Plot resulting function for different Ro values
        fig, ax = plt.subplots()
        pl = ax.scatter(x1, y, c=x2, marker='.', label=label)
        xth = np.linspace(31, 36.5, 1000)
        if plot_fit:
            for xfix in [4.2, 4.5, 4.8, 5.1]:
                x2fix = np.zeros(len(xth)) + xfix
                Xth = np.array([xth, x2fix])
                ax.plot(xth, multi_var_model(Xth, a[0], b[0], c[0]), 'k')
        ax.set_xlabel(r'$\log$ Energy [erg]', fontsize=14)
        ax.set_ylabel(r'$\log$' + pi, fontsize=14)
        cbar = plt.colorbar(pl, ax=ax)
        cbar.set_label(r'$\log g$', fontsize=14)
        ax.set_xlim(x1.min() - 0.05, x1.max() + 0.05)
        ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(resfolder + pi + '_vs_energy_vs_logg_' + label + '.pdf')
        plt.close()

    return

def energy_rate(df, resfolder):

    # Energy rate per target as a function of Ro
    energy_bins = np.logspace(31, 37, 10)
    bin_edges = energy_bins[:-1] + 0.5*np.diff(energy_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots()
    cc = ['royalblue', 'orange', 'green']
    df.sort_values('Ro_bonanno', inplace=True)
    for i, ro_i in enumerate(df['Ro_bins'].unique()):
        flag = df['Ro_bins'] == ro_i

        histograms = []
        for _, tg in df[flag].groupby('ticname'):
            counts, _ = np.histogram(tg['Energy [erg]'], bins=bin_edges)
            histograms.append(counts)

        hist_array = np.array(histograms)
        try:
            avg_hist = np.percentile(hist_array, 99., axis=0)/bin_centers
        except IndexError:
            continue
        std_hist = hist_array.std(axis=0)/bin_centers

        x = np.log10(bin_centers)
        y = np.log10(avg_hist)
        dy = 1./(avg_hist*np.log(10.))*std_hist
        flag = np.isinf(y)
        x = x[~flag]
        y = y[~flag]
        dy = dy[~flag]

        fit_rate = np.polyfit(x, y, 1., w=1./dy)
        print(ro_i, '\n\nFit rate/energy/target:', fit_rate)
        ax.errorbar(bin_centers, avg_hist, yerr=avg_hist*0.5, fmt='_', \
                    uplims=True, label=ro_i, color=cc[i])
        xTh = np.linspace(x.min(), x.max(), 100)
        ax.plot(10**xTh, 10**np.polyval(fit_rate, xTh), \
                color=cc[i], linestyle='--')

    ax.legend()
    ax.set_xlabel('Energy [erg]', fontsize=14)
    ax.set_ylabel('Flare rate [(erg day star)$^{-1}$]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(resfolder + 'energy_rate_target.pdf')
    plt.close()

    return

def condense_flare_cascades(df, resfolder):

    fcfile = resfolder + 'flare_cascades.csv'
    if len(glob.glob(fcfile)) == 0:
        dfc = copy.deepcopy(df)
        dfc['peaks_per_event'] = np.zeros(len(dfc))
        # This should be built for every LC
        dfc['ident'] = np.zeros(len(dfc))
        count = 0
        dfc.drop(columns='Duration [min]', inplace=True)
        dfc.rename(columns={'total_duration':'Duration [min]'}, inplace=True)
        dfc['Duration [min]'] *= 24.*60.
        dfc.drop(columns='FWHM [min]', inplace=True)
        dfc.rename(columns={'t12':'FWHM [min]'}, inplace=True)
        group_cols = ['LCname', 'n_event']
        cols_to_sum = ['ED [s]', 'Energy (Shibayama) [erg]', \
                        'Energy [erg]', 'Energy [J]']
        dfc[cols_to_sum] = dfc.groupby(group_cols)[cols_to_sum].transform('sum')
        dfc['peaks_per_event'] = dfc.groupby(group_cols)['LCname'].transform('count')
        dfc['Peak time [BTJD]'] \
                = dfc.groupby(group_cols)['Peak time [BTJD]'].transform('mean')
        cols_to_max = ['Peak luminosity [erg s$^{-1}$]', 'Peak amplitude', \
                'Area [m$^2$]', 'Peak flux [W m$^{-2}$]', 'Peak luminosity [W]', \
                'Peak flux [erg s$^{-1}$ cm$^{-2}$]']
        dfc[cols_to_max] = dfc.groupby(group_cols)[cols_to_max].transform('max')
        dfc['Impulsiveness [min$^{-1}$]'] = dfc['Peak amplitude']/dfc['FWHM [min]']

        # Identifier for every cascade
        dfc['ident'] = dfc.groupby(group_cols).ngroup()
        dfc.drop_duplicates(subset='ident', inplace=True)
        dfc.to_csv(fcfile, index=False)
    else:
        dfc = pd.read_csv(fcfile)

    dfc = dfc.astype({'logg':'float', 'Teff [K]':'float', 'BP-RP_0':'float', \
                'feh':'float', 'Duration [min]':'float', 'radius':'float', \
                'FWHM [min]':'float', 'Energy [erg]':'float', \
                'Impulsiveness [min$^{-1}$]':'float', \
                'Peak amplitude':'float'}, copy=False)

    return dfc

def get_non_flaring_stars(resfolder, maxT, maxR, maxTmag, maxProt, maxFAP):

    print('\nGetting non-flaring stars...')

    # Data for non-flaring stars
    file_without_flares = resfolder + 'lcs_without_flares.pic'
    non_flaring_csv = resfolder + 'non_flaring_stars.csv'
    if not os.path.exists(non_flaring_csv):
        not_flaring = pickle.load(open(file_without_flares, 'rb'))
        gnf = pd.read_csv(resfolder + 'tic_notflaring_gaia_DR3_reddening.csv')
        keep_idx = gnf.groupby('OBJECT')['ang_sep'].idxmin()
        #keep_idx = gnf.groupby('OBJECT')['phot_g_mean_mag'].idxmin()
        gnf = gnf.loc[keep_idx].reset_index(drop=True)
        nf_pars = {}
        pp =  ['OBJECT', 'TEFF', 'LOGG', 'MH', 'RADIUS', 'Prot_[days]', 'FAP', \
                'TESSMAG', 'RA_OBJ', 'DEC_OBJ', 'phot_var']
        for p in pp:
            nf_pars[p] = []
        for ni, nf in enumerate(not_flaring):
            if ni % 10000 == 0:
                print(ni)
            for p in pp:
                if type(nf[p]) == str or type(nf[p]) == float or type(nf[p]) == np.float64:
                    nf_pars[p].append(nf[p])
                else:
                    nf_pars[p].append(np.nan)
        nfl = pd.DataFrame.from_dict(nf_pars).dropna()
        nfl.rename(columns={'TEFF':'Teff [K]', 'LOGG':'logg', 'MH':'feh'}, \
                inplace=True)
        flagNF = np.logical_and.reduce((nfl['Teff [K]'] < maxT, \
                    nfl['RADIUS'] < maxR, nfl['Prot_[days]'] < maxProt, \
                    nfl['FAP'] < maxFAP, nfl['TESSMAG'] <= maxTmag))
        nfl = nfl[flagNF]
        nfl.to_csv(resfolder + 'non_flaring_stars.csv', index=False)
    else:
        nfl = pd.read_csv(non_flaring_csv)

    nfl['tau_conv_cranmer'] = compute_tau_conv_cranmer(nfl['Teff [K]'])
    nfl['Ro_cranmer'] = nfl['Prot_[days]']/nfl['tau_conv_cranmer']

    # Add Gaia colours
    gnf = pd.read_csv(resfolder + 'tic_notflaring_gaia_DR3_reddening.csv')
    keep_idx = gnf.groupby('OBJECT')['ang_sep'].idxmin()
    gnf = gnf.loc[keep_idx].reset_index(drop=True)
    nfl = pd.merge(nfl, gnf, on='OBJECT')
    nfl.rename(columns={'OBJECT':'ticname'}, inplace=True)

    nfl['BP-RP_0'] = nfl['bp_rp'] - nfl['ebpminrp_gspphot']
    nfl['tau_conv_bonanno'] = compute_tau_conv_bonanno( \
            nfl['BP-RP_0'].astype(float), nfl['Teff [K]'].astype(float), \
            nfl['logg'].astype(float), nfl['feh'].astype(float))
    nfl['Ro_bonanno'] = nfl['Prot_[days]']/nfl['tau_conv_bonanno']
    ro_bins = [0., 0.1, 0.5, np.inf]
    nfl['Ro_bins'] = pd.cut(nfl['Ro_bonanno'], bins=ro_bins, \
        labels=[r'$Ro \leq 0.1$', r'$0.1 < Ro \leq 0.5$', r'$Ro > 0.5$'])

    # Herbst+2021
    Tspot = -3.58e-5*nfl['Teff [K]']**2 + 1.0188*nfl['Teff [K]'] + 239.3
    star_area = np.pi*(nfl['RADIUS']*constants.R_sun.to(units.cm))**2
    Aspot = nfl['phot_var']*star_area*(1. - (Tspot/nfl['Teff [K]'])**4)**-1
    sun_area = np.pi*constants.R_sun.to(units.cm)**2
    Aspot /= sun_area
    nfl['Aspot [Asun]'] = Aspot
    nfl['Aspot_mean [Asun]'] = nfl.groupby('ticname')['Aspot [Asun]'].transform('mean')

    flag = np.logical_and.reduce((nfl['Teff [K]'] > 0., nfl['Teff [K]'] <= maxT, \
                nfl['RADIUS'] > 0., nfl['RADIUS'] <= maxR, \
                nfl['TESSMAG'] <= maxTmag, ~np.isnan(nfl['logg']), \
                ~np.isnan(nfl['feh'])))
    nfl = nfl[flag]

    '''
    plt.loglog(nfl['Ro_cranmer'], nfl['Ro_bonanno'], '.')
    flag = np.logical_and(nfl['BP-RP_0'] > 0.55, nfl['BP-RP_0'] < 1.25)
    plt.loglog(nfl['Ro_cranmer'][flag], nfl['Ro_bonanno'][flag], '.', \
            label=r'$0.55 < G_{BP} - G_{RP} < 1.25$')
    plt.legend()
    plt.xlabel(r'$Ro$ (Cranmer+)', fontsize=14)
    plt.ylabel(r'$Ro$ (Bonanno et al. 2025)', fontsize=14)
    x = np.linspace(1e-3, 1., 1000)
    plt.loglog(x, x)
    plt.tight_layout()
    plt.savefig(resfolder + 'Ro_bonanno_vs_cranmer_notflaring.pdf')
    plt.close()
    '''
    return nfl

def multi_var_model(X, a, b, c):
    x, y = X
    return a*x + b*y + c

if __name__ == "__main__":
    #call_LC(sys.argv[1])
    call_sector(sys.argv[1], sys.argv[2])
