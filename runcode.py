# Tools for (multiple) TESS data sets
import os
import sys
import copy
import lc_class
#import phot_analysis
import pickle
from astropy.io import fits
from astropy import units, constants
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import flare_detection as fd
import pandas as pd
import powerlaw
homedir = os.path.expanduser('~')
from pdb import set_trace
import glob
#from rpy2.robjects.packages import importr
#import rpy2.robjects.r
#powerlaw = importr('poweRlaw')

#datadir = '/beegfs/gbruno/data/'
datadir = '/media/giovanni/data_'
throughput_folder = datadir + '/filters/'

def call_sector(nsector, lcini):
    # Just for naming on SLURM
    nsector = nsector.split('S')[1]
    lcs = glob.glob(datadir + 'lightcurves/S' + str(nsector) + '/*fits')
    for lc in lcs[int(lcini)*100:int(lcini)*100 + 100]:
        call_LC(str(nsector), lc)
    return

def call_LC(nsector, LC):

    #saveresfolder = '/beegfs/gbruno/results/S' + str(nsector) + '/' \
    #                + LC.split('/')[-1]
    saveresfolder = '/home/giovanni/Projects/flares_TESS_fullsample/results_local/' \
                            + LC.split('/')[-1]
    fout_str = saveresfolder.replace('.fits', '_results.pic')
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

    # From Teff, estimate stellar radius
    # rstar = phot_analysis.teff_radius_calib(tstar)

    # Compute correlted noise level
    try:
        lc = fits.open(LC, ignore_missing_simple=True)
    except OSError:
        return [], []

    flarespar, flaresflag = fd.find_flares(LC, wth=wth, fth=fth, \
        flare_threshold=flare_threshold, flatten=True, plot_flat=False, \
        saveplots=saveresfolder, normalise=True, \
        fit_continuum=1, complexity=5, clip=True, \
        rebin=rebin, verbose=False)

    return flarespar, flaresflag

def get_results_sample(resfolder, simulate_bootstrap=False, \
                                                        sectors=range(27, 84)):

    wth, fth = np.loadtxt(throughput_folder + 'TESS_TESS.Red.dat', unpack=True)

    results = {}
    parameters = ['Amplitude', 'Duration [min]', 'FWHM [min]', 'Energy [erg]', \
        'ED [s]', 'Teff [K]', 'logg', 'radius', 'tessmag', 'Prot', 'FAP', \
        'redchi2', 'ticname', 'ra', 'dec', 'scatter_SN']

    for sector in sectors:
        foutname = resfolder.replace('cluster', 'analysis') + 'results_sector' \
                            + str(sector) + '.pic'
        if os.path.exists(foutname) and os.path.getsize(foutname) > 100:
            print('Sector ' + str(sector) + ': results already collected.')
            results['S' + str(sector)] = pickle.load(open(foutname, 'rb'))
            continue

        sample = glob.glob(resfolder + 'S' + str(sector) + '/*-s00' \
                        + str(sector) + '*results.pic')
        params_s = {}
        for p in parameters:
            params_s[p] = []
        for LCfile in sample:
            print(LCfile)
            try:
                resLC = pickle.load(open(LCfile, 'rb'))
            except EOFError:
                # The file is empty
                continue
            if len(resLC) == 0:
                continue

            header = resLC[-1]

            for flare in resLC[:-1]:
                try:
                    for n in np.arange(flare.npeaks):
                        duration = flare.durations[n].to(units.min).value
                        if type(duration) == np.ndarray:
                            params_s['Duration [min]'].append( \
                                    flare.durations[n].to(units.min).value[3])
                        else:
                            # Somehow fit failed
                            continue
                        params_s['redchi2'].append(flare.result_nodip.redchi)
                        #pres = flare.result_nodip.params_Ftest
                        #if type(pres) == str:
                        params_s['Amplitude'].append( \
                            flare.result_nodip.params_LM['ampl' + str(n)].value)
                        params_s['FWHM [min]'].append( \
                            flare.result_nodip.params_LM['fwhm' + str(n)].value)
                        #else:
                        #    params_s['Amplitude'].append(pres[0]['ampl' + str(n)][2][1])
                        #    params_s['FWHM [min]'].append(pres[0]['fwhm' + str(n)][2][1])
                        params_s['ED [s]'].append(flare.EDs[n].value)
                        # Get flare energy if not already computed, assuming
                        # a temperature for the flare
                        #if header['TEFF'] is not None and header['RADIUS'] \
                        #    is not None:
                        try:
                            flarecopy = copy.deepcopy(flare)
                            flarecopy.Tstar = header['TEFF']*units.K
                            flarecopy.Rstar = header['RADIUS']*constants.R_sun
                            flarecopy.Tflare = 9000.*units.K
                            flarecopy.wth = wth*units.AA
                            flarecopy.fth = fth*units.AA
                            flarecopy.yprof = flare.get_single_profile(n, \
                                                                double_t=False)
                            flarecopy.get_flare_energy(method='shibayama')
                            params_s['Energy [erg]'].append( \
                                                    flarecopy.energy.value)
                            params_s['Teff [K]'].append(header['TEFF'])
                            params_s['radius'].append(header['RADIUS'])
                        except TypeError:
                            params_s['Energy [erg]'].append(None)
                            params_s['Teff [K]'].append(None)
                            params_s['radius'].append(None)
                        # Stellar params
                        params_s['logg'].append(header['LOGG'])
                        params_s['ra'].append(header['RA_OBJ'])
                        params_s['dec'].append(header['DEC_OBJ'])
                        params_s['ticname'].append(header['OBJECT'])
                        params_s['tessmag'].append(header['TESSMAG'])
                        params_s['Prot'].append(header['Prot_[days]'])
                        params_s['FAP'].append(header['FAP'])
                        params_s['scatter_SN'].append(header['scatter_SN'])
                except AttributeError as ae:
                    errmsg = 'OSError: No SIMPLE card found, this file does not appear to be a valid FITS file.'
                    if errmsg in str(ae):
                        continue

        for p in parameters:
            if p != 'Duration [min]':
                params_s[p] = np.array(params_s[p])
            else:
                params_s[p] = np.array(params_s[p])
        results['S' + str(sector)] = params_s

        fout = open(foutname, 'wb')
        pickle.dump(params_s, fout)
        fout.close()

    params = {}
    for p in parameters:
        params[p] = np.hstack([results['S' + str(i)][p] for i in sectors])

    plt.figure()
    plt.hist(params['redchi2'], bins=20, log=True)
    plt.xlabel(r'Fit $\tilde{\chi}^2$', fontsize=14)
    plt.ylabel('Flares', fontsize=14)
    plt.tight_layout()
    plt.savefig(resfolder.replace('cluster', 'analysis') + 'redchi2_fits.pdf')
    plt.close()

    flag = np.logical_or.reduce((params['Teff [K]'] == None, \
                params['radius'] == None, params['scatter_SN'] > 3.))
    for p in params.keys():
        params[p] = np.array(params[p][~flag])
    keys, values = zip(*params.items())
    df = pd.DataFrame(data=np.transpose(values), columns=keys)
    df.to_csv(resfolder.replace('cluster', 'analysis') + 'flare_params.csv', \
                    index=False)

    fig, axs = plt.subplots()
    size = (np.log10(df['radius'].astype(int)) + 5)**2
    sym = ['*', '^', '+']
    Prange = [10., 20., np.inf]
    for i, sy in enumerate(sym):
        if i == 0:
            Pr = df['Prot'] <= Prange[i]
        else:
            Pr = np.logical_and(df['Prot'] > Prange[i - i], \
                        df['Prot'] <= Prange[i])
        pl = axs.scatter(df['Duration [min]'][Pr], df['Amplitude'][Pr], \
            c=df['Teff [K]'][Pr], s=size[Pr], marker=sy)
    axs.set_xlabel('Duration [min]', fontsize=14)
    axs.set_ylabel('Amplitude', fontsize=14)
    axs.set_xscale('log')
    axs.set_yscale('log')
    cbar = fig.colorbar(pl)
    cbar.set_label(r'$T_\mathrm{eff}$ [K]', fontsize=14)
    plt.tight_layout()
    plt.savefig(resfolder.replace('cluster', 'analysis') \
                                    + 'ampl_vs_duration_scatterplot.pdf')
    plt.close()

    # Separate results by spectral type
    tlim = [4200., 5300., 5950., 7200., 9600., 31000.]
    rlim = [0.48, 0.83, 1.12, 1.62, 1.80, 7.2]
    sptype = ['M', 'K', 'G', 'F', 'A', 'B']
    Prot = {}

    for ll in range(len(tlim)):
        print(sptype[ll] + ' stars...')
        if ll == 0:
            flag2 = np.logical_and(df['Teff [K]'] <= tlim[ll], \
                            df['radius'] <= rlim[ll])
        else:
            flag2 = np.logical_and.reduce((df['Teff [K]'] > tlim[ll - 1], \
                    df['Teff [K]'] <= tlim[ll], \
                    df['radius'] > rlim[ll - 1], \
                    df['radius'] <= tlim[ll]))

        if len(df['Amplitude']) > 0:
            for par in ['Amplitude', 'FWHM [min]', 'Duration [min]', \
                        'Energy [erg]', 'ED [s]']:
                fit = powerlaw.Fit(df[par][flag2])#, original_data=True)
                #if par == 'Energy [erg]' and simulate_bootstrap:
                #    simulate_pl_unc(fit, len(df[par][flag2]), \
                #        folderout=resfolder.replace('cluster', 'analysis'))
                #else:
                #    continue
                #x, y = fit.pdf(original_data=True)
                #y0 = y[x == fit.xmin]
                #y /= y0
                #fig = fit.plot_ccdf(color='k')#, original_data=True)
                #R, p = fit.distribution_compare('truncated_power_law', \
                #        'lognormal', normalized_ratio=True)

                # Fit power law for every object in this subset
                targets = df['ticname'][flag2].drop_duplicates()
                target_pars = {}
                fast_rotators = []
                for tgp in ['alpha', 'sigma', 'Prot', 'fap', 'nobs', 'alpha_t', 'lambda']:
                    target_pars[tgp] = []
                valid_targets = 0
                print('P < 1 day:')
                for tg in targets:
                    flag_tg = df['ticname'][flag2] == tg
                    if np.sum(flag_tg) < 100:
                        continue
                    valid_targets += 1
                    fit_tg = powerlaw.Fit(df[par][flag2][flag_tg])
                    # Fit with a truncated power law
                    R_tg, p_tg = fit_tg.distribution_compare('power_law', 'truncated_power_law', \
                                        normalized_ratio=True)
                    target_pars['alpha'].append(fit_tg.alpha)
                    target_pars['sigma'].append(fit_tg.sigma)
                    target_pars['alpha_t'].append(fit_tg.truncated_power_law.parameter1)
                    target_pars['lambda'].append(fit_tg.truncated_power_law.parameter2)
                    target_pars['Prot'].append(np.median(df['Prot'][flag2][flag_tg]))
                    if np.median(df['Prot'][flag2][flag_tg]) < 1.:
                        fast_rotators.append(tg)
                    target_pars['fap'].append(np.median(df['FAP'][flag2][flag_tg]))
                    target_pars['nobs'].append(np.sum(flag_tg))

                for tgp in target_pars.keys():
                    target_pars[tgp] = np.array(target_pars[tgp])

                fig_tg, ax_tg = plt.subplots(ncols=2, figsize=(12, 5))
                xx = np.arange(valid_targets)
                ax_tg[0].errorbar(xx, target_pars['alpha'], \
                    yerr=target_pars['sigma'], fmt='o', label=r'Standard PL $\alpha$', \
                    capsize=2)
                ax_tg[0].plot(xx, target_pars['alpha_t'], \
                    '^', label=r'Truncated PL $\alpha$')
                plt.legend()
                y1 = np.zeros(len(xx)) + fit.alpha + fit.sigma
                y2 = np.zeros(len(xx)) + fit.alpha - fit.sigma
                ax_tg[0].fill_between(xx, y1, y2, color='orange')
                #ax_tg[0].set_yscale('log')
                ax_tg[0].set_ylabel(r'$\alpha$', fontsize=14)
                ax_tg[0].set_xlabel('Target', fontsize=14)
                ax_tg[1].hist((target_pars['alpha'] - fit.alpha) \
                    /target_pars['sigma'], \
                    log=False, orientation='horizontal', bins=20, \
                    histtype='step', linewidth=2)
                ax_tg[1].set_ylabel(r'$(\alpha - \alpha_i)/\sigma_i$', \
                            fontsize=14)
                ax_tg[1].set_xlabel('Targets', fontsize=14)
                plt.savefig(resfolder.replace('cluster', 'analysis') \
                            + par + '_alphadiff_' + sptype[ll] + '.pdf')
                plt.close()

                figP, axP = plt.subplots()
                flagFAP = target_pars['fap'] <= 0.01
                Prot[sptype[ll]] = target_pars['Prot'][flagFAP]
                if np.sum(flagFAP) > 1:
                    x = target_pars['Prot'][flagFAP]
                    y = target_pars['alpha'][flagFAP]
                    yerr = target_pars['sigma'][flagFAP]
                    linfit = Polynomial.fit(x, y, 1, w=1./yerr)
                    xx = np.linspace(x.min(), x.max(), 1000)
                    axP.errorbar(x, y, yerr=yerr, fmt='.', capsize=2)
                    plP = axP.scatter(x, y, c=target_pars['nobs'][flagFAP])
                    axP.plot(xx, linfit(xx), '--', label=str(linfit))
                    axP.set_xscale('log')
                    plt.setp(axP, xticks=[0.3, 0.5, 1., 3., 5., 10.], \
                                xticklabels=['0.3', '0.5', '1', '3', '5', '10'])
                    cbar = figP.colorbar(plP)
                    cbar.set_label('Number of flares', fontsize=14)
                    axP.set_xlabel(r'$P_\mathrm{rot}$ [days]', fontsize=14)
                    axP.set_ylabel(r'$\alpha$', fontsize=14)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(resfolder.replace('cluster', 'analysis') + par \
                                    + '_alpha_vs_Prot_' + sptype[ll] + '.pdf')
                    plt.close()

                def get_Rsign(R):
                    if np.sign(R) > 0:
                        return 'non supported'
                    else:
                        return r'supported'

                fig = fit.plot_pdf(color='k')#, original_data=True)
                fit.power_law.plot_pdf(linestyle=':', \
                            label=r'Power law: $\alpha={:.2f} \pm {:.2f}$'.format( \
                            fit.alpha, fit.sigma), ax=fig)
                R1, p1 = fit.distribution_compare('power_law', 'lognormal', \
                        normalized_ratio=True)
                text = get_Rsign(R1) + r', $p={:.2f}, \, \mu={:.2f}, \, \sigma={:.2f}$'.format( \
                        p1, fit.lognormal.parameter1, fit.lognormal.parameter2)
                fit.lognormal.plot_pdf(linestyle='--', \
                            ax=fig, label='Lognormal: ' + text)
                R2, p2 = fit.distribution_compare('power_law', 'truncated_power_law', \
                        normalized_ratio=True)
                text = get_Rsign(R2) \
                    + r', $p={:.2f}, \, \alpha={:.2f}, \, \lambda={:.2f}$'.format( \
                    p2, fit.truncated_power_law.parameter1, \
                    fit.truncated_power_law.parameter2)
                fit.truncated_power_law.plot_pdf(linestyle='-.', \
                            label='Truncated power law: ' + text, ax=fig)
                #fig.loglog(x, y, color='k')#
                #fig.semilogy([fit.xmin, fit.xmin], [y.min(), y.max()], 'k--', \
                #        label=r'$x_\mathrm{min}$')
                plt.legend()
                plt.xlabel(par, fontsize=14)
                plt.ylabel(r'$N(x)$', fontsize=14)
                plt.tight_layout()
                plt.savefig(resfolder.replace('cluster', 'analysis') \
                                + par + '_PDF_' + sptype[ll] + '.pdf')
                plt.close()

                # Outcome of different xmin
                fig2, ax = plt.subplots()
                pl2 = ax.scatter(fit.xmins, fit.alphas, c=fit.Ds)
                ax.set_xlabel(r'$x_\mathrm{min}$', fontsize=14)
                ax.set_ylabel(r'$\alpha$', fontsize=14)
                cbar = fig2.colorbar(pl2)
                cbar.set_label('KS distance', fontsize=14)
                plt.savefig(resfolder.replace('cluster', 'analysis') + par \
                                + '_x0_map_' + sptype[ll] + '.pdf')
                plt.close()

    plt.figure()
    for spt in sptype:
        plt.hist(Prot[spt], bins=20, log=False, label=spt)
    plt.legend()
    plt.xlabel(r'$P_\mathrm{rot}$ [days]', fontsize=14)
    plt.ylabel('Stars', fontsize=14)
    plt.savefig(resfolder.replace('cluster', 'analysis') + 'Prot.pdf')

    return

def simulate_pl_unc(fit, ndata, n=300, unc=0.5, folderout=''):
    '''
    Get power law estimate from data, bootstrap realizations to estimate
    uncertainties based on possible parameter error bars.

    Parameters
    ----------
    ndata (int): number of observed (and simulated) data points
    n (int): number of bootstrap iterations
    unc (float): fractional uncertainty on data points

    Returns
    -------
    Derived uncertainty on power law slope
    '''

    simulated_data = fit.truncated_power_law.generate_random(ndata)

    alphas, sigmas, x0s = [], [], []
    alphas_truncated, lambdas = [], []
    for i in range(n):
        print(i, '\t')
        newdata = abs(simulated_data*np.random.normal(loc=1., scale=unc, \
                                                    size=len(simulated_data)))
        fit_th = powerlaw.Fit(newdata)
        alphas.append(fit_th.alpha)
        sigmas.append(fit_th.sigma)
        x0s.append(fit_th.xmin)
        R2, p2 = fit_th.distribution_compare('power_law', 'truncated_power_law', \
            normalized_ratio=True)
        alphas_truncated.append(fit_th.truncated_power_law.parameter1)
        lambdas.append(fit_th.truncated_power_law.parameter2)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
    axs[0][0].hist(alphas)
    axs[0][1].hist(sigmas)
    axs[0][2].hist(x0s)
    axs[1][0].hist(alphas_truncated)
    axs[1][1].hist(lambdas)
    axs[0][0].set_xlabel(r'$\alpha$', fontsize=14)
    axs[0][1].set_xlabel(r'$\sigma$', fontsize=14)
    axs[0][2].set_xlabel(r'$x_0$', fontsize=14)
    axs[1][0].set_xlabel(r'$\alpha_t$', fontsize=14)
    axs[1][1].set_xlabel(r'$\lambda$', fontsize=14)
    fig.supylabel('Counts', fontsize=14)
    plt.savefig(folderout + 'uncertainty_simulated_data_' + str(int(unc*100)) \
                + '_n' + str(n) + '.pdf')

    plt.show()
    set_trace()

    return

if __name__ == "__main__":
    #call_LC(sys.argv[1])
    call_sector(sys.argv[1], sys.argv[2])
