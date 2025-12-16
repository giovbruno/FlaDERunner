'''
Class for flares and relative properties.
'''

import lmfit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
from scipy.integrate import trapezoid
#from scipy.integrate import trapz as trapezoid
from scipy import signal
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
from astropy import constants
from astropy.stats import sigma_clip
from astropy import timeseries
from corner import corner
import copy
import models
import stopit
from pdb import set_trace

daytomin = 24*60.
percentiles_3sigma = [0.15, 2.5, 16., 50., 84., 97.5, 99.85]

class flare:
    def __init__(self, tdata, ydata, yerrdata, noise_level, \
            targetname=None, tref=None, traw=None, yraw=None, \
            rednoise=None, tbeg=None, tend=None, \
            tpeak=None, Tstar=None, Tflare=None, Rstar=None, \
            wth=None, fth=None):

        self.targetname = targetname
        self.tdata = tdata*u.day
        self.ydata = ydata
        self.yerrdata = yerrdata
        self.noise_level = noise_level
        self.rednoise = rednoise
        self.traw = traw
        self.yraw = yraw
        self.tbeg = tbeg
        self.tend = tend
        self.tref = tref
        self.tpeak = tpeak
        self.Tstar = Tstar
        self.Tflare = Tflare
        self.Rstar = Rstar
        self.wth = wth
        self.fth = fth

    def get_model_from_result(self, fit_result):
        return self.ydata - fit_result.residual*self.yerrdata

    def fit_line(self, verbose=True):
        '''
        Polynomial fit to the flare profle.
        '''

        fit_params_line = lmfit.Parameters()
        fit_params_line.add('a', value=0., vary=True)
        fit_params_line.add('b', value=0., vary=True)
        fit_params_line.add('c', value=0., vary=True)

        tt = self.tdata.value

        result = lmfit.minimize(models.residual_line, fit_params_line, \
                calc_covar=False, args=(tt, self.ydata, self.yerrdata), \
                method='least_squares', nan_policy='omit')
        if verbose:
            print('0 flares : BIC = {:.2f}, redchi2 = {:.2f}'.format( \
                result.bic, result.redchi))

        self.result_line = result

        return

    def get_flare_t12(self):
        '''
        Get flare FWHM extremes and add attribute for FWHM extent (t12).
        '''

        fmax = self.ydata.argmax()
        tmax = self.tdata[fmax]
        diffs = abs(self.ydata - 0.5*self.ydata.max())
        checkleft = self.tdata < tmax
        checkright = self.tdata >= tmax
        if np.sum(checkleft) == 0 or np.sum(checkright) == 0:
            print('Too close to a data set border')
            self.npeaks = 0.
            return
        x1 = self.tdata[diffs[checkleft].argmin()]
        bb = np.sum(self.tdata < tmax)
        x2 = self.tdata[diffs[checkright].argmin() + bb]
        t12 = x2 - x1
        self.t12 = t12

        return x1, x2

    def fit_flare_profile(self, complexity, threshold, \
            fit_continuum=False, uncertainties='LM', fit_slowdecay=True, \
            plots=False, min_datapoints=3, min_dist=5, plotname='', \
            verbose=True):
        '''
        Fit multiple flare profiles to the data, using Mendoza's model.

        Parameters
        ----------
        fit_slowdecay (bool): whether to fit for the slow decay timescale,
        following the results by Seli+2024, or to fix it to the value
        found by Mendoza+2022.
        '''

        if not 'result_line' in dir(self):
            print('You must first run fit_line to run the AIC tests.')
            return

        x1, x2 = self.get_flare_t12()

        tt = self.tdata.value
        noiselev = self.noise_level

        self.npeaks = 0
        for n in np.arange(1, complexity + 1):
            fit_params = lmfit.Parameters()
            for nf in np.arange(n):
                fit_params.add('ampl' + str(nf), min=self.noise_level*3, \
                        max=5*self.ydata.max())
                fit_params.add('tpeak' + str(nf), min=self.tbeg, max=self.tend)
                # Reduced minimum FWHM from twice to 1/10th data cadence, just
                # to set a lower limit.
                fit_params.add('fwhm' + str(nf), \
                        min=np.diff(tt).min(), max=self.tend - self.tbeg)
                fit_params.add('D2' + str(nf), value=1.2151, min=0.5, max=1.5, \
                            vary=fit_slowdecay)

            fit_params.add('d', value=0., vary=True)
            fit_params.add('e', value=0., vary=True)
            fit_params.add('f', value=0., vary=True)

            if fit_continuum == -1:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = False
            elif fit_continuum == 0:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = True
            elif fit_continuum == 1:
                fit_params['d'].vary = False
                fit_params['e'].vary = True
                fit_params['f'].vary = True
            elif fit_continuum == 2:
                fit_params['d'].vary = True
                fit_params['e'].vary = True
                fit_params['f'].vary = True

            # Parameters for dip
            fit_params.add('Gdip', value=0., min=-1, max=0., vary=False)
            if tt.min() != self.tbeg:
                fit_params.add('x0', min=tt.min(), max=self.tbeg, vary=False)
            else:
                fit_params.add('x0', min=tt.min(), \
                    max=tt.min() + np.diff(tt).min(), vary=False)

            fit_params.add('w1', value=2./daytomin, min=0.1/daytomin, \
                        max=20./daytomin, vary=False)
            fit_params.add('w2', value=2./daytomin, min=0.1/daytomin, \
                        max=20./daytomin, vary=False)
            fit_params.add('n', value=2.1, min=2., max=4., vary=False)

            fit_kws = {}
            fit_kws['niter'] = 10
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            # If the minimization fails, change the initial parameters
            scramble = True
            scramble_i = 0
            mini = lmfit.Minimizer(models.mendoza_residuals, fit_params, \
                    fcn_args=(tt, self.ydata, self.yerrdata), \
                    fcn_kws={'complexity': n}, \
                    nan_policy='omit', calc_covar=True)
            while scramble:
                try:
                    result = mini.minimize(method='basinhopping', **fit_kws)
                    scramble = False
                except FloatingPointError:
                    for p in fit_params:
                        if p != 'd' and p != 'e' and p != 'f':
                            fit_params[p].value = np.random.uniform( \
                                low=fit_params[p].min + 1e-5, \
                                high=fit_params[p].max - 1e-5)
                    scramble_i += 1
                    if scramble_i > 100:
                        self.npeaks = 0
                        break
                    else:
                        pass

            # If the fit fails after the first flare, stop here
            if n > 1 and result.success == False:
                break
            if n == 1 and result.bic >= self.result_line.bic - 10.:
                self.result_nodip = result
                break
            elif n > 1 and result.bic > bestbic - 10.:
                break
            else:
                if n == 1:
                    # Get amplitude of residuals in case a QPP candidate
                    # is found
                    try:
                        oneflare_resid = result.residual*self.yerrdata
                        oscill_ampl = 0.5*np.ptp(oneflare_resid)
                        oscill_ampl_ratio = oscill_ampl/result.params['ampl0']
                        oscill_ampl_ratio_err \
                            = oscill_ampl/result.params['ampl0'].value**2 \
                              *result.params['ampl0'].stderr
                        oneflare_profile = self.ydata - oneflare_resid
                        oneflare_model = copy.deepcopy(result)
                    except TypeError:
                        print('Issue with minimization solution')
                        return
                if n > 1:
                    # If the additional flare model does not have a SNR > 4,
                    # stop here
                    thisflare = self.ydata - result.residual*self.yerrdata
                    if thisflare.max() < threshold*noiselev:
                        break
                    # If the flare peak is < min_dist points close to another
                    # peak, stop here
                    t_peaks_sol = []
                    tooclose = False
                    for tp in np.arange(self.npeaks + 1):
                        t_peaks_sol.append(result.params['tpeak' + str(tp)])
                        if (np.diff(np.sort(t_peaks_sol)) \
                                        < np.diff(tt).min()*min_dist).any():
                            tooclose = True
                    if tooclose:
                        break
                    # if the flare model lasts less than min_datapoints,
                    # stop here
                    flare_start = tt[thisflare > noiselev][0]
                    flare_end = tt[thisflare > noiselev][-1]
                    if flare_end - flare_start < min_datapoints*np.diff(tt).min():
                        break
                bestbic = np.copy(result.bic)
                self.npeaks += 1
                self.result_nodip = copy.deepcopy(result)
                # In case uncertainties are calculated with the F-test
                self.saved_mini = copy.deepcopy(mini)
                if n == 1:
                    strl = 'flare '
                else:
                    strl = 'flares'
                if verbose:
                    print(n, strl, ': BIC = {:.2f}, redchi2 = {:.2f}'.format( \
                        result.bic, result.redchi))

            if n == complexity:
                break

        # If at least a flare is found, estimate uncertainties
        if self.npeaks > 0:
            X = copy.deepcopy(self.result_nodip)
            self.result_nodip.params_LM = X.params
            # Get flare luminosity, energy and relative uncertainties
            # See how energy is measured in Feinstein+. you either have
            # the uncertainty in stellar luminosity, or the one in
            # flare temperature.
            if type(uncertainties) == str and uncertainties == 'mcmc':
                if verbose:
                    print('--> Sampling parameter posteriors...')
                # Initialize walkers around best solution
                result = lmfit.minimize(models.mendoza_residuals, \
                   X.params, args=(tt, self.ydata, self.yerrdata), \
                   kws={'complexity': self.npeaks}, method='emcee', \
                   nwalkers=len(X.params)*3, steps=2000*self.npeaks + 2000, \
                   burn=1000*self.npeaks, \
                   is_weighted=True, nan_policy='omit', progress=True)
                self.result_nodip.params_emcee = result.params
            elif type(uncertainties) == str and uncertainties == 'Ftest':
                if verbose:
                    print('--> Estimating uncertainties via F-test...')
                vnames = [el for el in X.var_names if el not in ['d', 'e', 'f', \
                        'Gdip', 'x0' ,'w1', 'w2', 'n'] and 'tpeak' not in el]
                try:
                    # Abort in case this takes too long
                    with stopit.ThreadingTimeout(60) as context_manager:
                        ci, trace = lmfit.conf_interval(self.saved_mini, X, \
                          sigmas=[1, 2], p_names=vnames, trace=True, maxiter=200)

                    if context_manager.state == context_manager.EXECUTED:
                        lmfit.printfuncs.report_ci(ci)
                        self.result_nodip.params_Ftest = [ci, trace]
                    elif context_manager.state == context_manager.TIMED_OUT:
                        msg = 'F-test required too long. Using LM uncertainties'
                        print(msg)
                        self.result_nodip.params_Ftest = msg
                except lmfit.minimizer.MinimizerException:
                    msg = 'F-test failed. Using LM uncertainties'
                    print(msg)
                    self.result_nodip.params_Ftest = msg
                    pass
                except ValueError:
                    msg = 'F-test raised a ValueError. Using LM uncertainties'
                    print(msg)
                    self.result_nodip.params_Ftest = msg
                    pass
            elif type(uncertainties) == int:
                if verbose:
                    print('--> Estimating uncertainties via bootstrap...')
                # Generate n (uncertainties) random samples of the
                # distribution and minimize residual function to get fitted
                # pars. Then draw parameter distributions
                bs_pars = {}
                durations = {}
                for nf in range(self.npeaks):
                    durations[nf] = []

                plt.close('all')
                for p in X.var_names:
                    bs_pars[p] = []
                for ub in range(uncertainties):
                    # Generate new data set
                    newy = []
                    for i, yi in enumerate(self.ydata):
                        newy.append(yi + np.random.normal(scale=self.yerrdata[i]))
                    bs_mini = lmfit.Minimizer(models.mendoza_residuals, \
                        X.params, fcn_args=(tt, newy, self.yerrdata), \
                        fcn_kws={'complexity': self.npeaks}, \
                        nan_policy='omit', calc_covar=False)
                    result_bs = bs_mini.minimize(method='powell')
                    for p in X.var_names:
                        bs_pars[p].append(result_bs.params[p].value)
                    flarecopy = copy.deepcopy(self)
                    flarecopy.result_nodip.params = result_bs.params
                    for nf in range(self.npeaks):
                        durations[nf].append(flarecopy.get_single_duration(nf))

                # Get percentiles from bootstrap
                self.result_nodip.params_bootstrap = {}
                for p in X.var_names:
                    pval = np.percentile(bs_pars[p], percentiles_3sigma)
                    self.result_nodip.params_bootstrap[p] = pval

        # Flare duration is estimated via bootstrap if no previous bootstrap is
        # performed
        self.durations = []
        if self.npeaks > 0 and type(uncertainties) == str:
            checkerr = np.array([self.result_nodip.params[pvar].stderr \
                                for pvar in self.result_nodip.var_names])
            checkerr = (checkerr == None).any()
            if checkerr: # Get just one value
                for nf in np.arange(self.npeaks):
                    self.durations.append(self.get_single_duration(nf))
            else:
                durations = {}
                for nf in range(self.npeaks):
                    durations[nf] = []
                # Random estimates of the flare profile
                flarecopy = copy.deepcopy(self)
                for _ in range(1000):
                    for bsp in flarecopy.result_nodip.var_names:
                        flarecopy.result_nodip.params[bsp].value \
                                = np.random.normal( \
                                loc=self.result_nodip.params[bsp].value, \
                                scale=self.result_nodip.params[bsp].stderr)
                    # For each iteration, get estimates for individual durations
                    for nf in np.arange(self.npeaks):
                        durations[nf].append(flarecopy.get_single_duration(nf))
                for nf in range(self.npeaks):
                    durations_unit = durations[nf][0].unit
                    durations_values = [x.value for x in durations[nf]]
                    self.durations.append(np.percentile(durations_values, \
                        percentiles_3sigma)*durations_unit)
        elif self.npeaks > 0 and type(uncertainties) == int:
            for nf in np.arange(self.npeaks):
                durations_unit = durations[nf][0].unit
                durations_values = [x.value for x in durations[nf]]
                self.durations.append(np.percentile(durations_values, \
                        percentiles_3sigma)*durations_unit)

        # If more than one flare is detected, look for possible QPP parameters
        if self.npeaks > 1:
            if verbose:
                print('--> QPP candidate parameters saved')
            self.QPP_candidate = {}
            self.QPP_candidate['oscill_ampl'] = oscill_ampl
            self.QPP_candidate['oscill_ampl_ratio'] = oscill_ampl_ratio
            self.QPP_candidate['oscill_ampl_ratio_err'] = oscill_ampl_ratio_err
            sort_tpeaks = np.sort(t_peaks_sol)
            self.QPP_candidate['oscill_P'] = np.mean(t_peaks_sol)
            self.QPP_candidate['oscill_P_err'] = np.std(t_peaks_sol)
            self.QPP_candidate['oneflare_profile'] = oneflare_profile
            self.QPP_candidate['oneflare_resid'] = oneflare_resid
            self.QPP_candidate['oneflare_model'] = oneflare_model

        # Equivalent duration for every flare (luminosity and/or independent
        # flare energy estimation are left for the result analysis phase, given
        # that that can be done in several ways).
        self.EDs = {}
        self.energies = {}
        for nsol in np.arange(self.npeaks):
            self.EDs[nsol] = self.get_flare_ED(nsol, double_t=True)
            if self.Tstar is not None and self.Rstar is not None:
                self.energies[nsol] \
                        = self.get_flare_energy(nsol, method='shibayama')

        if plots:
            plt.close('all')
            if self.npeaks > 0:
                if type(uncertainties) == str and uncertainties == 'LM':
                    print(lmfit.report_fit(self.result_nodip.params_LM))
                    bestpar = X.params
                if type(uncertainties) == str and uncertainties == 'emcee':
                    print(lmfit.report_fit(self.result_nodip.params_emcee))
                    bestpar = self.result_nodip.params_emcee
                elif type(uncertainties) == str and uncertainties == 'F-test':
                    lmfit.printfuncs.report_ci(ci)
                    bestpar = lmfit.Parameters()
                    for p in ci.keys():
                        bestpar.add(p, value=ci[p][3][1])
                elif type(uncertainties) == int:
                    bestpar = lmfit.Parameters()
                    print('\nLM result:\n')
                    print(lmfit.report_fit(self.result_nodip.params_LM))
                    print('\nBootstrap result:\n')
                    pp = copy.deepcopy(self.result_nodip.params_bootstrap)
                    for p in pp.keys():
                        pdiff = np.diff(pp[p])
                        print(p, r'= {:.2} + {:.2} - {:.2} ({:.2f}%)'.format( \
                            pp[p][2], pdiff[1], pdiff[2], \
                            np.mean([pdiff[1], pdiff[2]])/abs(pp[p][2])*100.))
                        bestpar.add(p, value=pp[p][2])
                model = self.ydata - self.result_nodip.residual*self.yerrdata
            else:
                model = np.polyval(self.result_line.params, tt)

            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].plot(tt, self.ydata, 'k', alpha=1.)
            if type(uncertainties) == str and uncertainties != 'LM' \
                or type(uncertainties) == int:
                axs[0].plot(tt, model, 'r', linewidth=3, label='Best LM model')
            axs[0].plot(tt, model, 'r', linewidth=3, label='Best model')
            axs[0].set_ylabel('Normalised flux', fontsize=14)
            axs[1].plot(tt, self.result_nodip.residual*self.yerrdata, 'k')
            axs[1].set_xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            axs[1].set_ylabel(r'Residuals', fontsize=14)
            plt.tight_layout()
            if plotname == '':
                plt.show()
                set_trace()
            else:
                plt.savefig(plotname)
            plt.close()

        return

    def fit_dip(self, xmin=0., xmax=0., plots=False, threshold=3., \
            bootstrap_uncertainties=-1):
        '''
        Fix all parameters except the continuum, compare models on this.

        bootstrap_uncertainties (int): if < 0, the LM errors or the estimated
        red noise level are used to determine the QUALITATIVE significance
        of the dip detection (amplitude wrt noise).
        '''
        if not 'result_nodip' in dir(self):
            print('You must first run fit_flare_profile to run the AIC tests.')
            return

        tt = self.tdata.value
        params = copy.deepcopy(self.result_nodip.params_LM)

        # Compare model with and without dip
        for p in params.keys():
            if p != 'f':
                params[p].vary = False
            else:
                params[p].vary = True

        fit_kws = {}
        fit_kws['niter'] = 10
        fit_kws['minimizer_kwargs'] = {}
        fit_kws['minimizer_kwargs']['method'] = 'powell'
        result_nodip_ = lmfit.minimize(models.mendoza_residuals, params, \
           args=(tt, self.ydata, self.yerrdata), \
           kws={'complexity': self.npeaks}, method='basinhopping', \
           nan_policy='omit', calc_covar=False, **fit_kws)

        for p in result_nodip_.params.keys():
            if str(p) not in ['Gdip', 'x0', 'w1', 'w2', 'n', 'f']:
                params[p].vary = False
            else:
                params[p].vary = True

        # Attempt several initial pars for the dip position
        chi2min = np.inf
        scramble = True
        scramble_i = 0

        while scramble:
            if xmin == 0. and xmax == 0.:
                params['x0'].value = np.random.uniform( \
                    low=params['x0'].min, high=params['x0'].max)
            else:
                params['x0'].min = xmin
                params['x0'].max = xmax
                params['x0'].value = np.random.uniform(low=xmin, high=xmax)

            fit_kws = {}
            fit_kws['niter'] = 10
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            mini = lmfit.Minimizer(models.mendoza_residuals, params, \
                fcn_args=(tt, self.ydata, self.yerrdata), \
                fcn_kws={'complexity': self.npeaks}, \
                nan_policy='omit', calc_covar=True)
            try:
                result_dip_ = mini.minimize(method='basinhopping', **fit_kws)
                scramble = False
            except ValueError:
                break
            except FloatingPointError:
                for p in params:
                    try:
                        params[p].value = np.random.uniform( \
                            low=params[p].min + 1e-5, high=params[p].max - 1e-5)
                    except OverflowError:
                        result_dip_ = copy.deepcopy(result_nodip_)
                        break
                scramble_i += 1
                if scramble_i > 100:
                    result_dip_ = copy.deepcopy(result_nodip_)
                    break
                else:
                    pass

            # Update solution if chi2 improves
            if iter == 0 or result_dip_.redchi < chi2min or scramble == False:
                result_dip = copy.deepcopy(result_dip_)

        dip_delta_bic = result_dip.bic - result_nodip_.bic
        result_dip = copy.deepcopy(result_dip_)

        if bootstrap_uncertainties < 0:
            # Compute QUALITATIVE significance on the basis of
            # estimated red noise level (see Pont+2006)
            dip_width = (result_dip.params['w1'] + result_dip.params['w2']) \
                        /abs(np.diff(tt)[np.diff(tt) != 0.]).min()
            rednoise_level = self.rednoise[1][ \
                abs(np.array(self.rednoise[0]) - dip_width).argmin()]
            dip_significance = abs(result_dip.params['Gdip'].value \
                                /rednoise_level)
        else:
            Gdip = []
            for _ in range(bootstrap_uncertainties):
                newy = []
                for i, yi in enumerate(self.ydata):
                    newy.append(yi + np.random.normal(scale=self.yerrdata[i]))
                bs_mini = lmfit.Minimizer(models.mendoza_residuals, \
                    result_dip_.params, fcn_args=(tt, newy, self.yerrdata), \
                    fcn_kws={'complexity': self.npeaks}, \
                    nan_policy='omit', calc_covar=False)
                result_bs = bs_mini.minimize(method='powell')
                Gdip.append(result_bs.params['Gdip'].value)
            Gdip_perc = np.percentile(Gdip, percentiles_3sigma)
            Gdip_perc_diff = np.diff(Gdip_perc)

            # Adding 1e-6 to avoid possible zeros in Gdip_perc_diff
            dip_significance = abs(Gdip_perc[3]/max([1e-6, Gdip_perc_diff[3], \
                        Gdip_perc_diff[2]]))

        if plots:
            plt.figure()
            plt.plot(tt, self.ydata)
            model = self.get_model_from_result(result_dip)
            plt.plot(tt, model)
            plt.xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            plt.ylabel('Normalised flux', fontsize=14)
            plt.show()
            set_trace()
            plt.close()

        return result_dip, dip_delta_bic, dip_significance

    def get_single_profile(self, nsol, model='mendoza', double_t=True):
        '''
        Computes an individual flare profile using fitted parameters.

        Parameters
        ----------
        double_t: whether to double the length of the time axis in order to
        avoid cut flare profiles.
        '''

        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

        if double_t:
            tt = np.arange(self.tdata.min().value, self.tdata.max().value*2, \
                    np.diff(self.tdata).min().value)
        else:
            tt = np.copy(self.tdata.value)

        param_i = lmfit.Parameters()
        X = copy.deepcopy(self.result_nodip)

        if model == 'davenport':
            param_i.add('t00', value=X.params['t0' + str(nsol)].value)
            param_i.add('A0', value=X.params['A' + str(nsol)].value)
            param_i.add('tau10', value=X.params['tau1' + str(nsol)].value)
            param_i.add('tau20', value=X.params['tau2' + str(nsol)].value)
            param_i.add('t120', value=X.params['t12' + str(nsol)].value)
            param_i.add('taurise0', value=X.params['taurise' + str(nsol)].value)
            thisflare = models.exp_doubledecay(self.tdata.value, param_i, \
                    complexity=1)

        elif model == 'mendoza':
            param_i.add('tpeak0', value=X.params['tpeak' + str(nsol)].value)
            param_i.add('ampl0', value=X.params['ampl' + str(nsol)].value)
            param_i.add('fwhm0', value=X.params['fwhm' + str(nsol)].value)
            param_i.add('D20', value=X.params['D2' + str(nsol)].value)
            thisflare = models.flare_model_mendoza(tt, param_i, complexity=1)

        return tt*self.tdata.unit, thisflare

    def get_full_profile(self, result, use_residuals=True, mode='mendoza'):
        '''
        The model without residuals has no polynomial added.
        '''
        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

        if use_residuals:
            if mode == 'mendoza':
                return self.get_model_from_result(result)
            else:
                y = self.ydata/self.ydata.max()
                yerr = self.yerrdata/self.ydata.max()
                return (y - result.residual*yerr)*self.ydata.max()
        else:
            tt = (self.tdata/self.t12).value
            return models.exp_convolved(tt, result.params, self.npeaks) \
                + models.flare_dip(result.params, tt)

    def get_single_duration(self, n, double_t=True):
        '''
        Flare #n (in case of complex profile) duration based
        on pre-determined noise level.
        '''

        tt, thisflare = self.get_single_profile(n, double_t=double_t)

        above_noise = thisflare > self.noise_level
        if np.sum(above_noise) > 0:
            flare_start = tt[above_noise][0]
            flare_end = tt[above_noise][-1]
            return flare_end - flare_start
        else:
            tunit = tt.unit
            return 0.*tunit

    def plot_models(self, plot_instance=None, mode='mendoza', \
            title='', showplot=False, plot_LS=False, plot_raw_flux=False, \
            plot_dip=True, plot_cme=True, figsize=(13, 5)):

        # Generate frames
        left, width = 0.1, 0.85
        bottom, height = 0.5, 0.4

        if plot_LS:
            rect_flare = [left, bottom, width, height]
            rect_resid = [left, bottom - 0.1, width, 0.1]
            rect_periodo = [left, bottom - 0.4, width, 0.2]
            fig = plt.figure(figsize=(20, 10))
        else:
            rect_flare = [left, 0.35, width, 0.55]
            rect_resid = [left, 0.15, width, 0.2]
            fig = plt.figure(figsize=figsize)

        axflare = plt.axes(rect_flare)
        axresid = plt.axes(rect_resid)
        if plot_LS:
            axperiodo = plt.axes(rect_periodo)

        # Plot data and models
        tplot = self.tdata.to(u.min).value
        axflare.plot(tplot, self.ydata, 'k')
        axresid.plot(tplot, self.result_nodip.residual, 'k')

        quiet = np.polyval([self.result_nodip.params['d'], \
            self.result_nodip.params['e'], self.result_nodip.params['f']], \
            self.tdata.value)
        for nsol in np.arange(self.npeaks):
            tt, thisflare = self.get_single_profile(nsol, double_t=False)
            axflare.plot(tplot, quiet + thisflare)

        if plot_dip:
            model_dip = self.get_full_profile(self.result_dip)
            axflare.plot(tplot, model_dip, linewidth=3, \
                label=r'Dip $\Delta$BIC = {:.2f}'.format(self.dip_delta_bic))

        if plot_cme:
            model_cme = self.get_full_profile(self.result_cme)
            axflare.plot(tplot, model_cme, linewidth=3, \
                label=r'CME $\Delta$BIC = {:.2f}'.format(self.cme_delta_bic))

        if self.npeaks == 1:
            label = str(self.npeaks) + ' flare'
        else:
            label = str(self.npeaks) + ' flares'

        chilab = 'Cumulative profile (' + label + r', $\chi_r^2=$ {:.1f})'
        model_nodip = self.get_full_profile(self.result_nodip)
        axflare.plot(tplot, model_nodip, linewidth=3, \
            label=chilab.format(self.result_nodip.redchi))
        axflare.set_xticks([])
        if mode == 'mendoza':
            axflare.plot(tplot, np.polyval([self.result_nodip.params['d'], \
                self.result_nodip.params['e'], self.result_nodip.params['f']], \
                self.tdata.value), label='Quiet stellar flux')
        else:
            axflare.plot(tplot, np.polyval([self.result_nodip.params['d'], \
                self.result_nodip.params['e'], self.result_nodip.params['f']], \
                (self.tdata/self.t12).value)*self.ydata.max(), \
                label='Quiet stellar flux')
        if plot_raw_flux:
            axflare.plot(tplot, self.yraw/self.yraw.max() - 1., 'k', \
                    alpha=0.4, label='Scaled raw flux')
        axresid.set_xlabel('Time since peak [min]', fontsize=16)
        axflare.set_ylabel('Normalised flux - 1', fontsize=16)
        axresid.set_ylabel('Residuals', fontsize=16)

        # If a QPP candidate with at least three peaks was found, plot
        # the model with just one flare and the residuals from the QPP model
        if 'QPP_candidate' in dir(self) and self.npeaks > 3:
            axflare.plot(tplot, self.QPP_candidate['oneflare_profile'], '--', \
                        c='turquoise', label='One-flare model')
            left, bottom, width, height = [0.7, 0.62, 0.2, 0.2]
            ax_qpp = fig.add_axes([left, bottom, width, height])
            ax_qpp.plot(tplot, self.QPP_candidate['oneflare_resid'], 'k')
            ax_qpp.set_title('Residuals of one-flare model', fontsize=12)

        axflare.legend(frameon=False, prop={'size': 12})

        if plot_LS:
            axperiodo.set_xlabel('Frequency [mHz]', fontsize=16)
            axperiodo.set_ylabel('Power', fontsize=16)

        if plot_LS:
            try:
                # Need to generate LS first, now lacking
                freq, power = self.ls_residuals.autopower(nyquist_factor=0.5)
                falev = self.ls_residuals.false_alarm_level(0.01)
                axperiodo.semilogx(freq.value*1e3, power)
                axperiodo.semilogx([freq[0].value*1e3, freq[-1].value*1e3], \
                        [falev, falev], 'r--', label='1 % FAP')
                axperiodo.legend(loc='upper right', prop={'size': 14})
            except FloatingPointError:
                pass

        axflare.set_title(title, fontsize=18)
        plt.tight_layout()
        if plot_instance is not None and not showplot:
            plot_instance.savefig(fig)
        else:
            plt.show()
            set_trace()

        return

    def get_flare_energy(self, npeak, method='shibayama', double_t=True):
        '''
        Given a flare profile fit and temperature, computes total energy output.
        Formulae from Shubayama et al. (2013) and Davenport et al. (2014).
        See also https://arxiv.org/pdf/1810.03277.pdf

        The Equivalent Duration can be split for the fast rise-fast decay
        and for the gradual decay phase.

        Parameters
        ----------
        tprof [array]: must be in seconds
        wth, fth: filter throughput (wth in A)
        rms: if >0, only the flux above the rms level is considered for energy
        determination.
        double_t: whether to double the time time axis duration to avoid
        cutting flare profiles. If True, the time axis must be measured in days.
        '''
        tt, thisflare = self.get_single_profile(npeak, double_t=double_t)

        if method == 'shibayama':
            bbstar = BlackBody(self.Tstar)(self.wth)
            bbflare = BlackBody(self.Tflare)(self.wth)
            lumratio = trapezoid(bbstar*self.fth, x=self.wth) \
                        / trapezoid(bbflare*self.fth, x=self.wth)
            Aflare_t = thisflare*np.pi*self.Rstar**2*lumratio
            Lflare = constants.sigma_sb*(self.Tflare**4)*Aflare_t
            flare_energy = trapezoid(Lflare, x=tt.to(u.s)).to(u.erg)

        elif method == 'davenport':
            # Get also ED for slow and fast decay parts
            fwhm = self.result_nodip.params['fwhm' + str(npeak)].value*u.day
            tfast = tt.to(u.s) <= fwhm.to(u.s)
            EDfast = self.get_flare_ED(npeak, double_t=double_t, flag=tfast)
            EDslow = self.get_flare_ED(npeak, double_t=double_t, flag=~tfast)
            ED = self.get_flare_ED(npeak, double_t=double_t)
            if 'stellar_luminosity' in dir(self):
                flare_energy = ED*self.stellar_luminosity
            else:
                print('You must compute the quiescent stellar ' \
                        + 'luminosity first.')
                set_trace()

        return flare_energy

    def get_flare_ED(self, npeak, double_t=True, flag=[]):
        '''
        Calculate equivalent duration for a flare.
        The time array can be doubled in case the flare profile is cut because
        of lack of data.
        '''

        tt, thisflare = self.get_single_profile(npeak, double_t=double_t)

        if len(flag) == 0:
            flag = np.full(len(tt), True)

        yprof = thisflare[flag]
        tprof = tt[flag]
        ED = trapezoid(yprof, x=tprof.to(u.s))

        return ED

    def get_stellar_luminosity(self, instrument, mag, distance):
        '''
        Use stellar magnitude, distance, and instrument zero point (from SVO
        service) to convert magnitude to quiescent stellar luminosity for a
        given fitted peak. Vega mag is assumed to be 0.
        '''
        if instrument == 'CHEOPS': # using Gaia G bandpass
            zeropoint = 2.49769e-9
            lambdaeff = 5850.88
        if instrument == 'TESS':
            zeropoint = 1.33161e-9
            lambdaeff = 7452.64

        # Flux densities --> fluxes with effective wavelength
        F0 = zeropoint*u.erg/u.cm**2/u.s/u.A
        F = F0*10**(-mag/2.5)*lambdaeff*u.A
        L_quiesc = 4.*np.pi*distance.to(u.cm)**2*F
        self.stellar_luminosity = L_quiesc

        return self.stellar_luminosity
