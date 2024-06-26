#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:07:12 2024

@author: pqt7tv
"""

# Import modules
import os
from scipy.io import readsav
import numpy as np
import corner
import lmfit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class GaussianFitter:
    def __init__(self, spectrum_file, z=0, l='', burn_in=1000, thinning=3):
        self.spectrum_file = spectrum_file
        self.z = z
        self.l = l
        #self.n_steps = n_steps  # Total number of MCMC steps
        self.burn_in = burn_in  # Number of burn-in steps
        self.thinning = thinning  # Thinning parameter for MCMC
    
    # Data Preparation
    def read_file(self):
        """
        Reads an sav file containing two columns: observed wavelength and corresponding flux (smoothed).
    
    
        Returns:
        - name (str): The name of the supernova extracted from the file name.
        - date (str): The observation date extracted from the file name.
        - wavelengths (list of float): A list of observed wavelengths.
        - fluxes (list of float): A list of corresponding flux values.
        
        Raises:
        - IOError: If there is an error reading the file.
        """
        try:
            # Extract name and date from the file path
            file_name = os.path.basename(self.spectrum_file)
            name = file_name.split("_")[0]
            date = file_name.split("_")[1]
            
            # Read the ASCII file
             
            spec = readsav(self.spectrum_file)
            
            # Normalize the flux by dividing by the mean flux
            flux = spec['f_ft'] / np.mean(spec['f_ft'])        
            
            # Apply redshift correction to wavelengths
            wv = spec['w'] / (1 + self.z)
            
            flux = np.array(flux)
            wv = np.array(wv)
            
            print(f"File is accepted '{self.spectrum_file}'")
            return name, date, wv, flux
        
        except Exception as e:
            print(f"Error reading file '{self.spectrum_file}': {str(e)}")
            return None, None, None, None

    
    # Specify the Region of the Spectrum
    def select_line(self):
        """
        Selects the region of the spectrum corresponding to a specified absorption 
        feature by returning its wavelength limits and rest wavelength.
    
        Returns:
        - tuple: A tuple (w0, w1, rest_wavelength) representing the left and right 
        wavelength limits and the rest wavelength of the absorption feature. 
        If the specified line is not found in the predefined list, it returns 
        (0, 0, 0) as default values.
        """
        # Dictionary mapping absorption feature names to corresponding limits
        absorption_limits = {
            'CaII': (3550, 3900, 3945.),
            'SIIblue': (5000, 5650, 5454.),
            'SIIred': (5000, 5650, 5640.),
            'SiII': (6000, 6350, 6355.),
            'HeI4471': (4200, 4600, 4471.),
            'HeI5876': (5000, 6000, 5876.),
            'HeI6678': (6100, 6540, 6678.),
            'HeI7065': (6700, 7050, 7065),
            'HeI7281': (7000, 7300, 7281),
            'FeII5169': (4200, 5300, 5169),
            'FeII5018': (4750, 5000, 5018),
            'Halpha': (6000, 6400, 6564),
            'IR_CaII': (7500, 8500, 8579.),}
    
        return absorption_limits.get(self.l, (0, 0, 0))  # Default to zeros if line not found

    def create_plot(self, title):
        """
        Create a customized plot for spectral analysis.

        Parameters:
        -title (str): Title for the plot.

        Returns:
        -matplotlib.figure.Figure: Initialized figure.
        """
        plt.figure(figsize=(15, 5))
        plt.minorticks_on()
        plt.tick_params('both', length=12, width=2, which='major', direction='in', top=True, right=False)
        plt.tick_params('x', length=8, width=1, which='minor', direction='in', top=True, right=False)
        plt.ylabel(r"Relative Flux", fontsize=18)
        plt.xlabel(r"Rest Wavelength [$\AA$]", size=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title, fontdict={'fontsize': 15})
        return plt.gcf()
    
    # Select the wavelength range around the absorption
    def get_bounds(self, name, date, wv, flux):
        """
        Get wavelength bounds by selecting two points on a spectrum plot.
    
        Parameters:
        - name (str): Name or identifier for the spectrum.
        - date (str): Observation date extracted from the file name.
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of corresponding flux values.
    
        Returns:
        - list: List of two selected points (x, y) on the plot.
        """
        plot = self.create_plot(f"{name}  date = {date}")
        plt.figure(plot.number)  # Activate the figure window
        wv_l, wv_h, lam_s = self.select_line()
        msk = (wv > wv_l) & (wv < wv_h)
        wv_zoomed = wv[msk]
        flux_zoomed = flux[msk]
        plt.plot(wv_zoomed, flux_zoomed, label ='Zoomed-In Reduced Spectrum')
        plt.legend()
    
        x = plt.ginput(2, timeout=-1)  # Select two points (press Enter after each selection)
    
        return x 

    def selected_range(self, name, date, wv, flux):
        """
        Returns the selected wavelength and corresponding flux.
    
        Parameters:
        - name (str): Name or identifier for the spectrum.
        - date (str): Observation date extracted from the file name.
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of corresponding flux values.
    
        Returns:
        - tuple: A tuple containing arrays of selected wavelength and corresponding flux.
        """
        bounds = self.get_bounds(name, date, wv, flux)
        plt.scatter(bounds[0][0], bounds[0][1], c='red')
        plt.scatter(bounds[1][0], bounds[1][1], c='red')
        plt.draw()
        msk = np.logical_and(bounds[0][0] < wv, bounds[1][0] > wv)
        wvl_bins = np.argwhere(msk)
        wvc = wv[wvl_bins]
        fc = flux[wvl_bins]
        plt.plot(wvc, fc, c='red', label='Selected Region')
        plt.legend()
        plt.show(block=False)
        return wvc, fc
    
    # Model Definition
    def model(self, wv, flux, wvc, parameters):
        """
        Define a Gaussian model for an absorption feature in spectral data.
    
        Parameters:
        - wv (array-like): Array of wavelengths.
        - flux (array-like): Array of flux values.
        - wvc: selected wavelength region
        - parameters (list): List of Gaussian model parameters [amplitude, central wavelength, standard deviation].
    
        Returns:
        - msk (array-like): Indices of the selected wavelength range.
        - absorption_model (array-like): The Gaussian absorption model evaluated at the given wavelengths.
        """
        # Apply the wavelength range parameter
        a, mu, sigma = parameters
        msk = np.logical_and(wv >= np.min(wvc), wv <= np.max(wvc))
        absorption_model = -1 * a * np.exp(-0.5 * ((wv[msk] - mu) / sigma) ** 2) + (np.min(flux[msk]) + a)

        return msk, absorption_model
    
    # Relativistic Doppler Formula
    def relativistic_doppler(self, velocity=None, wavelength=None, c=299792.458):
        """
        Convert between velocity and wavelength using the relativistic Doppler formula.

        Parameters:
        - velocity (float, optional): Velocity in km/s. Positive for recession velocity, negative for blueshift.
        - wavelength (float, optional): Wavelength in Angstroms. Positive for redshift, negative for blueshift.
        - c (float, optional): Speed of light in km/s. Default is the speed of light in vacuum.

        Returns:
        - float: The corresponding wavelength if velocity is given, or the corresponding velocity if wavelength is given.
        """
        _,_, lam_s = self.select_line()
        if velocity is not None:
            # Calculate wavelength from velocity
            lam_o = lam_s * np.sqrt((1 + velocity / c) / (1 - velocity / c))
            return lam_o
        elif wavelength is not None:
            # Calculate velocity from wavelength
            velocity = c * ((wavelength / lam_s)**2 - 1) / ((wavelength / lam_s)**2 + 1)
            return velocity
        else:
            raise ValueError("Please provide either velocity or wavelength.")
    
    # Prior Distribution
    def uniform_prior(self, low, high):
        """
        Uniform prior distribution function.

        Parameters:
        - low: Lower bound of the uniform distribution.
        - high: Upper bound of the uniform distribution.

        Returns:
        - pdf: Function representing the uniform prior probability density.
        """
        def pdf(x):
            if low <= x <= high:
                return 1 / (high - low)
            else:
                return 0
        return pdf
    
    def combined_prior(self):
        """
        Combined prior distribution function for parameters: mu, sigma, amplitude, and delta_w.

        Returns:
        - prior: Function representing the combined prior probability density.
        """
        c=299792.458
        amp_low = 0.05
        amp_high = 0.6
        mu_low = self.relativistic_doppler(velocity = -40000)
        mu_high = self.relativistic_doppler(velocity = -8000)
        _, _, lam_s = self.select_line()
        sigma_low = lam_s * (5000/c)
        sigma_high = lam_s * (20000/c)
        mu_prior = self.uniform_prior(mu_low, mu_high)
        sigma_prior = self.uniform_prior(sigma_low, sigma_high)
        amp_prior = self.uniform_prior(amp_low, amp_high)

        def prior(parameters):
            amp, mu, sigma = parameters
            return mu_prior(mu) * sigma_prior(sigma) * amp_prior(amp)

        return prior
    
    # Likelihood Function
    def likelihood(self, wv, flux, wvc, parameters):
        """
        Calculate the likelihood of the observed flux given the model parameters.
    
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
        - wvc: selected wavelength region
        - parameters (list): List containing [amplitude, central_wavelength, sigma] of the Gaussian model.
    
        Returns:
        - likelihood (float): The likelihood of the observed flux given the model parameters.
        """
        a, mu, sigma = parameters
        
        msk, absorption_model = self.model(wv, flux, wvc, parameters)
        observed_data = flux[msk]
        # Calculate the likelihood as the product of probabilities at each wavelength
        epsilon = 1e-10  # Small epsilon to prevent division by zero
        likelihood = np.prod(np.exp(-0.5 * ((observed_data - absorption_model) / (sigma + epsilon)) ** 2)) / np.sqrt(2 * np.pi * (sigma**2 + epsilon))
        return likelihood
    
    # MCMC Sampling
    def proposal(self, wv, flux, wvc, initial_parameters, iterations, step_sizes):
        """
        Metropolis-Hastings algorithm to sample from the posterior distribution of parameters.
    
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
        - wvc: selected wavelength region
        - initial_parameters: list, initial guesses for the parameters [amplitude, mu, sigma, delta_w].
        - iterations: int, total number of iterations including burn-in.
        - step_sizes: list, step sizes for each parameter [amp_step, mu_step, sigma_step, delta_w_step].
    
        Returns:
        - samples: list of tuples, sampled parameter 
        values [amplitude, mu, sigma, delta_w] from the posterior distribution.
        """
        current_parameters = initial_parameters
        samples = []
    
        for i in range(iterations):
            # Propose new parameters
            proposed_parameters = [np.random.normal(current_parameters[i], step_sizes[i]) for i in range(3)]
            acceptance_ratio = self.likelihood(wv, flux, wvc, proposed_parameters) / self.likelihood(wv, flux, wvc, current_parameters)
    
            # Accept or reject the proposal
            if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                current_parameters = proposed_parameters
                if i >= self.burn_in and i % self.thinning == 0:
                    samples.append(current_parameters)  # Only append accepted proposals
        return samples

    
    # Posterior Distribution
    def posterior_analysis(self, wv, flux, wvc, initial_param, num_iterations):
        """
        Perform posterior analysis using the Metropolis-Hastings MCMC algorithm.
        
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
        - wvc: selected wavelength region
        - initial_param (float or array-like): The starting parameter value(s) for the MCMC algorithm.
        - num_iterations (int): The number of iterations to run the MCMC algorithm.
        
        Returns:
        - accepted_samples (list): A list of accepted parameter values sampled from the posterior distribution.
        """
        current_param = initial_param
        samples = []
        step_sizes = [0.1, 100, 10]
        prior = self.combined_prior()
        for _ in range(num_iterations):
            proposed_params = self.proposal(wv, flux, wvc, current_param, 10000, step_sizes)
            for proposed_param in proposed_params:
                prior_proposed = prior(proposed_param)
                prior_current = prior(current_param)
                likelihood_proposed = self.likelihood(wv, flux, wvc, proposed_param)
                likelihood_current = self.likelihood(wv, flux, wvc, current_param)
                acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current)
                if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                    current_param = proposed_param
                samples.append(current_param)  # Append only accepted proposal
        print("Posterior analysis is complete.")
        return samples
    
    
    # Set Initial Parameters
    def set_initial_param(self, wvc, fc):
        """
        Set initial parameters for the MCMC algorithm.
        
        Parameters:
        - wvc: selected wavelength region
        - fc: flux to the selected wavelength range

        Returns:
        - list: Initial parameter values [amplitude, mu, sigma].
        """
        c=299792.458
        a = 0.1
        mu_low = self.relativistic_doppler(velocity = -40000)
        mu_high = self.relativistic_doppler(velocity = -8000)
        mu = ((mu_high-mu_low)/2) + mu_low
        _, _, lam_s = self.select_line()
        sigma_low = lam_s * (5000/c)
        sigma_high = lam_s * (20000/c)
        sigma = (sigma_high-sigma_low)/2
        delta_w = np.max(wvc) - np.min(wvc)
        init_param = [a, mu, sigma]
        # Print initial parameter values
        print("Selected delta_w:", delta_w)
        print("Initial parameter values:")
        print("Amplitude:", a)
        print("Mu:", mu)
        print("Sigma:", sigma)
        return init_param
    
    # Fit a Gaussian 
    # Define the Gaussian function
    def gaussian(self, x, amp, mu, sigma, baseline):
        """
        Calculate the value of an inverted Gaussian function at given x values.

        Parameters:
        - x (array-like): Independent variable values.
        - amp (float): Amplitude of the Gaussian function.
        - mu (float): Mean of the Gaussian function.
        - sigma (float): Standard deviation of the Gaussian function.
        - baseline (float): Baseline value added to the Gaussian function.

        Returns:
        - array-like: Values of the Gaussian function at the given x values.
        """
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + baseline

    # Function to fit the Gaussian model to the data
    def fit_gaussian(self, wvc, fc):
        """
        Fit an inverted Gaussian model to the given data.

        Parameters:
        - wvc (array-like): Independent variable values (e.g., wavelengths).
        - fc (array-like): Dependent variable values (e.g., flux).

        Returns:
        - tuple: A tuple containing the fitted parameters (amp, mu, sigma, baseline)
                 and the covariance matrix.
        """
        # Create a new model
        model = lmfit.Model(self.gaussian)
        
        # Set initial parameter values
        c=299792.458
        a = 0.1
        mu_low = self.relativistic_doppler(velocity = -40000)
        mu_high = self.relativistic_doppler(velocity = -8000)
        mu = ((mu_high-mu_low)/2) + mu_low
        _, _, lam_s = self.select_line()
        sigma_low = lam_s * (5000/c)
        sigma_high = lam_s * (20000/c)
        sigma = (sigma_high-sigma_low)/2
        params = model.make_params(amp=-a, mu=mu, sigma=sigma, baseline=np.mean(fc))
        
        # Perform the fit
        result = model.fit(fc, params, x=wvc)
        
        # Get the fitted parameters and covariance matrix
        fit_params = [result.best_values[param] for param in ['amp', 'mu', 'sigma', 'baseline']]
        covariance = result.covar
        
        plt.plot(wvc, self.gaussian(wvc, *fit_params), label="lmfit Fitted Gaussian", linestyle="--")
        plt.draw()
        plt.show(block=False)
        return fit_params, covariance
    
    def plot_results(self, wv, flux, wvc, posterior_mean):
        """
        Plot the spectrum with the fitted Gaussian model.
    
        Parameters:
        - wv (array-like): Wavelength values of the spectrum.
        - flux (array-like): Flux values of the spectrum.
        - wvc (array-like): Wavelength values used for fitting the Gaussian model.
        - posterior_mean (array-like): Posterior mean of the Gaussian parameters obtained from MCMC.
    
        Returns:
        - None
    
        This function plots the spectrum with the fitted Gaussian model overlaid. It uses the model function
        to calculate the model flux based on the posterior mean of the Gaussian parameters. The plot is displayed
        and saved as an image file with the name '{name}_{date}_fitted_gaussian.png'. The plot window is not
        blocked to allow for interactive use, and a brief pause is added to allow time for the plot window to refresh.
        """
        msk, model_flux = self.model(wv, flux, wvc, posterior_mean)
        plt.plot(wv[msk], model_flux, label='Fitted Gaussian Model', color='blue')
        plt.legend()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)  # Allow time for plot window to refresh
        plt.savefig(f'{name}_{date}_fitted_gaussian.png')  # Save the plot as an image file
        plt.close()
    
    def plot_trace(self, name, date, parameter_samples, parameter_names):
        """
        Plot the trace plots for MCMC parameters.
        
        Parameters:
        - parameter_samples (list of lists): List of parameter samples from the MCMC chain.
        - parameter_names (list of str): List of parameter names.
        """
        num_params = len(parameter_names)
        iterations = len(parameter_samples)
        
        fig, axes = plt.subplots(num_params, 1, figsize=(8, 4*num_params))
        
        for i in range(num_params):
            param_samples = [sample[i] for sample in parameter_samples]
            axes[i].plot(np.arange(iterations), param_samples, color='blue', alpha=0.6)
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel(parameter_names[i])
            axes[i].set_title(f'Trace plot of {parameter_names[i]}')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{name}_{date}_trace_plots.png')
        plt.close()
    
    # Check goodness of the function
    def chi_square(self, observed_data, expected_data):
        """
        Calculate the chi-square goodness-of-fit statistic.
    
        Parameters:
        - observed_data (array-like): Array of observed data.
        - expected_data (array-like): Array of expected data (model predictions).
    
        Returns:
        - chi_sq (float): The chi-square statistic.
        """
        # Ensure the observed and expected data have the same length
        assert len(observed_data) == len(expected_data), "Observed and expected data must have the same length"
        
        # Calculate the sum of squared differences between observed and expected data
        squared_diffs = (np.array(observed_data) - np.array(expected_data)) ** 2
        
        # Calculate the chi-square statistic
        chi_sq = np.sum(squared_diffs / np.array(expected_data))
        
        
        return chi_sq
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit Gaussian model to supernova spectrum.")
    parser.add_argument("spectrum_file", type=str, help="Path to spectrum file")
    parser.add_argument("-z", "--redshift", type=float, default=0, help="Galaxy redshift")
    parser.add_argument("-l", "--line", type=str, default=None, help="Wavelength of the absorption line")
    args = parser.parse_args()
    
    fitter = GaussianFitter(args.spectrum_file, args.redshift, args.line)
    name, date, wv, flux = fitter.read_file()
    wvc, fc = fitter.selected_range(name, date, wv, flux)
    wvc = np.array(wvc).flatten()
    fc = np.array(fc).flatten()
    _, _, lam_s = fitter.select_line()
    initial_param = fitter.set_initial_param(wvc, fc)
    samples = fitter.posterior_analysis(wv, flux, wvc, initial_param, num_iterations = 150)
    posterior_mean = np.mean(samples, axis=0)
    print("Posterior mean of parameters:", posterior_mean)
    
    # velocity and associated uncertainty calculation
    velocity = fitter.relativistic_doppler(wavelength = posterior_mean[1])
    print(f"Calculated Velocity: {velocity}")
    unc = lam_s * ((np.std(samples, axis=0))[1]/299792.458)
    print(f"Calculated uncertainty: {unc}")
    
    # chi_square calculation
    _, model_flux = fitter.model(wv, flux, wvc, posterior_mean)
    chi_square = fitter.chi_square(fc, model_flux)
    print(f"Chi-square value: {chi_square}")
    
    # lmfit model
    fit_params, covariance = fitter.fit_gaussian(wvc, fc)
    vel_sci = fitter.relativistic_doppler(wavelength = fit_params[1])
    unc_sci = lam_s * (np.sqrt(covariance[1][1])/299792.458)
    chi_square_lm = fitter.chi_square(fc, fitter.gaussian(wvc, *fit_params))
    print("Fitted parameters from lmfit:", fit_params)
    print(f"Calculated Velocity (lmfit): {vel_sci}")
    print(f"Calculated Uncertainty (lmfit): {unc_sci}")
    print(f"Chi-square value (lmfit): {chi_square_lm}")
    
    
    #plots
    fitter.plot_results(wv, flux, wvc, posterior_mean)
    parameter_names = ['a', 'mu', 'sigma']
    fitter.plot_trace(name, date, samples, parameter_names)
    corner_plot = corner.corner(np.array(samples), labels=['amp', 'mu', 'sigma'])
    plt.savefig(f'{name}_{date}_corner_plot.png')
    
    # Output file
    output_filename = name + "_" + date + "_" + "results.txt"
    with open(output_filename, "w") as f:
        f.write("Parameter Values:\n")
        for parameter_name, param_value in zip(["Amplitude", "Mu", "Sigma"], posterior_mean):
            f.write(f"{parameter_name}: {param_value}\n")
        f.write(f"Velocity: {velocity} km/s\n")
        f.write(f"Uncertainty: {unc}\n")
        f.write(f"Chi-Square: {chi_square}\n")