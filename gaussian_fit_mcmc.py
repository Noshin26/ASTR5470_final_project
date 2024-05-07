#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:07:12 2024

@author: pqt7tv
"""

# Import modules
import os
from astropy.io import ascii
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class GaussianFitter:
    def __init__(self, spectrum_file, z=0, l='', n_steps=10000, burn_in=0, thinning=2):
        self.spectrum_file = spectrum_file
        self.z = z
        self.l = l
        self.n_steps = n_steps  # Total number of MCMC steps
        self.burn_in = burn_in  # Number of burn-in steps
        self.thinning = thinning  # Thinning parameter for MCMC
    
    # Data Preparation
    def read_file(self):
        """
        Reads an ASCII file containing two columns: observed wavelength and corresponding flux.
    
        This function reads an ASCII file containing spectral data. It extracts the name of the supernova and the observation date from the file name. 
        The observed wavelengths and corresponding flux values are extracted from the file and returned.
    
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
            spec = ascii.read(self.spectrum_file)
            
            # Normalize the flux by dividing by the mean flux
            flux = spec['col2'] / np.mean(spec['col2'])
            
            # Apply redshift correction to wavelengths
            wv = spec['col1'] / (1 + self.z)
            
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
    def model(self, wv, flux, parameters):
        """
        Define a Gaussian model for an absorption feature in spectral data.
    
        Parameters:
        - wv (array-like): Array of wavelengths.
        - flux (array-like): Array of flux values.
        - parameters (list): List of Gaussian model parameters [amplitude, central wavelength, standard deviation, wavelength-range parameter].
    
        Returns:
        - msk (array-like): Indices of the selected wavelength range.
        - absorption_model (array-like): The Gaussian absorption model evaluated at the given wavelengths.
        """
        # Apply the wavelength range parameter
        a, mu, sigma, delta_w = parameters
        lower_bound = mu - (delta_w * 0.5)
        upper_bound = mu + (delta_w * 0.5)
        msk = np.logical_and(wv >= lower_bound, wv <= upper_bound)
        if np.any(msk):
        # Calculate the absorption model only if msk is not empty
            absorption_model = -1 * a * np.exp(-0.5 * ((wv[msk] - mu) / sigma) ** 2) + (np.min(flux[msk]) + a)
        else:
            # Handle the case where msk is empty (e.g., print a message or handle it differently)
            print("Warning: No data points in the selected wavelength range.")
            print("Lower bound:", lower_bound)
            print("Upper bound:", upper_bound)
        
        # Calculate the Gaussian absorption model
        #absorption_model = -1 * a * np.exp(-0.5 * ((wv[msk] - mu) / sigma) ** 2) + (np.min(flux[msk]) + a)

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
    
    def gaussian_prior(self, mu, sigma):
        """
        This function returns a Gaussian prior distribution function.
        
        Parameters:
        - mu (float): the mean of the Gaussian prior.
        - sigma (float): the standard deviation of the Gaussian prior.
        
        Returns:
        - pdf: a function representing the Gaussian prior probability density.
        """
        def pdf(x):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-0.5) * ((x - mu) / sigma) ** 2)
        return pdf
    
    def combined_prior(self):
        """
        Combined prior distribution function for parameters: mu, sigma, amplitude, and delta_w.

        Returns:
        - prior: Function representing the combined prior probability density.
        """
        c=299792.458
        amp_low = 0
        amp_high = 3
        mu_low = self.relativistic_doppler(velocity = -20000)
        mu_high = self.relativistic_doppler(velocity = -5000)
        sigma_low = 0
        _, _, lam_s = self.select_line()
        sigma_high = lam_s * (20000/c)
        delta_w_mu = 0
        delta_w_sigma = (100/3)
        mu_prior = self.uniform_prior(mu_low, mu_high)
        sigma_prior = self.uniform_prior(sigma_low, sigma_high)
        amp_prior = self.uniform_prior(amp_low, amp_high)
        delta_w_prior = self.gaussian_prior(delta_w_mu, delta_w_sigma)

        def prior(parameters):
            amp, mu, sigma, delta_w = parameters
            return mu_prior(mu) * sigma_prior(sigma) * amp_prior(amp) * delta_w_prior(delta_w)

        return prior
    
    # Likelihood Function
    def likelihood(self, wv, flux, parameters):
        """
        Calculate the likelihood of the observed flux given the model parameters.
    
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
        - parameters (list): List containing [amplitude, central_wavelength, sigma, delta_w] of the Gaussian model.
    
        Returns:
        - likelihood (float): The likelihood of the observed flux given the model parameters.
        """
        a, mu, sigma, delta_w = parameters
        
        msk, absorption_model = self.model(wv, flux, parameters)
        observed_data = flux[msk]
        # Calculate the likelihood as the product of probabilities at each wavelength
        epsilon = 1e-10  # Small epsilon to prevent division by zero
        likelihood = np.prod(np.exp(-0.5 * ((observed_data - absorption_model) / (sigma + epsilon)) ** 2)) / np.sqrt(2 * np.pi * (sigma**2 + epsilon))
        return likelihood
    
    # MCMC Sampling
    def proposal(self, wv, flux, initial_parameters, iterations, step_sizes):
        """
        Metropolis-Hastings algorithm to sample from the posterior distribution of parameters.
    
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
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
            proposed_parameters = [np.random.normal(current_parameters[i], step_sizes[i]) for i in range(4)]
            #proposed_parameters[3] = max(100, proposed_parameters[3])  # Enforce > 100 A
            #proposed_parameters[3] = min(proposed_parameters[3], 3000)  # Enforce < 3000 A
            lower_bound = proposed_parameters[1] - (proposed_parameters[3] * 0.5)
            upper_bound = proposed_parameters[1] + (proposed_parameters[3] * 0.5)
            msk = np.logical_and(wv >= lower_bound, wv <= upper_bound)
            # Discard proposed parameters if lower bound is less than a certain value or upper bound is bigger than a certain value
            if not any(msk):
                # Discard the proposed parameters
                continue
            # Compute the likelihood ratio
            acceptance_ratio = self.likelihood(wv, flux, proposed_parameters) / self.likelihood(wv, flux, current_parameters)
    
            # Accept or reject the proposal
            if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                current_parameters = proposed_parameters
                if i >= self.burn_in and i % self.thinning == 0:
                    samples.append(current_parameters)  # Only append accepted proposals
        return samples

    
    # Posterior Distribution
    def posterior_analysis(self, wv, flux, initial_param, num_iterations):
        """
        Perform posterior analysis using the Metropolis-Hastings MCMC algorithm.
        
        Parameters:
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of observed flux values corresponding to the wavelengths.
        - initial_param (float or array-like): The starting parameter value(s) for the MCMC algorithm.
        - num_iterations (int): The number of iterations to run the MCMC algorithm.
        
        Returns:
        - accepted_samples (list): A list of accepted parameter values sampled from the posterior distribution.
        """
        current_param = initial_param
        samples = []
        step_sizes = [0.1, 10, 1, 2]
        prior = self.combined_prior()
        for _ in range(num_iterations):
            proposed_params = self.proposal(wv, flux, current_param, 10000, step_sizes)
            for proposed_param in proposed_params:
                prior_proposed = prior(proposed_param)
                prior_current = prior(current_param)
                likelihood_proposed = self.likelihood(wv, flux, proposed_param)
                likelihood_current = self.likelihood(wv, flux, current_param)
                acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current)
                if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                    current_param = proposed_param
                samples.append(current_param)  # Append only accepted proposal
        print("Posterior analysis is complete.")
        return samples
    
    
    # Set Initial Parameters
    def set_initial_param(self, name, date, wv, flux):
        """
        Set initial parameters for the MCMC algorithm.
        
        Parameters:
        - name (str): Name or identifier for the spectrum.
        - date (str): Observation date extracted from the file name.
        - wv (array-like): Array of observed wavelengths.
        - flux (array-like): Array of corresponding flux values.

        Returns:
        - list: Initial parameter values [amplitude, mu, sigma, delta_w].
        """
        c=299792.458
        # Randomly initialize parameters
        a = np.random.rand() * 3
        mu_low = self.relativistic_doppler(velocity = -20000)
        mu_high = self.relativistic_doppler(velocity = -5000)
        mu = np.random.uniform(mu_low, mu_high)
        sigma_low = 0
        _, _, lam_s = self.select_line()
        sigma_high = lam_s * (20000/c)
        sigma = np.random.uniform(sigma_low, sigma_high)
        wvc, fc = self.selected_range(name, date, wv, flux)
        delta_w = np.max(wvc) - np.min(wvc)
        init_param = [a, mu, sigma, delta_w]
        # Print initial parameter values
        print("Initial parameter values:")
        print("Amplitude:", a)
        print("Mu:", mu)
        print("Sigma:", sigma)
        print("Delta W:", delta_w)
        return init_param
    
    def plot_results(self, name, date, wv, flux, posterior_mean):
        # Plot spectrum with fitted Gaussian model
        msk, model_flux = self.model(wv, flux, posterior_mean)
        plt.plot(wv[msk], model_flux, label='Fitted Gaussian Model', color='blue')
        plt.legend()
        plt.draw()
        plt.show(block = False)
        plt.pause(0.001)  # Allow time for plot window to refresh
    
    def plot_trace(parameter_samples, parameter_names):
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
        plt.show()
    
    # Check goodness of the function
    def chi_square(observed_data, expected_data):
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
    initial_param = fitter.set_initial_param(name, date, wv, flux)
    samples = fitter.posterior_analysis(wv, flux, initial_param, num_iterations = 50)
    posterior_mean = np.mean(samples, axis=0)
    print("Posterior mean of parameters:", posterior_mean)
    fitter.plot_results(name, date, wv, flux, posterior_mean)
    parameter_names = ['a', 'mu', 'sigma', 'delta_w']
    fitter.plot_trace(samples, parameter_names)
    