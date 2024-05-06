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
    def __init__(self, spectrum_file, z=0, l='', n_steps=10000, burn_in=1000, thinning=1):
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
        plt.show()
        return wvc, fc
    
    # Model Definition
    def model(self, a, mu, sigma, delta_w):
        """
        Define a Gaussian model for an absorption feature in spectral data.

        Parameters:
        - a (float): Amplitude of the absorption feature.
        - mu (float): Central wavelength of the absorption feature.
        - sigma (float): Standard deviation, determining the spread or width of the absorption.
        - delta_w (float): Wavelength-range parameter.

        Returns:
        - msk (array-like): indices of the selected wavelength range
        - absorption_model (array-like): The Gaussian absorption model evaluated at the given wavelengths.
        """
        # Apply the wavelength range parameter
        lower_bound = mu - (delta_w * 0.5)
        upper_bound = mu + (delta_w * 0.5)
        msk = np.logical_and(self.wv >= lower_bound, self.wv <= upper_bound)
        lam = self.wv[msk]
        
        # Calculate the Gaussian absorption model
        absorption_model = -1 * a * np.exp(-0.5 * ((lam - mu) / sigma) ** 2) + (np.min(self.fc) + a)

        return msk, absorption_model
    
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
    
    def combined_prior(self, mu_low, mu_high, sigma_low, sigma_high, amp_low, amp_high, delta_w_mu, delta_w_sigma):
        """
        Combined prior distribution function for parameters: mu, sigma, amplitude, and delta_w.

        Parameters:
        - mu_low: Lower bound of the uniform distribution for mu.
        - mu_high: Upper bound of the uniform distribution for mu.
        - sigma_low: Lower bound of the uniform distribution for sigma.
        - sigma_high: Upper bound of the uniform distribution for sigma.
        - amp_low: Lower bound of the uniform distribution for amplitude.
        - amp_high: Upper bound of the uniform distribution for amplitude.
        - delta_w_mu: Float, the mean of the Gaussian prior for delta_w.
        - delta_w_sigma: Float, the standard deviation of the Gaussian prior for delta_w.

        Returns:
        - prior: Function representing the combined prior probability density.
        """
        mu_prior = self.uniform_prior(mu_low, mu_high)
        sigma_prior = self.uniform_prior(sigma_low, sigma_high)
        amp_prior = self.uniform_prior(amp_low, amp_high)
        delta_w_prior = self.gaussian_prior(delta_w_mu, delta_w_sigma)

        def prior(parameters):
            amp, mu, sigma, delta_w = parameters
            return mu_prior(mu) * sigma_prior(sigma) * amp_prior(amp) * delta_w_prior(delta_w)

        return prior
    
    # Likelihood Function
    def likelihood(self, parameters):
        """
        Calculate the likelihood of the data given the parameters.
        
        Parameters:
        - parameters: array-like, containing [amplitude, central_wavelength, sigma, delta_w].
        
        Returns:
        - likelihood: float, the likelihood of the data given the parameters.
        """
        a, mu, sigma, delta_w = parameters
        
        msk, absorption_model = self.model(a, mu, sigma, delta_w)
        observed_data = self.flux[msk]
        # Calculate the likelihood as the product of probabilities at each wavelength
        epsilon = 1e-10  # Small epsilon to prevent division by zero
        likelihood = np.prod(np.exp(-0.5 * ((observed_data - absorption_model) / (sigma + epsilon)) ** 2)) / np.sqrt(2 * np.pi * (sigma**2 + epsilon))
        return likelihood
    
    # MCMC Sampling
    def proposal(self, initial_parameters, iterations, step_sizes):
        """Metropolis-Hastings algorithm to sample from the posterior distribution of parameters.
        
        Parameters:
        - initial_parameters: list, initial guesses for the parameters [amplitude, mu, sigma, delta_w].
        - iterations: int, number of iterations to run the algorithm.
        - step_sizes: list, step sizes for each parameter [amp_step, mu_step, sigma_step, delta_w_step].
        
        Returns:
        - samples: list of tuples, sampled parameter 
        values [amplitude, mu, sigma, delta_w] from the posterior distribution.
        """
        current_parameters = initial_parameters
        samples = []

        for _ in range(iterations):
            # Propose new parameters
            proposed_parameters = [np.random.normal(current_parameters[i], step_sizes[i]) for i in range(4)]
            proposed_parameters[3] = max(0, proposed_parameters[3])  # Enforce non-negativity
            # Compute the likelihood ratio
            acceptance_ratio = self.likelihood(proposed_parameters) / self.likelihood(current_parameters)
            
            # Accept or reject the proposal
            if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                current_parameters = proposed_parameters
                samples.append(current_parameters)  # Only append accepted proposals

        return samples
    
    # Posterior Distribution
    def posterior_analysis(self, initial_param, num_iterations, likelihood, prior, proposal):
        """
        Perform posterior analysis using the Metropolis-Hastings MCMC algorithm.
        
        Parameters:
        - initial_param (float or array-like): The starting parameter value(s) for the MCMC algorithm.
        - num_iterations (int): The number of iterations to run the MCMC algorithm.
        - likelihood (function): A function that computes the likelihood of the data given a parameter.
        - prior (function): A function that computes the prior probability of a given parameter.
        - proposal (function): A function that suggests a new parameter value based on the current one.
        
        Returns:
        - accepted_samples (list): A list of accepted parameter values sampled from the posterior distribution.
        """
        current_param = initial_param
        samples = []
        for _ in range(num_iterations):
            step_sizes = [0.1, 10, 1, 1]
            proposed_params = self.proposal(current_param, 1000, step_sizes)
            for proposed_param in proposed_params:
                prior_proposed = prior(proposed_param)
                prior_current = prior(current_param)
                likelihood_proposed = likelihood(proposed_param)
                likelihood_current = likelihood(current_param)
                acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current)
                if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                    current_param = proposed_param
                samples.append(current_param)  # Append only accepted proposal
        return samples
    
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
        if velocity is not None:
            # Calculate wavelength from velocity
            lam_o = self.l * np.sqrt((1 + velocity / c) / (1 - velocity / c))
            return lam_o
        elif wavelength is not None:
            # Calculate velocity from wavelength
            velocity = c * ((wavelength / self.l)**2 - 1) / ((wavelength / self.l)**2 + 1)
            return velocity
        else:
            raise ValueError("Please provide either velocity or wavelength.")
    
    # Set Initial Parameters
    def set_initial_param(self):
        """
        Set initial parameters for the MCMC algorithm.

        Parameters:
        - wvc (array-like): Array of wavelengths for the selected region.

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
        wvc = self.selected_range()
        delta_w = np.max(wvc) - np.min(wvc)
        init_param = [a, mu, sigma, delta_w]
        return init_param
    
    
    def fit_gaussian(self):
        # Initialize initial parameters
        initial_param = self.set_initial_param()
    
        # Run Metropolis-Hastings MCMC to fit Gaussian model
        samples = self.posterior_analysis(initial_param, num_iterations=5000, likelihood=self.likelihood,
                                           prior=self.combined_prior, proposal=self.proposal)
    
        # Return posterior mean
        return np.mean(samples, axis=0)
    
    def plot_results(self):
        # Plot spectrum with fitted Gaussian model
        pass

# Example usage:
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
    