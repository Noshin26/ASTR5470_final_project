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
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

class GaussianFitter:
    def __init__(self, spectrum_file, z=0, n_steps=10000, burn_in=1000, thinning=1):
        self.spectrum_file = spectrum_file
        self.z = z
        self.n_steps = n_steps  # Total number of MCMC steps
        self.burn_in = burn_in  # Number of burn-in steps
        self.thinning = thinning  # Thinning parameter for MCMC
    

# Data Preparation
def read_file(file_path, z):
    """
    Reads an ASCII file containing two columns: observed wavelength and corresponding flux.
    
    Parameters:
    - file_path (str): The path to the ASCII file to be read.
    - z (float): Galaxy redshift
    
    Returns:
    - name (str): The name of the supernova extracted from the file name.
    - date (str): The observation date extracted from the file name.
    - wavelengths (list of float): A list of observed wavelengths.
    - fluxes (list of float): A list of corresponding flux values.
    """
    try:
        # Extract name and date from the file path
        file_name = os.path.basename(file_path)
        name = file_name.split("_")[0]
        date = file_name.split("_")[1]
        
        # Read the ASCII file
        spec = ascii.read(file_path)
        
        # Normalize the flux by dividing by the mean flux
        flux = spec['col2'] / np.mean(spec['col2'])
        
        # Apply redshift correction to wavelengths
        wv = spec['col1'] / (1 + z)
        
        print(f"File is accepted '{file_path}'")
        return name, date, wv, flux
    
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return None, None, None, None


    def log_likelihood(self, parameters):
        # Compute the log-likelihood of the data given the parameters
        pass
    
    def log_prior(self, parameters):
        # Compute the log-prior of the parameters
        pass
    
    def metropolis_hastings(self):
        # Initialize parameters and sample storage
        current_params = [1, 0, 1]  # Amplitude, mean, stddev
        samples = np.zeros((self.n_steps, 3))
        
        # Perform Metropolis-Hastings MCMC
        for i in range(self.n_steps):
            proposed_params = [np.random.normal(current_params[j], 0.1) for j in range(3)]
            log_alpha = self.log_likelihood(proposed_params) + self.log_prior(proposed_params) - \
                        self.log_likelihood(current_params) - self.log_prior(current_params)
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                current_params = proposed_params
            samples[i] = current_params
            
        return samples[self.burn_in::self.thinning]  # Return samples after burn-in with thinning
    
    def fit_gaussian(self):
        # Run Metropolis-Hastings MCMC to fit Gaussian model
        samples = self.metropolis_hastings()
        return np.mean(samples, axis=0)  # Return posterior mean
    
    def plot_results(self):
        # Plot spectrum with fitted Gaussian model
        pass

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit Gaussian model to supernova spectrum.")
    parser.add_argument("spectrum_file", type=str, help="Path to spectrum file")
    parser.add_argument("-z", "--redshift", type=float, default=0, help="Galaxy redshift")
    parser.add_argument("-l", "--line", type=float, default=None, help="Wavelength of the absorption line")
    args = parser.parse_args()
    
    #fitter = GaussianFitter(args.spectrum_file, args.redshift)
    #posterior_mean = fitter.fit_gaussian()
    #print("Posterior mean of parameters (amplitude, mean, stddev):", posterior_mean)
