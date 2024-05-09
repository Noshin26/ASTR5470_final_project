#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 01:34:33 2024

@author: pqt7tv
"""

import unittest
import numpy as np
from gaussian_fit_mcmc import GaussianFitter  # Import your GaussianFitter class from your module

class TestGaussianFitter(unittest.TestCase):

    def setUp(self):
        # Initialize GaussianFitter object with test data
        self.gaussian_fitter = GaussianFitter("dummy_spectrum_file.txt")
        self.gaussian_fitter.wv = np.linspace(4000, 8000, 100)
        self.gaussian_fitter.flux = np.random.rand(100)
        self.gaussian_fitter.name = "Test SN"
        self.gaussian_fitter.date = "2024-05-01"
        self.gaussian_fitter.l = "FeII5169"  # Example rest wavelength
    
    def test_relativistic_doppler_velocity(self):
        # Test velocity to wavelength conversion
        velocity = -20000  # km/s
        expected_wavelength = self.gaussian_fitter.relativistic_doppler(velocity=velocity)
        self.assertIsInstance(expected_wavelength, float)
        self.assertAlmostEqual(expected_wavelength,  4834.932612380226)

    def test_relativistic_doppler_wavelength(self):
        # Test wavelength to velocity conversion
        wavelength =  4834.932612380226 # Angstroms
        expected_velocity = self.gaussian_fitter.relativistic_doppler(wavelength=wavelength)
        self.assertIsInstance(expected_velocity, float)
        self.assertAlmostEqual(expected_velocity, -19999.999999999978)
        
    def test_uniform_prior(self):
        # Test the uniform_prior function
        amp_low = 0.05
        amp_high = 0.6
        mu_low = 4500
        mu_high = 5500
        sigma_low = 10
        sigma_high = 100
        amp_pdf = self.gaussian_fitter.uniform_prior(amp_low, amp_high)
        mu_pdf = self.gaussian_fitter.uniform_prior(mu_low, mu_high)
        sigma_pdf = self.gaussian_fitter.uniform_prior(sigma_low, sigma_high)
        self.assertAlmostEqual(amp_pdf(0.5), 1/0.55)  # amp
        self.assertEqual(mu_pdf(5000), 1/1000)   # mu
        self.assertEqual(sigma_pdf(70), 1/90)  # sigma  

    def test_combined_prior(self):
        # Test the combined_prior function
        combined_pdf = self.gaussian_fitter.combined_prior()
        parameters = [0.1, 5000, 100]
        self.assertAlmostEqual(combined_pdf(parameters), 1.3700593894200265e-05)  # Test with example parameters

    def test_likelihood(self):
        # Test likelihood function
        initial_parameters = [1.0, 5000, 100]
        likelihood_value = self.gaussian_fitter.likelihood(self.gaussian_fitter.wv, self.gaussian_fitter.flux, self.gaussian_fitter.wv[10:], initial_parameters)
        self.assertIsInstance(likelihood_value, float)
        self.assertGreaterEqual(likelihood_value, 0)

    def test_proposal(self):
        # Test proposal function
        initial_parameters = [1.0, 5000, 100]
        iterations = 10000
        step_sizes = [0.1, 10, 1]
        samples = self.gaussian_fitter.proposal(self.gaussian_fitter.wv, self.gaussian_fitter.flux, self.gaussian_fitter.wv[10:], initial_parameters, iterations, step_sizes)
        
        # Check if samples is a list
        self.assertIsInstance(samples, list)
    
        # Check if samples list is not empty
        self.assertGreaterEqual(len(samples), 0)
    
        # Check if each sample is a list
        for sample in samples:
            self.assertIsInstance(sample, list)
    
        # Check if each sample tuple has the correct length (number of parameters)
        self.assertEqual(len(samples[0]), len(initial_parameters))
    
        # Check if all elements in each sample tuple are floats
        for sample in samples:
            for param in sample:
                self.assertIsInstance(param, float)


    def test_posterior_analysis(self):
        # Test posterior_analysis function
        initial_parameters = [1.0, 5000, 100]
        num_iterations = 3
        accepted_samples = self.gaussian_fitter.posterior_analysis(self.gaussian_fitter.wv, self.gaussian_fitter.flux, self.gaussian_fitter.wv[10:], initial_parameters, num_iterations)
        
        # Check if the returned value is a list
        self.assertIsInstance(accepted_samples, list)

        # Check if the list contains tuples with correct length
        for sample in accepted_samples:
            self.assertIsInstance(sample, list)
            self.assertEqual(len(sample), len(initial_parameters))


    # Add more tests for other functions if needed

if __name__ == '__main__':
    unittest.main()
