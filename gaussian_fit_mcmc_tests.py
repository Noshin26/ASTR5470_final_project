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
        self.gaussian_fitter.l = 6564  # Example rest wavelength
        
    def test_uniform_prior(self):
        # Test the uniform_prior function
        amp_low = 0
        amp_high = 3
        mu_low = 4500
        mu_high = 5500
        sigma_low = 0
        sigma_high = 100
        amp_pdf = self.gaussian_fitter.uniform_prior(amp_low, amp_high)
        mu_pdf = self.gaussian_fitter.uniform_prior(mu_low, mu_high)
        sigma_pdf = self.gaussian_fitter.uniform_prior(sigma_low, sigma_high)
        self.assertEqual(amp_pdf(1), 1/3)  # amp
        self.assertEqual(mu_pdf(5000), 1/1000)   # mu
        self.assertEqual(sigma_pdf(70), 1/100)  # sigma  

    def test_gaussian_prior(self):
        # Test the gaussian_prior function
        mu = 0
        sigma = 100/3
        gaussian_pdf = self.gaussian_fitter.gaussian_prior(mu, sigma)
        self.assertAlmostEqual(gaussian_pdf(0), 0.01196826841)  # Value at mean
        self.assertAlmostEqual(gaussian_pdf(1), 0.0119628839)  # Value at one standard deviation away from mean

    def test_combined_prior(self):
        # Test the combined_prior function
        amp_low = 0
        amp_high = 3
        mu_low = 4500
        mu_high = 5500
        sigma_low = 0
        sigma_high = 100
        delta_w_mu = 0
        delta_w_sigma = 100/3
        combined_pdf = self.gaussian_fitter.combined_prior(mu_low, mu_high, sigma_low, sigma_high, amp_low, amp_high, delta_w_mu, delta_w_sigma)
        parameters = [1, 5000, 70, 1]
        self.assertAlmostEqual(combined_pdf(parameters), (1/3)*(1/1000)*(1/100)*0.0119628839)  # Test with example parameters

    def test_likelihood(self):
        # Test likelihood function
        initial_parameters = [1.0, 5000, 100, 1000]
        likelihood_value = self.gaussian_fitter.likelihood(self.gaussian_fitter.wv, self.gaussian_fitter.flux, initial_parameters)
        self.assertIsInstance(likelihood_value, float)
        self.assertGreaterEqual(likelihood_value, 0)

    '''def test_proposal(self):
        # Test proposal function
        initial_parameters = [1.0, 5000, 100, 1000]
        iterations = 100
        step_sizes = [0.1, 10, 1, 1]
        samples = self.gaussian_fitter.proposal(initial_parameters, iterations, step_sizes)
        self.assertIsInstance(samples, list)
        self.assertGreaterEqual(len(samples), 0)

    def test_posterior_analysis(self):
        # Test posterior_analysis function
        initial_parameters = [1.0, 5000, 100, 1000]
        num_iterations = 30
        samples = self.gaussian_fitter.posterior_analysis(initial_parameters, num_iterations, 
                                                           self.gaussian_fitter.likelihood, 
                                                           self.gaussian_fitter.combined_prior, 
                                                           self.gaussian_fitter.proposal)
        self.assertIsInstance(samples, list)
        self.assertGreaterEqual(len(samples), 0)'''

    # Add more tests for other functions if needed

if __name__ == '__main__':
    unittest.main()
