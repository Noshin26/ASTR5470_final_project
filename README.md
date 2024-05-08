# ASTR5470_final_project
# Calculating Absorption Velocity of Different Lines in Broad-line Type Ic (Ic-bl) Supernovae (SNe) by Fitting a Gaussian Model to the Absorption Feature using Markov Chain Monte Carlo Simulation. 
This Python script enables users to analyze spectra of SNe Ic-bl by fitting Gaussian models to selected wavelength ranges. Users can interactively select a wavelength range on a plot of the spectrum and input parameters for galaxy redshift and spectrum line of interest.
## Usage
1. Clone the repository or download the `gaussian_fit_mcmc.py` file.
2. Navigate to the directory containing `gaussian_fit_mcmc.py`.
3. To run the script, use the following command in your terminal:
'python gaussian_fit_mcmc.py spectrum_file.ascii -z <galaxy_redshift> -l <absorption_line>'
Replace spectrum_file.ascii with the path to your spectrum data file. Additionally, include the following optional arguments:
-z: Specifies the galaxy redshift.
-l: Specifies the absorption line.
# Unittest
A testing script gaussian_fit_mcmc_tests.py is provided to test key functions of the main script. You can run the tests using the following command: 'python gaussian_fit_mcmc_tests.py'
Ensure that you have installed the unittest module to execute the tests.
# Instructions
Upon running the command, an interactive window will open, displaying the supernova spectrum.
Click on the two points on te spectra, the first power is for the lower bound of the selected range and the second point is for the upper bound of the selected range. 
# Output
# Output Printed to Console:
1. Parameters: Amplitude, mean, and sigma of the fitted Gaussian model.
Mean Velocity and Standard Deviation: Mean velocity and one standard deviation of the fitted model.
2. Chi-Square Value: Value of the chi-square test.
# Generated Output Files:
1. Plot of Observed Spectrum with Fitted Gaussian Model:
Description: A plot showing the observed spectrum with the fitted Gaussian model overlaid.
File Type: Image file (e.g., PNG).
2. Fitted Parameters File:
Description: A file containing the fitted parameters, velocity, one standard deviation, and chi-square value.
File Type: Text file.
3. Corner Plot:
Description: A corner plot displaying the correlations between different parameters.
File Type: Image file (e.g., PNG).
4. Parameter Trace Plot:
Description: A plot showing the trace of each parameter during the Markov Chain Monte Carlo simulation.
File Type: Image file (e.g., PNG).

# References
This work is adapted (with modifcation) from Modjaz et al. 2016.
