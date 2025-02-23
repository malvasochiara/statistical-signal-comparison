# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, rayleigh, cramervonmises
from tqdm import tqdm

def perform_shapiro_test_for_normality_on_dataset(dataset, is_same_signal=True):
    """
    Function to analyze the p-values from the Shapiro-Wilk test applied to the input dataset to check for normality.

    The data is tested for normality using the Shapiro-Wilk test.

    Parameters:
    dataset (list of np.ndarray): List of numpy arrays containing the data (either "same" or "different" signal).
    is_same_signal (bool): If True, p-values less than 0.05 are considered "wrong" (indicating non-normality for the same signal).
                            If False, p-values greater than 0.05 are considered "wrong" (indicating non-normality for the different signal).

    Returns:
    tuple: Percentage of "wrong" p-values.
    """
    
    # List to store p-values
    p_values = []

    # Perform Shapiro-Wilk test on the dataset
    for i in tqdm(range(len(dataset))):
        stat, p = shapiro(dataset[i])
        p_values.append(p)

    # Plotting the histogram of p-values
    plt.figure(figsize=(5, 3))
    plot_color = 'purple' if is_same_signal else 'orange'  # Set color based on is_same_signal
    plt.hist(p_values, bins=30, density=True, alpha=0.5, label='Signal data', color=plot_color)
    plt.xlabel('p')
    plt.ylabel('Frequency')
    plt.title('P values')
    plt.legend()
    plt.show()

    # Count the "wrong" p-values based on the is_same_signal flag
    count_wrong = sum(p < 0.05 for p in p_values) if is_same_signal else sum(p > 0.05 for p in p_values)

    # Calculate and return the percentage of wrong p-values
    return (count_wrong / len(p_values)) * 100






def perform_ks_test_for_rayleigh_distribution(dataset, is_expected_distribution=True):
    """
    Function to perform the Kolmogorov-Smirnov test on the dataset to check if it follows a Rayleigh distribution.
    The function specifically checks if the data follows the Rayleigh distribution and plots the p-values distribution.

    Parameters:
    dataset (list of np.ndarray): List of numpy arrays containing the data to be tested.
    is_expected_distribution (bool): If True, p-values less than 0.05 are considered "wrong" (indicating a poor fit for the expected Rayleigh distribution).
                                      If False, p-values greater than 0.05 are considered "wrong" (indicating a poor fit for an unexpected distribution).

    Returns:
    list: List of p-values from the KS test.
    """
    
    p_values = []
    for data in tqdm(dataset):
        # Calculate the Rayleigh distribution parameter (sigma)
        sigma = (np.mean(data) * np.sqrt(2 / np.pi))
        D, p_value = kstest(data, 'rayleigh', args=(0, sigma))
        p_values.append(p_value)

    # Plotting the p-value distribution
    plt.figure(figsize=(5, 3))
    plot_color = 'purple' if is_expected_distribution else 'orange'
    plt.hist(p_values, bins=30, density=True, alpha=0.5, label='Rayleigh signal', color=plot_color)
    plt.xlabel('p')
    plt.ylabel('Frequency')
    plt.title('P values - Rayleigh signals')
    plt.legend()
    plt.show()

    # Count the "wrong" p-values based on the is_expected_distribution flag
    count_wrong = sum(p < 0.05 for p in p_values) if is_expected_distribution else sum(p > 0.05 for p in p_values)

    # Print the results
    print(f"Percentage of Rayleigh signals with p < 0.05: {(count_wrong / len(p_values)) * 100}%")
    print(f"Number of Rayleigh signals with p-value < 0.05: {count_wrong}")

    return (count_wrong / len(p_values)) * 100




##############################################################################
# Set of function that were used to asses wether the most extreme value 
# belonged to a Rayleigh distribution



def fit_rayleigh(data):
    """Fit Rayleigh distribution to the data and return parameters (mu, sigma)."""
    return rayleigh.fit(data)

def test_single_value_rayleigh_cdf(data):
    """Test if the maximum value of data follows a Rayleigh distribution using the CDF."""
    param = fit_rayleigh(data)  # Fit Rayleigh distribution
    x = np.max(data)  # Maximum value as the extreme value
    p_value = rayleigh.cdf(x - param[0], scale=param[1])  # CDF at the extreme value
    return p_value

def test_single_value_rayleigh_sf(data):
    """Test if the maximum value of data follows a Rayleigh distribution using the Survival Function."""
    param = fit_rayleigh(data)  # Fit Rayleigh distribution
    x = np.max(data)  # Maximum value as the extreme value
    p_value = rayleigh.sf(x - param[0], scale=param[1])  # Survival function at the extreme value
    return p_value

def cramer_von_mises_rayleigh(data):
    """Perform the Cramer-von-Mises test for Rayleigh distribution on the data."""
    param = fit_rayleigh(data)  # Fit Rayleigh distribution
    mu, sigma = param[0], param[1]
    value = np.max(data)  # Maximum value as the extreme value
    sample = rayleigh.rvs(loc=mu, scale=sigma, size=1000)  # Generate Rayleigh-distributed samples
    sample = np.append(sample, value)  # Append the maximum value to the sample
    result = cramervonmises(sample, 'rayleigh', args=(mu, sigma))  # Perform the test
    return result.pvalue

def plot_pvalue_distribution_and_count_wrong(data_list, test_method, is_rayleigh_expected=True):
    """
    Plot the distribution of p-values and count the percentage of wrong p-values.
    
    Parameters:
    data_list (list of np.ndarray): List of datasets to be tested.
    test_method (function): The method for testing the extreme value (e.g., CDF, SF, etc.).
    is_rayleigh_expected (bool): If True, we expect the p-value to be < 0.05 (Rayleigh distribution).
                                  If False, we expect p-value > 0.05.
    """
    p_values = []
    
    # Compute p-values for each data set
    for data in tqdm(data_list):
        p_value = test_method(data)
        p_values.append(p_value)
    
    # Plot p-value distribution
    plt.figure(figsize=(5, 3))
    color = 'purple' if is_rayleigh_expected else 'orange'
    plt.hist(p_values, bins=30, density=True, alpha=0.5, label='Tested data', color=color)
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('P-value Distribution')
    plt.legend()
    plt.show()
    
    # Count the percentage of wrong p-values
    if is_rayleigh_expected:
        wrong_count = sum(p < 0.05 for p in p_values)
    else:
        wrong_count = sum(p > 0.05 for p in p_values)
    
    # Print the percentage of wrong p-values
    print(f"Percentage of wrong p-values: {(wrong_count / len(p_values)) * 100:.2f}%")
    print(f"Number of wrong p-values: {wrong_count} out of {len(p_values)}")
    
##############################################################################
# see if the most extreme value follows a Gumbel distribution

def sf_gumbel(data):
    """Calculate the survival function for the maximum value of the data, assuming a Gumbel distribution."""
    x = np.max(data)  # Maximum value as the extreme value
    a = 1.283 / np.var(data)  # Gumbel shape parameter
    u = np.mean(data) - 0.45 * np.var(data)  # Gumbel location parameter
    z = (x - u) * a
    return 1 - np.exp(-np.exp(-z))

def plot_pvalue_distribution_and_count_wrong_gumbel(data_list, test_method, is_gumbel_expected=True):
    """
    Plot the distribution of p-values and count the percentage of wrong p-values for Gumbel distribution.
    
    Parameters:
    data_list (list of np.ndarray): List of datasets to be tested.
    test_method (function): The method for testing the extreme value (Gumbel survival function).
    is_gumbel_expected (bool): If True, we expect the p-value to be < 0.05 (Gumbel distribution).
                               If False, we expect p-value > 0.05.
    """
    p_values = []
    
    # Compute p-values for each data set
    for data in tqdm(data_list):
        p_value = test_method(data)
        p_values.append(p_value)
    
    # Plot p-value distribution
    plt.figure(figsize=(5, 3))
    color = 'purple' if is_gumbel_expected else 'orange'
    plt.hist(p_values, bins=20, density=True, alpha=0.5, label='Tested data', color=color)
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('P-value Distribution')
    plt.legend()
    plt.show()
    
    # Count the percentage of wrong p-values
    if is_gumbel_expected:
        wrong_count = sum(p < 0.05 for p in p_values)
    else:
        wrong_count = sum(p > 0.05 for p in p_values)
    
    # Print the percentage of wrong p-values
    print(f"Percentage of wrong p-values: {(wrong_count / len(p_values)) * 100:.2f}%")
    print(f"Number of wrong p-values: {wrong_count} out of {len(p_values)}")