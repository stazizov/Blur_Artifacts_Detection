from collections import Counter
#from metrics.intersection_over_union import iou
import numpy as np
from scipy.stats import kurtosis
from scipy import stats
from matplotlib import pyplot as plt
import cpbd



def calc_kurtosis(image_region: np.ndarray, fisher=True):
    '''
    image_region - square window from the original image
    fisher - boolean to indicate subtraction of 3 from kurtosis to give 0.0 in case of normal distribution
    '''
    kurt = kurtosis(image_region, fisher=fisher)

    return kurt

def power_spectrum(image_region: np.ndarray):
    n = image_region.shape[0]
    fourier_image = np.fft.fftn(image_region)
    amplitudes = np.square(np.abs(fourier_image)).flatten()
    
    freq = np.fft.fftfreq(n) * n
    freq_grid = np.meshgrid(freq, freq)
    norm = np.sqrt(freq_grid[0] ** 2 + freq_grid[1] ** 2).flatten()
    n_bins = np.arange(0.5, n // 2 + 1, 1.0)
    num_values = 0.5 * (n_bins[1:] + n_bins[:-1])

    bins, _, _ = stats.binned_statistic(norm, amplitudes, statistic="mean", bins=n_bins)
    bins *= np.pi * (n_bins[1:] ** 2 - n_bins[:-1] ** 2)

    return np.log(bins)

image_sample = np.random.rand(32, 32)
spectrum = power_spectrum(image_sample)

