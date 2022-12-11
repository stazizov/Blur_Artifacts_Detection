import numpy as np
from scipy.stats import kurtosis
from scipy import stats
from .compute import compute
from matplotlib import pyplot as plt
from pywt import dwt2
import cv2
from skimage.measure import blur_effect


class HistologicalTransforms:
    def __init__(self, kurtosis_fisher=True, nrp_filter_size=11) -> None:
        self.fisher = kurtosis_fisher
        self.nrp_filter_size = nrp_filter_size

    def haar_transform(self, image_crop: np.ndarray) -> np.ndarray:
        """
        return summed wavelet transforms (shape is [N // 2, N // 2], where NxN - Image region shape)
        """
        if image_crop.ndim == 3:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        tr1, (tr2, tr3, tr4) = dwt2(image_crop, 'haar')
        concatenated_transform = np.array([tr1, tr2, tr3, tr4])
        return concatenated_transform.mean(axis=0)

    def cpbd(self, image_region: np.ndarray):
        '''
        
        :param image_region: crop of the given image
        :return: probabilistic of blurness of the sample appropriate to the paper (scalar from 0 to 1)
        
        paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4983426/
        '''
        return compute(image_region)

    def calc_kurtosis(self, image_region: np.ndarray):
        '''
        return kurtosis - an array of measures of tailedness of a distribution along an axis

        image_region - square window from the original image
        fisher - boolean to indicate subtraction of 3 from kurtosis to give 0.0 in case of normal distribution
        '''
        if image_region.ndim == 3:
            image_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        kurt = kurtosis(image_region, fisher=self.fisher)

        return kurt

    def power_spectrum(self, image_region: np.ndarray):
        """
        :param image_region:
        :return:
        """
        if image_region.ndim == 3:
            image_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        n = image_region.shape[0]
        fourier_image = np.fft.fftn(image_region)
        amplitudes = (np.abs(fourier_image) ** 2).flatten()
        
        freq = np.fft.fftfreq(n) * n
        freq_grid = np.meshgrid(freq, freq)
        norm = np.sqrt(freq_grid[0] ** 2 + freq_grid[1] ** 2).flatten()
        n_bins = np.arange(0.5, n // 2 + 1, 1.0)
        num_values = 0.5 * (n_bins[1:] + n_bins[:-1])

        bins, _, _ = stats.binned_statistic(norm, amplitudes, statistic="mean", bins=n_bins)
        bins *= np.pi * (n_bins[1:] ** 2 - n_bins[:-1] ** 2)

        return np.log(bins)

    def nrp(self, image_region: np.ndarray):
        '''
            No-Reference Perceptual Blur Metric
            returns strength of blur (0 for no blur, 1 for maximal blur)

            paper: https://hal.archives-ouvertes.fr/hal-00232709
        '''
        assert image_region.shape[0] > self.nrp_filter_size

        return blur_effect(image_region, h_size=self.nrp_filter_size)
    
    def __call__(self, image_region: np.ndarray):
        haar = self.haar_transform(image_region) # 2d
        cpdb = self.cpbd(image_region) # scalarr
        kurtosis = self.calc_kurtosis(image_region) # scalar
        spectrum = self.power_spectrum(image_region) # 1d
        nrp = self.nrp(image_region) # scalar

        return {
            "haar": haar,
            "cpdb": cpdb,
            "kurtosis": kurtosis,
            "power_spectrum": spectrum,
            "nrp": nrp
        }
        


class LaplacianBlurDetector:
    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def estimate_blur(self, image: np.array):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return blur_map, score, bool(score < self.threshold)
    
    def fix_image_size(self, image: np.array, expected_pixels: float = 2E6):
        ratio = np.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
        return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

    def pretty_blur_map(self, blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):

        abs_image = np.abs(blur_map).astype(np.float32)
        abs_image[abs_image < min_abs] = min_abs

        abs_image = np.log(abs_image)
        cv2.blur(abs_image, (sigma, sigma))
        return cv2.medianBlur(abs_image, sigma)

    def __call__(self, image: np.ndarray):
        """
        :param image: any image
        :return: blurry image or not
        """
        resized_image = self.fix_image_size(image)
        is_blurry = self.estimate_blur(resized_image)[-1]
        return is_blurry