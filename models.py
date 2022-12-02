import numpy as np
from metrics import HistologicalTransforms


class CPDBModel:
    '''
    a simple heuristic model which assigns a label on the image crop with 
    respect to threshold from the metric cpbd (the threshold is evaluated from the statistics)
    of the training data
    '''

    @staticmethod
    def cpbd(image: np.ndarray):
        pass
    
    def __init__(self, percentile: float = 51) -> None:
        '''
        percentile - percentile (must be between 0 and 100 inclusive) which will be
        the anchor in evaluating the threshold
        '''
        self.threshold = 0
        self.metric = HistologicalTransforms().cpbd
        self.percentile = percentile

    def fit(self, images):
        '''
        fits the model by calculating the metric and assigning the threshold
        to the initialized percentile
        '''
        metrics = np.array([self.metric(img) for img in images])
        self.threshold = np.percentile(metrics, self.percentile, out=0.5)
    
    def predict(self, img):
        res = self.cpbd(img)
        return 1 if res >= self.threshold else 0
