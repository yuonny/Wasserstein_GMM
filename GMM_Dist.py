from sklearn.mixture import GaussianMixture
from scatterplot_3D import X
import numpy as np
import os 
os.chdir('/Users/davidkim/Documents/GitHub/Wasserstein_GMM')

from GMM_Visualization.src.main.GMMViz.GaussianMixtureModel import GMM
from GMM_Visualization.src.main.GMMViz.GmmPlot import GmmViz
from GMM_Visualization.src.main.GMMViz.DataGenerater import DataGenerater
#import plotly.io as pio

gmm = GaussianMixture(n_components = 3, covariance_type = 'diag')
gmm.fit(X) 
