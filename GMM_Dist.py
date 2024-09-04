from sklearn.mixture import GaussianMixture
from scatterplot_3D import X
import numpy as np
import os 
os.chdir('/Users/davidkim/Documents/GitHub/Wasserstein_GMM')

from GMM_Visualization.src.main.GMMViz.GaussianMixtureModel import GMM
from GMM_Visualization.src.main.GMMViz.GmmPlot import GmmViz
from GMM_Visualization.src.main.GMMViz.DataGenerater import DataGenerater
#import plotly.io as pio
#fitting the gaussian to the image

#GMM is different from GausianMixture and it is specific to this use case 
print("hi")
gmm = GMM(n_clusters = 3)
gmm.fit(X) 


#see the visualization
#GmmViz is the class and the V3F is an object with the parameters gmm and utiPlotly
V3F = GmmViz(gmm, utiPlotly=False)
'''
V3F.plot(fig_title= "GMM-3D", 
         path_prefix="GMM_Images", # image will be stored in the `path_prefix` directory.
         show_plot = False, #  tells whether to show the figure through the editor or not. Default is `False`.
         save_plot = True, # export the figures. Default is True
         max_iter = 33)
 '''

GmmViz.generateGIF(image_path = "GMM_Images", # directory of the images showing each iteraction
                   output_path_filename = "GMM_Images/gif/GMM-3D-Parms.gif", 
                   fps = 2) # Adjust the timing of each frame in the GIF file

