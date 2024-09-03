import numpy as np
import matplotlib.pyplot as plt
#from sklearn.mixture import GaussianMixture
#import matplotlib.image
#from mpl_toolkits.mplot3d import Axes3D

# reads the image 
# png so u dont have to normalize 
u = plt.imread('/Users/davidkim/Desktop/pikachu.png')


#a tuple package in python that automatticaly assigns the number of rows, columns, and channels in the image
nru,ncu,nch = u.shape
print(u.shape)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

axes[0].imshow(u)
axes[0].set_title('First image')
fig.tight_layout()


# reshape only works when the size of the reshape is equal to inital shape of the image
#u[:, :, :3] splits channels 
# u need to reshape from 3d to 2D to be able to access data 
X = u[:, :, :3].reshape((nru * ncu, 3))
nb   = 3000
r    = np.random.RandomState(42)
idX  = r.randint(X.shape[0], size=(nb,))
Xs   = X[idX, :]

fig  = plt.figure(2, figsize=(20, 10))
axis = fig.add_subplot(1, 2, 1, projection="3d")
axis.scatter(Xs[:, 0], Xs[:,1],Xs[:, 2], c=Xs,s=100)
axis.set_xlabel("Red"), axis.set_ylabel("Green"), axis.set_zlabel("Blue")
plt.show()


