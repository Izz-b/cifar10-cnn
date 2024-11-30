import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # Denormalize images
    npimg = img.numpy()
    plt.ioff()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Transpose so that RGB channels are last
    plt.show()



