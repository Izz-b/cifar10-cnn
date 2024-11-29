import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
    img = img / 2 + 0.5 # Denormalize images
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Transpose so that RGB channels are last
    plt.show()

def save_model(model, path):
    # saving the model's state dictionary as opposed to the whole model is better since it's more lightweight
    torch.save(model.state_dict(), path)
