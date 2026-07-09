import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Load image and convert to grayscale (2D array)
img = Image.open(r'Stage_00_Visual_Math\Day_10_SVD\data\junel-mujar-iBktcTF8kso-unsplash.jpg').convert('L')
img_array = np.array(img, dtype=np.float64)

# 2. Compute SVD safely on a 2D matrix
U, S, V_transpose = np.linalg.svd(img_array)

def compress_image(U, S, Vt, k):
    U = U[:,:k]
    S = np.diag(S[:k])
    Vt = Vt[:k,:]

    return U @ S @ Vt

compresed_image = compress_image(U,S,V_transpose,100)
plt.imshow(compresed_image , cmap='gray')
plt.show()