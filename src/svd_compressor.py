import pickle

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

"Datasets like CIFAR-10 are too large to be stored as simple text files. Instead, the creators pickled the data into binary batches"


image = Image.open("hamster.jpeg")

print(image.format, image.size, image.mode)
pix = np.array(image)
print(pix.shape)
print(pix)

red_matrix = pix[:, :, 0]
green_matrix = pix[:, :, 1]
blue_matrix = pix[:, :, 2]

max_rank = min(image.height, image.width)
print(f"Max rank {max_rank}")

def compress_matrix(matrix, k):
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        k = min(k, len(s))
        "rank k means only k eigenvectors"
        u_k = U[:, :k]
        vt_k = Vt[:k, :]
        "make it kxk instead od default 1xk"
        s_k = np.diag(s[:k])
        "Ak = sum of 1 to k ui*si*vit"
        A_compressed = np.dot(u_k, np.dot(s_k, vt_k))

        "finding relative error using formula we saw in the lecture6 = sum singular vlaues from k+1 to r OVER sum 1 to r where r is og rank"
        relative_error = (np.sum(s[k:]**2))/(np.sum(s**2))
        "reconstructing A_compressed matrix to a jpeg"
        #print(A_compressed)
        return A_compressed, relative_error

k_values = [max_rank, 5, 20, 30, 40, 60]
"The variable s returned by NumPy is a 1D array of singular values"
for k in k_values:
    red_compressed, red_relative_error = compress_matrix(red_matrix,k)
    green_compressed, green_relative_error = compress_matrix(green_matrix,k)
    blue_compressed, blue_relative_error = compress_matrix(blue_matrix,k)
    print(f"For k={k} relative red error is:  {red_relative_error}")

    img_k = np.stack([red_compressed, green_compressed, blue_compressed], axis=2)
    " ensures valid colour pixels 0->255"
    img_k = np.clip(img_k, 0, 255).astype(np.uint8)
    plt.figure()
    plt.title(f"k = {k}")
    plt.imshow(img_k)
    plt.axis("off")
    plt.show()





