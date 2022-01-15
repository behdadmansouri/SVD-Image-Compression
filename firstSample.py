import matplotlib.pyplot as plt
import numpy as np


def reduce_rank(A):
    A = np.array(A)
    U = np.linalg.svd(A)[0]
    E = np.linalg.svd(A)[1]
    V_transpose = np.linalg.svd(A)[2]
    U_reduced = U[:,:t]
    E_reduced = np.diag(E[:t])
    V_transpose_reduced = V_transpose[:t, :]

    return np.matmul(U_reduced, np.matmul(E_reduced, V_transpose_reduced))


original_image = plt.imread('noisy.jpg')
t = 15
matrix = np.array(original_image).transpose((2, 1, 0))

for i in range(len(matrix)):
    matrix[i] = reduce_rank(matrix[i])

matrix = matrix.transpose((2, 1, 0))
plt.imshow(matrix)
plt.imsave('cleaned.jpeg', matrix)
plt.show()