import matplotlib.pyplot as plt
import numpy as np

# In this section the image is read  and saved as a matrix
filename = "1.bmp"
image = plt.imread(filename)
image_matrix = np.array(image)
# In this section the SVD of the image_matrix is calculated
k = 10000
image_matrix_t = image_matrix.transpose()  # The transpose of the image matrix is calculated so the elements of the array represents the columns instead of the rows
reduced_image_matrix_t = image_matrix_t
for i in range(len(image_matrix_t)): # basically loops for rgb (3 times)
    matrix_svd = np.linalg.svd(image_matrix_t[i])
    U = matrix_svd[0]
    S = matrix_svd[1]
    V_t = matrix_svd[2]
    # In this section the S matrix is reduced and saved as R
    m = len(U[0])
    n = len(V_t)
    R = np.zeros((m, n))
    for j in range(min(m, n)):
        if j <= k:
            R[j][j] = S[j]
    reduced_image_matrix_t[i] = np.matmul(U, np.matmul(R, V_t))
# In the last section the new matrix is calculated and saved as the noise reduced image
reduced_image_matrix = reduced_image_matrix_t.transpose()
plt.imsave("reduced_image" + str(k) + "_" + str.replace(filename, "bmp", "jpg"), reduced_image_matrix)