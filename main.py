import matplotlib.pyplot as plt
import numpy as np

def get_image(path):
    image = plt.imread(path)
    return np.array(image)

def svd_calculator(input_matrix, matrix_degree):
    input_matrix = input_matrix.transpose()
    output_matrix = rank_reducer(input_matrix, matrix_degree)
    output_matrix = output_matrix.transpose()
    return output_matrix


def rank_reducer(input_matrix, matrix_degree):
    final_matrix = input_matrix
    for i in range(3):
        svd = np.linalg.svd(input_matrix[i])
        U = svd[0]
        S = svd[1]
        V_transpose = svd[2]
        output_matrix = np.zeros((len(U[0]), len(V_transpose)))
        for j in range(min(len(U[0]), len(V_transpose))):
            if j <= matrix_degree:
                output_matrix[j][j] = S[j]
        mulres = np.matmul(output_matrix, V_transpose)
        mulres2 = np.matmul(U, mulres)
        final_matrix[i] = mulres2
    return final_matrix


if __name__ == '__main__':
    image_path = input("input file path to image")
    matrix = get_image(image_path)
    degree = input("input degree")
    result_matrix = svd_calculator(matrix, degree)
    plt.imshow(result_matrix)
    plt.show()
    plt.imsave("reduced_image{}_{}".format(str(degree), str.replace(image_path, "bmp", "jpg")), result_matrix)
