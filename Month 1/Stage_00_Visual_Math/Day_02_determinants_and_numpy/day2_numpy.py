import numpy as np
import time
v1 = np.array([2,4])

v2 = np.array([4,6])


addition = v1 +v2

scalar_product = 3*v1

dp =np.dot(v1 ,v2)

print(
    f"""addition result: {addition},
        scalar_product result: {scalar_product},
        dot_product result: {dp}"""
)


## MATRIX
m1 = np.matrix([[1,2,3],
      [2,3,4],
      [4,5,6]])

m2 = np.matrix([[2,3,4],
      [4,3,2],
      [2,5,7]])

print(np.matmul(m1,m2))


def dot_product(v1, v2):
    result = 0

    for i in range(len(v1)):
        result = result + int((v1[i]*v2[i]))

    return result

def matrix_matrix_mult(m1, m2):
    result = []

    for i in range(len(m1)):
        m1_row = m1[i]

        new =[]

        for j in range(len(m2[0])):
            m2_col =[]
            
            for k in range(len(m2)):
                m2_col.append(m2[k][j])
            

            dp =dot_product(m1_row,m2_col)

            new.append(dp)

        result.append(new)


    return result

# Creates two 200x200 matrices filled with random numbers
big_matrix_1 = np.random.rand(200, 200)
big_matrix_2 = np.random.rand(200, 200)



print(np.matmul(big_matrix_1.tolist(),big_matrix_2.tolist()))
