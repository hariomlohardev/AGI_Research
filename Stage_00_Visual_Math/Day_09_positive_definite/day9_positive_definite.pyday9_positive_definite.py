import numpy as np 

A = np.array([[2,1],
              [1,2]])


eigne_value , Q = np.linalg.eig(A)

Lambda = np.diag(eigne_value)

result = Q @ Lambda @ Q.T

print(result)


"""" IS POSITIVE DEFINITE"""



def is_positive_definite(matrix):

    eigne_value , Q = np.linalg.eig(matrix)
    print(eigne_value)

    result = True
    
    for i in range(len(eigne_value)):
        if eigne_value[i] <= 0:
            result = False
            break

    return result
    
M2 = np.array([[1, 2], [2, 1]])


print(is_positive_definite(A))
print(is_positive_definite(M2))
