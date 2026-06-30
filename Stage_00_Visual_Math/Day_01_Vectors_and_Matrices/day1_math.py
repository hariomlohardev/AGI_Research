def add_vectors(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(v1[i]+v2[i])

    return result

print(add_vectors([4,-2],[1,5]))


def scale_vector(scalar, v):
    result = []
    for i in range(len(v)):
        result.append(scalar*int(v[i]))

    return result

print(scale_vector(3,[4,-2]))


def dot_product(v1, v2):
    result = 0

    for i in range(len(v1)):
        result = result + int((v1[i]*v2[i]))

    return result

print(dot_product([3,4],[2,-1]))



def matrix_vector_mult(matrix, vector):
    result = []

    for i in range(len(matrix)):
        m= matrix[i]
        print('sddddddddddddd',m)
        total =0
        for j in range(len(m)):
            total= total+int(vector[j]*m[j])

            print('dsssssssssss',total)

        result.append(total)


    return result


print(matrix_vector_mult([[2,0],[0,3]],[2,1]))