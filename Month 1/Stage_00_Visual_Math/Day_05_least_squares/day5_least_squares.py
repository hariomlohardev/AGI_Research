import numpy as np
import matplotlib.pyplot as plt

# House sizes (X) and intercept column
X = np.array([[1, 1],
              [1, 2],
              [1, 3]]) # First column is 1s, second column is house size

y = np.array([2, 3, 5]) # Prices



theta = np.linalg.inv(X.T @ X) @ X.T @ y

c , m = theta[0],theta[1]

line_y = m * X[:, 1] + c

eror_rate =0
for i in range(len(y)):
    differnce =y[i]-line_y[i]
    sq_diff =differnce**2

    eror_rate += sq_diff
    print(eror_rate)

print(eror_rate)


plt.scatter(X[:, 1], y, color='red', label='Real Data') # Plot data points
plt.plot(X[:, 1], line_y, color='blue', label='Best Fit Line') # Plot your ML line
plt.xlabel('House Size')
plt.ylabel('Price')
plt.legend()
plt.show()


# x = [1,2,3]
# y = [2,3,5]

# line_y = [11/6,10/3,29/6]



# plt.scatter(x,y,color='red',label='data')
# plt.plot(x,line_y,color='pink',label='best fit ')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()