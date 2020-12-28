import math

def get_Gaussian_matrix(n, sig):
    # n must be odd and sig must larger than 1
    if (n / 2 == 0):
        print('Kernel Size must be an odd number')
        return
    if (sig <= 0):
        print('Sigma must be larger than 0')
        return
    center = (n-1) / 2
    print('Center:', center)
    gaussian_matrix = []
    sum = 0
    for i in range(n):
        gaussian_matrix.append([])
        for j in range(n):
            gaussian_matrix[i].append(1 / (2*math.pi*sig*sig) * math.exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sig*sig)))
            sum += gaussian_matrix[i][j]
    for i in range(n):
        for j in range(n):
            gaussian_matrix[i][j] /= sum
            print('i = ', i, ' j = ', j,' value = ',gaussian_matrix[i][j])
    print(sum)
