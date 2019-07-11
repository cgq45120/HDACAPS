import numpy as np

if __name__ == "__main__":
    a = np.arange(1,21,1).reshape((2,2,5))
    print(a.shape)
    print(a)
    b = np.ones((2,2,1))
    print(b)
    print(b.shape)
    c = [a,b]
    # c = np.hstack((a,b))
    print(c)