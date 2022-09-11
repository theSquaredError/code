'''Graph world with grids'''
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X = np.linspace(-10,10,num=21, dtype=np.int32)
    Y = np.linspace(-10,10, num=21,dtype=np.int32)
    xx, yy = np.meshgrid(X,Y)
    # print("X coordinates:\n{}\n".format(xx))
    # print("Y coordinates:\n{}".format(yy))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xx, yy, ls="None", marker=".")
    # plt.show()