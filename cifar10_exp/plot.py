import matplotlib as plt
import numpy as np


def main():
    SGD_l = np.load('experiment/SGD/loss.npy')
    SGD_a = np.load('experiment/SGD/accuracy.npy')
    SGD_momentum_l = np.load('experiment/SGD_momentum/loss.npy')
    SGD_momentum_a = np.load('experiment/SGD_momentum/accuracy.npy')
    Adagrad_l = np.load('experiment/Adagrad/loss.npy')
    Adagrad_a = np.load('experiment/Adagrad/accuracy.npy')
    RMSprop_l = np.load('experiment/RMSprop/loss.npy')
    RMSprop_a = np.load('experiment/RMSprop/accuracy.npy')
    Adam_l = np.load('experiment/Adam/loss.npy')
    Adam_a = np.load('experiment/Adam/accuracy.npy')
    epoch = [x for x in range(100)]
    plt.plot(epoch, SGD_l, SGD_momentum_l, Adagrad_l, RMSprop_l, Adam_l)
    plt.plot(epoch, SGD_a, SGD_momentum_a, Adagrad_a, RMSprop_a, Adam_a)


if __name__ == "__main__":
    main()
