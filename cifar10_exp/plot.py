import matplotlib.pyplot as plt
import numpy as np


def main(plot_type='accuracy'):
    SGD_l = np.load('experiment/SGD/loss.npy')
    SGD_a = np.load('experiment/SGD/accuracy.npy')
    #SGD_momentum_l = np.load('experiment/SGD_momentum/loss.npy')
    #SGD_momentum_a = np.load('experiment/SGD_momentum/accuracy.npy')
    Adagrad_l = np.load('experiment/Adagrad/loss.npy')
    Adagrad_a = np.load('experiment/Adagrad/accuracy.npy')
    RMSprop_l = np.load('experiment/RMSprop/loss.npy')
    RMSprop_a = np.load('experiment/RMSprop/accuracy.npy')
    Adam_l = np.load('experiment/Adam/loss.npy')
    Adam_a = np.load('experiment/Adam/accuracy.npy')
    epoch = [x for x in range(100)]
    if plot_type == 'loss':
        plt.plot(epoch, SGD_l, color='red', label='SGD')
        plt.plot(epoch, Adagrad_l, color='orange', label='Adagrad')
        plt.plot(epoch, RMSprop_l, color='blue', label='RMSprop')
        plt.plot(epoch, Adam_l, color='purple', label='Adam')
        plt.legend()
        plt.xlabel('train epoch')
        plt.ylabel('loss function value')
        plt.show()
    elif plot_type == 'accuracy':
        plt.plot(epoch, SGD_a, color='red', label='SGD')
        plt.plot(epoch, Adagrad_a, color='orange', label='Adagrad')
        plt.plot(epoch, RMSprop_a, color='blue', label='RMSprop')
        plt.plot(epoch, Adam_a, color='purple', label='Adam')
        plt.legend(loc='lower right')
        plt.xlabel('train epoch')
        plt.ylabel('classify accuracy')
        plt.show()
    else:
        raise ValueError


if __name__ == "__main__":
    main('accuracy')
