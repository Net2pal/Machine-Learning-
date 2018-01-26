# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 22:41:36 2018

@author: Pulkit
"""

import numpy as np
import matplotlib.pyplot as plt



def step_gradient(b_current, k_current, train_x, train_y, learning_rate):
    b_grad = 0
    k_grad = 0
    n = len(train_y)
    for i in range(n):
        x = train_x[i]
        y = train_y[i]
        b_grad += (1/n) * (((k_current * x) + b_current) - y)
        k_grad += (1/n) * x * (((k_current * x) + b_current) - y)
    new_b = b_current - (learning_rate * b_grad)
    new_k = k_current - (learning_rate * k_grad)
    return [new_b, new_k]


def run_descent(train_x, train_y, init_b, init_k, num_iters, learning_rate):
    b = init_b
    k = init_k
    for i in range(num_iters):
        b, k = step_gradient(b, k, train_x, train_y, learning_rate)
    return [b, k]

def drawPlot(train_x, train_y, m, b):
    plt.plot(train_x, train_y, 'ro')
    plt.plot([0, 11], [0 + b, 7000*m + b], color='b', linestyle='-', linewidth=2)
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

def run():
    file = 'houses.csv'
    points = np.array(np.genfromtxt(file, delimiter=',', skip_header=1))
    learning_rate = 0.0000001 

    train_x = points[:,0]  # sizes of the houses
    train_y = points[:,1]  # prices
    init_b = 0
    init_k = 0

    print('{} - number of training examples'.format(len(train_y)))
    print('k = 0, b = 0 | initial parameters')

    num_iters = 200
    [b, k] = run_descent(train_x, train_y, init_b, init_k, num_iters, learning_rate)

    print('k = %.2f, b = %.2f | final parameters' % (k, b))
    drawPlot(train_x, train_y, k, b)


if __name__ == '__main__':
    run()