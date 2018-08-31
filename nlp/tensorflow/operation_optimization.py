# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def optimization():
    graph = tf.get_default_graph()
    session = tf.InteractiveSession(graph=graph)

    # Optimizers 用于学习 NN 参数，最小化误差, e.g. cross entropy error
    tf_x = tf.Variable(tf.constant(2.0, dtype=tf.float32), name='x')
    tf_y = tf_x**2
    minimize_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf_y)

    x_series, y_series = [], []
    tf.global_variables_initializer().run()
    for step in range(5):
        _, x, y = session.run([minimize_op, tf_x, tf_y])
        print('Step: ', step, ', x: ', x, ', y: ', y)
        x_series.append(x)
        y_series.append(y)
    session.close()
    return x_series, y_series


def plot(x_series, y_series):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=25, h=5)
    ax.plot(np.arange(-4, 4.1, 0.1), np.arange(-4, 4.1, 0.1)**2)
    ax.scatter(x_series, y_series, c='red', linewidths=4)

    x_offset, y_offset = 0.02, 0.75
    ax.annotate(
        'Starting point', xy=(2.01, 4.1), xytext=(2.5, 8), arrowprops=dict(facecolor='black', shrink=0.01), fontsize=20
    )
    ax.annotate('Optimization path', xy=(2.01, 4.1), xytext=(0.6, 5), arrowprops=None, fontsize=20)

    for index, (x, y) in enumerate(zip(x_series, y_series)):
        if index == len(x_series) - 1:
            break
        ax.annotate(
            '', xy=(x_series[index+1], y_series[index+1]+y_offset), xytext=(x-x_offset, y+y_offset),
            arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.01), fontsize=20
        )

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title('Optimizing y=x^2', fontsize=22)
    fig.savefig('optimization.jpg')

if __name__ == '__main__':
    x_series, y_series = optimization()
    plot(x_series, y_series)
