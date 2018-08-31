import tensorflow as tf


def mean_squared_error():
    print('\nMean Squeared Error')
    x = tf.constant(
        [
            [2, 4],
            [6, 8]
        ],
        dtype=tf.float32
    )
    print('x:\n%s' % x.eval())
    x_hat = tf.constant(
        [
            [1, 2],
            [3, 4]
        ],
        dtype=tf.float32
    )
    print('x_hat:\n%s' % x_hat.eval())
    MSE = tf.nn.l2_loss(x - x_hat)  # Mean Squared Error
    return MSE


def cross_entropy():
    print('\nCross Entropy')
    y = tf.constant(
        [
            [1, 0],
            [0, 1]
        ],
        dtype=tf.float32
    )
    print('y:\n%s' % y.eval())
    y_hat = tf.constant(
        [
            [3, 1],
            [2, 5]
        ],
        dtype=tf.float32
    )
    print('y_hat:\n%s' % y_hat.eval())
    CE = tf.reduce_mean(  # needs to be explicitly called
        tf.nn.softmax_cross_entropy_with_logits_v2(  # Cross Entropy
            logits=y_hat,
            labels=y
        )
    )
    return CE

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    print('MSE: %s\n' % session.run(mean_squared_error()))
    print('CE: %s\n' % session.run(cross_entropy()))
    session.close()
