import tensorflow as tf


def mean_squared_error(session):
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
    print('\nx_hat:\n%s' % x_hat.eval())
    MSE = tf.nn.l2_loss(x - x_hat)  # Mean Squared Error
    result = session.run(MSE)
    return result


def cross_entropy(session):
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
    print('\ny_hat:\n%s' % y_hat.eval())
    CE = tf.reduce_mean(  # needs to be explicitly called
        tf.nn.softmax_cross_entropy_with_logits_v2(  # Cross Entropy
            logits=y_hat,
            labels=y
        )
    )
    result = session.run(CE)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    print('MSE: %s\n' % mean_squared_error(session))
    print('CE: %s\n' % cross_entropy(session))
    session.close()
