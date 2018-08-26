import tensorflow as tf


def tensor_equal(x, y, session):
    x_equal_y = tf.equal(x, y, name=None)
    result = session.run(x_equal_y)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    x = tf.constant(
        [
            [1, 2],
            [3, 4]
        ],
        dtype=tf.int32
    )
    print('x: %s' % x.eval().tolist())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.int32
    )
    print('y: %s' % y.eval().tolist())
    print('x = y: %s' % tensor_equal(x, y, session).tolist())
    session.close()
