import tensorflow as tf


def tensor_add(x, y, session):
    x_add_y = tf.add(x, y)
    result = session.run(x_add_y)
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
    print('\nx: %s' % x.eval().tolist())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.int32
    )
    print('y: %s\n' % y.eval().tolist())
    print('x + y: %s\n' % tensor_add(x, y, session).tolist())
    session.close()
