import tensorflow as tf


def tensor_add(x, y, session):
    x_add_y = tf.add(x, y)
    result = session.run(x_add_y)
    return result


def matrix_multiplication(x, y, session):
    x_mul_y = tf.matmul(x, y)
    result = session.run(x_mul_y)
    return result


def logarithm(x, session):
    log_x = tf.log(x)
    result = session.run(log_x)
    return result


def row_sum(x, session):
    x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)
    result = session.run(x_sum_1)
    return result


def column_sum(x, session):
    x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True)
    result = session.run(x_sum_2)
    return result


def segment_sum(session):
    data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)
    print('data (casted to int): %s' % tf.cast(data, tf.int32).eval().tolist())
    segments = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    segment_ids = tf.constant(segments, dtype=tf.int32)
    print('three segments 0/1/2: %s' % segments)
    x_seg_sum = tf.segment_sum(data, segment_ids)
    result = session.run(x_seg_sum)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    x = tf.constant(
        [
            [1, 2],
            [3, 4]
        ],
        dtype=tf.float32
    )
    print('\nx:\n%s' % x.eval())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.float32
    )
    print('y:\n%s\n' % y.eval())
    print('x + y:\n%s\n' % tensor_add(x, y, session))
    print('x * y (matrix mulitplication):\n%s\n' % matrix_multiplication(x, y, session))
    print('log(x):\n%s\n' % logarithm(x, session))
    print('x row sum: %s\n' % row_sum(x, session).tolist())
    print('x column sum: %s\n' % column_sum(x, session).tolist())
    print('segment sum: %s\n' % segment_sum(session).tolist())
    session.close()
