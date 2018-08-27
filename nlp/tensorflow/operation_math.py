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


def column_sum(x, session):
    x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)
    result = session.run(x_sum_1)
    return result


def row_sum(x, session):
    x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True)
    result = session.run(x_sum_2)
    return result


def segment_sum(session):
    data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)
    print('data (casted to int): %s' % tf.cast(data, tf.int32).eval().tolist())
    segment_ids = tf.constant([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=tf.int32)
    print('three segments 0/1/2: %s' % segment_ids.eval().tolist())
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
    print('\nx: %s' % x.eval().tolist())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.float32
    )
    print('y: %s\n' % y.eval().tolist())
    print('x + y: %s\n' % tensor_add(x, y, session).tolist())
    print('# matrix, not element-wise multiplication\nx * y: %s\n' % matrix_multiplication(x, y, session).tolist())
    print('log(x): %s\n' % logarithm(x, session).tolist())
    print('x column sum: %s\n' % column_sum(x, session).tolist())
    print('x row sum: %s\n' % row_sum(x, session).tolist())
    print('segment sum: %s\n' % segment_sum(session))
    session.close()
