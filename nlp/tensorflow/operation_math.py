import tensorflow as tf


def segment_sum():
    data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)
    print('Vector:               %s' % tf.cast(data, tf.int32).eval().tolist())
    segments = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    segment_ids = tf.constant(segments, dtype=tf.int32)
    print('Three segments 0/1/2: %s' % segments)
    seg_sum = tf.segment_sum(data, segment_ids)
    return seg_sum

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
    print('\nX:\n%s' % x.eval())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.float32
    )
    print('Y:\n%s\n' % y.eval())

    x_add_y = tf.add(x, y)
    x_mul_y = tf.matmul(x, y)
    log_x = tf.log(x)
    x_sum_rows = tf.reduce_sum(x, axis=[1], keepdims=False)
    x_sum_columns = tf.reduce_sum(x, axis=[0], keepdims=True)

    print('X + Y:\n%s\n' % session.run(x_add_y))
    print('XY (matrix mulitplication):\n%s\n' % session.run(x_mul_y))
    print('log(X):\n%s\n' % session.run(log_x))
    print('X sum rows:\n%s\n' % session.run(x_sum_rows))
    print('X sum columns:\n%s\n' % session.run(x_sum_columns))
    print('Segment sum of a vector')
    seg_sum = segment_sum()
    print('result:\n%s\n' % session.run(seg_sum))
    session.close()
