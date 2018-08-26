import tensorflow as tf


def equal_to(x, y, session):
    x_equal_y = tf.equal(x, y, name=None)
    result = session.run(x_equal_y)
    return result


def less_than(x, y, session):
    x_less_y = tf.less(x, y)
    result = session.run(x_less_y)
    return result


def greater_equal(x, y, session):
    x_great_equal_y = tf.greater_equal(x, y)
    result = session.run(x_great_equal_y)
    return result


def condition_select(x, y, session):
    condition = tf.constant([[True, False], [True, False]], dtype=tf.bool)
    print('# select element from x if condition is True, else from y\n# condition: %s' % condition.eval().tolist())
    x_cond_y = tf.where(condition, x, y)
    result = session.run(x_cond_y)
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
    print('x: %s\n' % x.eval().tolist())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.int32
    )
    print('y: %s\n' % y.eval().tolist())
    print('x = y: %s\n' % equal_to(x, y, session).tolist())
    print('x < y: %s\n' % less_than(x, y, session).tolist())
    print('x >= y: %s\n' % greater_equal(x, y, session).tolist())
    print('select(x, y, condition): %s\n' % condition_select(x, y, session).tolist())
    session.close()
