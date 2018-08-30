import tensorflow as tf


def condition_select(x, y):
    condition = tf.constant([[True, False], [True, False]], dtype=tf.bool)
    print('Condition:\n%s' % condition.eval())
    x_cond_y = tf.where(condition, x, y)
    return x_cond_y

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    print('\nCompare X and Y:')
    x = tf.constant(
        [
            [1, 2],
            [3, 4]
        ],
        dtype=tf.int32
    )
    print('X:\n%s' % x.eval())
    y = tf.constant(
        [
            [4, 3],
            [3, 2]
        ],
        dtype=tf.int32
    )
    print('Y:\n%s\n' % y.eval())

    x_equal_y = tf.equal(x, y, name=None)
    x_less_y = tf.less(x, y)
    x_great_equal_y = tf.greater_equal(x, y)

    print('Is X == Y (element-wise)?\n%s\n' % session.run(x_equal_y))
    print('Is X < Y (element-wise)?\n%s\n' % session.run(x_less_y))
    print('Is X >= Y (element-wise)?\n%s\n' % session.run(x_great_equal_y))

    print('Select elements of X or Y depending on the condition')
    x_cond_y = condition_select(x, y)
    print('Result:\n%s' % session.run(x_cond_y))

    session.close()
