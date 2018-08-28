import tensorflow as tf


def very_simple_computation(w):
    x = tf.Variable(tf.constant(5.0, dtype=tf.float32), name='x')
    y = tf.Variable(tf.constant(2.0, dtype=tf.float32), name='y')
    print('[simple] x: %s, y: %s' % (x.name, y.name))
    z = x*2 + y**2
    return z


def not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, dtype=tf.float32))
    print('[not so simple] x: %s, y: %s' % (x.name, y.name))
    z = x*w + y**2
    return z


def another_not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, dtype=tf.float32))
    print('[another not so simple] x: %s, y: %s' % (x.name, y.name))
    z = w*x*y
    return z

if __name__ == '__main__':
    session = tf.InteractiveSession()
    print('\ncalling very_simple_computation 3 times, each time variables will be created repeatedly:')
    for _ignored in range(3):
        w = 2
        z = very_simple_computation(w)
    print('\ncreate with scope, x => scopeA/x, y => scopeA/y')
    with tf.variable_scope('scopeA'):
        z1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
    print('\nreuse scopeA:')
    with tf.variable_scope('scopeA', reuse=True):
        z2 = another_not_so_simple_computation(z1)
    print('\ncreate antoher scope, x => scopeB/x, y => scopeB/y')
    with tf.variable_scope('scopeB'):
        a1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
    print('\nreuse scopeB:')
    with tf.variable_scope('scopeB', reuse=True):
        a2 = another_not_so_simple_computation(a1)
    print('\nreuse scopeA again:')
    with tf.variable_scope('scopeA', reuse=True):
        zz1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
        zz2 = another_not_so_simple_computation(z1)
    tf.global_variables_initializer().run()
    assert tuple(session.run([z1, z2, a1, a2, zz1, zz2])) == (9, 90, 9, 90, 9, 90)
    session.close()
