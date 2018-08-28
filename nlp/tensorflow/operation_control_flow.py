import tensorflow as tf


def wrong_order(x, session):
    # tensorflow does not care about the statements order,
    # however the code is written,
    # unless explicitly told
    x_assign_op = tf.assign(x, x+5)
    z = x * 2
    return session.run(z)


def control_flow(x, session):
    x_assign_op = tf.assign(x, x+5)
    with tf.control_dependencies([x_assign_op]):
        z = x * 2
    return session.run(z)

if __name__ == '__main__':
    session = tf.InteractiveSession()
    x = tf.Variable(tf.constant(2.0), name='x')
    tf.global_variables_initializer().run()
    wrong_answer = wrong_order(x, session)
    print('\nexpects 14, get\n', wrong_answer)
    right_anser = control_flow(x, session)
    print('\nusing tf.control_dependencies to control the operation flow, gives the right answer:\n', right_anser)
    session.close()
