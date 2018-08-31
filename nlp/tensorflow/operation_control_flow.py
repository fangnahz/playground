import tensorflow as tf


def wrong_order(x):
    # tensorflow does not care about the statements order,
    # however the code is written,
    # unless explicitly told
    x_assign_op = tf.assign(x, x+5)
    z = x * 2
    return z


def control_flow(x):
    x_assign_op = tf.assign(x, x+5)
    with tf.control_dependencies([x_assign_op]):
        z = x * 2
    return z

if __name__ == '__main__':
    # 如果不传入则使用 default_graph
    session = tf.InteractiveSession()

    x = tf.Variable(tf.constant(2.0), name='x')
    tf.global_variables_initializer().run()
    wrong_answer = session.run(wrong_order(x))
    print('\nexpects 14, gets\n', wrong_answer)

    right_anser = session.run(control_flow(x))
    print('\nusing tf.control_dependencies to control the operation flow, gives the right answer:\n', right_anser)

    session.close()
