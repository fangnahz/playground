# index and access tensors
# scatter: assign values to specific indices of a given tensor
# gather: slice, get individual elements of a given tensor
import tensorflow as tf


def scatter_update(ref, session):
    indices = [1, 3]
    print('indices: %s' % indices)
    updates = tf.constant(
        [2, 4],
        dtype=tf.float32
    )
    print('updates: %s' % updates.eval().tolist())
    tf_scatter_update = tf.scatter_update(
        ref,
        indices,
        updates,
        use_locking=None,
        name=None
    )
    result = session.run(tf_scatter_update)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    ref = tf.Variable(
        tf.constant(
            [1, 9, 3, 10, 5],
            dtype=tf.float32
        ),
        name='scatter_update'
    )
    print('\nref: %s' % ref.eval().tolist())
    tf.global_variables_initializer().run()
    print('ref updated: %s' % scatter_update(ref, session))
