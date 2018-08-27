# index and access tensors
# scatter: assign values to specific indices of a given tensor
# gather: slice, get individual elements of a given tensor
import tensorflow as tf


def scatter_update_1d(session):
    ref = tf.Variable(
        tf.constant(
            [1, 9, 3, 10, 5],
            dtype=tf.float32
        ),
        name='scatter_update'
    )
    tf.global_variables_initializer().run()
    print('\nref: %s' % ref.eval().tolist())
    print('# 1d element update')
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


def scatter_update_nd(session):
    print('# initially zeros')
    print('# nd row update')
    indices = [[1], [3]]
    print('indices: %s' % indices)
    updates = tf.constant([[1, 1, 1], [2, 2, 2]])
    print('updates:\n%s' % updates.eval())
    shape = [4, 3]
    print('shape: %s' % shape)
    tf_scatter_update_nd = tf.scatter_nd(
        indices,
        updates,
        shape,
        name=None
    )
    result = session.run(tf_scatter_update_nd)
    return result


def scatter_update_nd_2(session):
    print('# initially zeros')
    print('# nd element update')
    indices = [[1, 0], [3, 1]]  # 2 x 2
    print('indices: %s' % indices)
    updates = tf.constant([1, 2])  # 2 x 1
    print('updates: %s' % updates.eval().tolist())
    shape = [4, 3]
    print('shape: %s' % shape)
    tf_scatter_update_nd = tf.scatter_nd(indices, updates, shape)
    result = session.run(tf_scatter_update_nd)
    return result


def gather_1d(session):
    print('# 1d element access')
    params = tf.constant(
        [1, 2, 3, 4, 5],
        dtype=tf.float32
    )
    print('params: %s' % params.eval().tolist())
    indices = [1, 4]
    print('indices: %s' % indices)
    tf_gather = tf.gather(
        params,
        indices,
        validate_indices=True,
        name=None
    )
    result = session.run(tf_gather)
    return result


def gather_nd(session):
    print('# nd slice (row) access')
    params = tf.constant(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ],
        dtype=tf.float32
    )
    print('params:\n%s' % params.eval())
    indices = [[0], [2]]
    print('indices: %s' % indices)
    tf_gather_nd = tf.gather_nd(params, indices, name=None)
    result = session.run(tf_gather_nd)
    return result


def gather_nd_2(session):
    print('# nd element access')
    params = tf.constant(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ],
        dtype=tf.float32
    )
    print('params:\n%s' % params.eval())
    indices = [[0, 1], [2, 2]]
    print('indices: %s' % indices)
    tf_gather_nd = tf.gather_nd(params, indices)
    result = session.run(tf_gather_nd)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    print('scatter ref: %s\n' % scatter_update_1d(session).tolist())
    print('scatter:\n%s\n' % scatter_update_nd(session))
    print('scatter:\n%s\n' % scatter_update_nd_2(session))
    print('gather 1d: %s\n' % gather_1d(session).tolist())
    print('gather nd (rows):\n%s\n' % gather_nd(session))
    print('gather nd (elements):\n%s\n' % gather_nd_2(session).tolist())
    session.close()
