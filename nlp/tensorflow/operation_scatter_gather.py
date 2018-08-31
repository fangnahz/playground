# index and access tensors
# scatter: assign values to specific indices of a given tensor
# gather: slice, get individual elements of a given tensor
import tensorflow as tf


def scatter_update_1d():
    ref = tf.Variable(
        tf.constant(
            [1, 9, 3, 10, 5],
            dtype=tf.float32
        ),
        name='scatter_update'
    )
    print('\nScatter update operation for 1-D')
    tf.global_variables_initializer().run()
    print('\nref: %s' % ref.eval().tolist())
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
    return tf_scatter_update


def scatter_update_nd():
    print('# Initially zeros')
    print('Scatter row update operation for n-D')
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
    return tf_scatter_update_nd


def scatter_update_nd_2():
    print('# initially zeros')
    print('Scatter element update operation for n-D')
    indices = [[1, 0], [3, 1]]  # 2 x 2
    print('indices: %s' % indices)
    updates = tf.constant([1, 2])  # 2 x 1
    print('updates: %s' % updates.eval().tolist())
    shape = [4, 3]
    print('shape: %s' % shape)
    tf_scatter_update_nd = tf.scatter_nd(indices, updates, shape)
    return tf_scatter_update_nd


def gather_1d():
    print('Gather operation for 1-D, access element')
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
    return tf_gather


def gather_nd():
    print('Gather operation for n-D, slice (row) access')
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
    return tf_gather_nd


def gather_nd_2():
    print('Gather operation for n-D, element access')
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
    return tf_gather_nd

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    print('updated:\n%s\n' % session.run(scatter_update_1d()).tolist())
    print('updated:\n%s\n' % session.run(scatter_update_nd()))
    print('updated:\n%s\n' % session.run(scatter_update_nd_2()))
    print('gather 1-D element:\n%s\n' % session.run(gather_1d().tolist()))
    print('gather n-D rows:\n%s\n' % session.run(gather_nd()))
    print('gather n-D elements:\n%s\n' % gather_nd_2(session).tolist())
    session.close()
