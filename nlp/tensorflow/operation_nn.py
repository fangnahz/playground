import tensorflow as tf

# Sigmoid activation: 1 / (1 + exp(-x))
# tf.nn.sigmoid(x)
# ReLU activation: max(0, x)
# tf.nn.relu(x)

# convolution:
#   1. element-wise multiplication on image patch that overlaps filter
#   2. sum
def convolution_example(session):
    x = tf.constant(
        [[
            [[1], [2], [3], [4]],
            [[4], [3], [2], [1]],
            [[5], [6], [7], [8]],
            [[8], [7], [6], [5]]
        ]],
        dtype=tf.float32
    )
    print('\ninput:\n%s' % x.eval())
    x_filter = tf.constant(
        [
            [
                [[0.5]], [[1]]
            ],
            [
                [[0.5]], [[1]]
            ]
        ],
        dtype=tf.float32
    )
    print('filter:\n%s' % x_filter.eval())
    x_strdie = [1, 1, 1, 1]
    x_padding = 'VALID'
    x_conv = tf.nn.conv2d(
        input=x,  # 4-d, [batch_size, height, width, channels]
                  #   * batch_size: images, words etc in a single batch of data
                  #   * height, width: of an input
                  #   * chnnels: depth of an input (RGB image channels is 3, one for each color)
        filter=x_filter,  # 4-d convolution window, [height, width, in_channels, out_channels]
                          #   * height, width: filter size (often smaller than input)
                          #   * in_channels: number of channels of the input to the layer
                          #   * out_channels: number of channels to be produced in the output of the layer
        strides=x_strdie,  # list, length 4, [batch_sride, height_stride, width_stride, channels_stride]
                           # number of elements to skip during a single shift of convolution window
                           # defaults to 1s
        padding=x_padding  # 'SAME' or 'VALID', how to handle convolution near boundaries
                           # 'VALID': no paddings, output smaller than input, (n - h + 1)
                           # 'SAME': zero paddings, output is of the same size as input
    )
    result = session.run(x_conv)
    return result

if __name__ == '__main__':
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    result = convolution_example(session)
    print('convolution result:\n%s\n' % result)
    session.close()
