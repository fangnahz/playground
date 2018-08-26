# coding: utf-8
import numpy as np
import tensorflow as tf


def the_sigmoid_example(W, b, session):
    '''
    The sigmoid example:
      h = sigmoid(W * x + b)
      sigmoid(x) = 1 / (1 + exp(-x))
    1. 任务: 顾客点餐，说明要什么样的食物，定义出一个 graph

    '''
    # 4. 主厨查看订单，告诉帮厨（parameter server
    #   4.1. 准备各种材料（常数，变量）
    #   4.2. 保存制作过程中的半成品材料（中间变量）
    # placeholder 是一个输入占位符，输入，用于算法的训练、测试
    # x 不是变量，不需要定义后执行 global_variables_initializer
    x = tf.placeholder(
        dtype=tf.float32,  # 占位符要求的输入数据类型
        shape=[1, 10],  # 占位符形状，一维向量
        name='x'  # 占位符名字，可选，在调试过程中使用
                  # tensorflow 不知道 python 框架中使用的变量名（上面的 x），只知道这里指定的名称
    )

    # 5. 主厨（operation executor）使用帮厨提供的材料根据顾客订单制作食物
    # operation: 使用输入、算法计算输出
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

    # 6. 总厨拿到主厨做好交出的食物，交给服务员，服务员把食物送给顾客
    # 执行运算获得输出，immutable tensors，保存最终结果，以及中间结果
    h_eval = session.run(
        h,
        feed_dict={x: np.random.rand(1, 10)}  # feed_dict 参数给 placeholder 赋值
    )
    print('\nsigmoid: %s\n' % h_eval)


def sigmoid_preloaded(W, b, session):
    # x: pre-loaded input
    # x 不是变量，不需要定义后执行 global_variables_initializer
    # 使用 constant 而不是 placeholder，不需要在执行 session.run 的时候赋值
    # 缺点：x 固定，不能在执行 session.run 时使用不同的 x 观察输出变化
    x = tf.constant(
        value=[[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]],
        dtype=tf.float32,
        name='x'
    )

    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

    h_eval = session.run(h)  # 不需要提供 feed_dict 参数
    print('preloaded sigmoid: %s\n' % h_eval)


def sigmoid_pipeline(W, b, session):
    # input pipelines 用于快速处理大量数据
    # 使用 queue 传递数据，支持预处理，支持并发
    # 创建输入管道
    filenames = ['test%d.txt' % i for i in range(1, 4)]
    filename_queue = tf.train.string_input_producer(
        filenames,
        capacity=3,  # amount of data held in the queue at a given time
        shuffle=True,  # shuffle data before spitting out
        name='string_input_producer'
    )
    # 检查文件是否存在
    for fn in filenames:
        if not tf.gfile.Exists(fn):
            raise ValueError('Failed to find file: ' + fn)
        else:
            print('File %s found.' % fn)
    # 在几个文件中每行代表一条数据，使用 TextLineReader
    reader = tf.TextLineReader()
    # read() 方法输入是文件名 queue，从文件中读数据，逐个输出 key、value
    #   key: 指明文件、当前读取的数据（行），这里不需要使用
    #   value: 当前读到的行
    _ignore_key, value = reader.read(filename_queue, name='text_read_op')
    # record_defaults: 在读到错误数据时使用
    record_defaults = [[-1.], [-1.], [-1.], [-1.], [-1.], [-1.], [-1.], [-1.], [-1.], [-1.]]
    # 把读到的文本行数据解码成 columns，输入文本中每行有 10 列
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(
        value,
        record_defaults=record_defaults
    )
    # 合并上面得到的 10 个数据成一个 tensor，即 feature，包含全部 columns
    featrues = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])
    # shuffle_batch 方法随机 shuffle 上面得到的 feature，得到指定 size 的一个 batch 赋值给 x
    # 数据从 .txt 文件中读取
    x = tf.train.shuffle_batch(
        [featrues],
        batch_size=3,
        capacity=5,
        name='data_batch',
        min_after_dequeue=1,
        num_threads=1
    )
    # Coordinator 类可以视为 thread manager，比如可以使用多线程 enqueue，dequeue 等，
    # 管理多个 QueueRunners
    coord = tf.train.Coordinator()
    # 自动创建 QueueRunner，从定义好的 queue 中获取数据，需要 explicitly 执行
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

    # 使用上面得到的 x 计算 h，打印出（5 步）执行的结果
    for step in range(5):
        x_eval, h_evla = session.run([x, h])  # 如果不指定 x，则 parameter server 不保存 x 结果，运算一样可以正常执行
        print('========== Setp %d ==========' % step)
        print('Evaluated data (x)')
        print(x_eval)
        print('Evaluated data (h)')
        print('%s\n' % h_evla)

    # 最后需要停止、join threads，否则会一直 hang 住
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    # 2. 服务员（session）会把客户点的餐记录在他的记事本，一个 tf.GraphDef 中
    graph = tf.Graph()
    # 3. session: 服务员
    #   3.1. 把客户点餐细节送到厨房，交给总厨（分布式 Master），总厨会分配任务给各个主厨、帮厨
    session = tf.InteractiveSession(graph=graph)

    # variables: mutalbe tensors，算法中使用的变量
    W = tf.Variable(
        tf.random_uniform(
            shape=[10, 5],  # 每个纬度的 size，这个变量的长度是 tensor 的 rank，注意与矩阵 rank 概念的区别
            minval=-0.1,
            maxval=0.1,
            dtype=tf.float32  # 运算中数据类型要保持一直，如果不一致需要 explicitly 使用 tf.cast 做类型转换
                              # e.g.: tf.cast(x, dype=tf.float32)
        ),  # 直接指定变量初始值
        name='W'  # 是 Grpah 中变量的 ID，缺省则会使用 tensorflow 的默认命名 scheme
                  # 如前所述，TensorFlow 的 Graph 不感知 python 框架中所使用的变量名
    )
    b = tf.Variable(
        tf.zeros(
            shape=[5],
            dtype=tf.float32
        ),  # 直接指定变量初始值
        name='b'
    )
    # 初始化变量，在这个命令后如果再定义变量，就需要定义以后再次执行，否则变量在 operation 中不可用，会抛出异常:
    #   FailedPreconditionError (see above for traceback): Attempting to use uninitialized value <var_name>
    tf.global_variables_initializer().run()

    the_sigmoid_example(W, b, session)
    sigmoid_preloaded(W, b, session)
    sigmoid_pipeline(W, b, session)

    session.close()
