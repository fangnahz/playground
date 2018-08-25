# coding: utf-8
import tensorflow as tf
import numpy as np


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
    print('sigmoid: %s' % h_eval)


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
    print('preloaded sigmoid: %s' % h_eval)

if __name__ == '__main__':
    # 2. 服务员（session）会把客户点的餐记录在他的记事本，一个 tf.GraphDef 中
    graph = tf.Graph()
    # 3. session: 服务员
    #   3.1. 把客户点餐细节送到厨房，交给总厨（分布式 Master），总厨会分配任务给各个主厨、帮厨
    session = tf.InteractiveSession(graph=graph)

    # variables: mutalbe tensors，算法中使用的变量
    W = tf.Variable(tf.random_uniform(
        shape=[10, 5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W'
    )
    b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')
    tf.global_variables_initializer().run()  # 初始化变量

    the_sigmoid_example(W, b, session)
    sigmoid_preloaded(W, b, session)

    session.close()
