import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os,time,shutil

"""
  1層分の構築
"""
def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0 / weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.matmul(input, W) + b)

"""
  全体の構築
"""
def inference(x):
    with tf.variable_scope("hidden_1"):
        # 隠れ層1層目
        hidden_1 = layer(x, [784, 256], [256])
    with tf.variable_scope("hidden_2"):
        # 隠れ層2層目
        hidden_2 = layer(hidden_1, [256, 256], [256])
    with tf.variable_scope("hidden_3"):
        # 隠れそう3層目
        hidden_3 = layer(hidden_1, [256, 10], [10])

    return hidden_3

"""
  損失関数
"""
def loss(output, y):
    # ソフトマックスクロスエントロピーを計算
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y)
    # 平均を算出
    loss = tf.reduce_mean(xentropy)

    return loss

"""
  トレーニングプロセス
"""
def training(cost, global_step):
    tf.summary.scalar("cost",cost)
    # 最急降下法
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # コストを最小化する設定
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

"""
  評価関数
"""
def evaluate(output, y):
    correct_prediction = tf.equal(tf.arg_max(output, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation",accuracy)
    return accuracy

if __name__ == '__main__':
    #データ
    mnist = input_data.read_data_sets("data",one_hot=True)
    # パラメータ
    learning_rate = 0.01
    epochs = 100
    batch_size = 100
    display_step = 1

    # logディレクトリを作り直す
    if os.path.exists("mlp_logs"):
        shutil.rmtree("mlp_logs")

    # 計算グラフを書く際のMain部分といった感じのスコープ
    with tf.Graph().as_default():
        with tf.variable_scope("mlp_model"):
            x = tf.placeholder(tf.float32, [None, 784], name="x") # 入力データ
            y = tf.placeholder(tf.float32, [None, 10], name="y")  # 教師データ

            with tf.name_scope('inference') as scope:
              output = inference(x)
            with tf.name_scope('cost') as scope:
              cost = loss(output, y)
            with tf.name_scope('global_step') as scope:
              global_step = tf.Variable(0,name="global_step",trainable=False)
            with tf.name_scope('train') as scope:
              train_op = training(cost, global_step)
            with tf.name_scope('evaluate') as scope:
              eval_op = evaluate(output, y)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(
                "mlp_logs",
                graph_def = sess.graph_def
            )

            # 初期化オプション
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # 実際に学習を行う
            for epoch in range(epochs):

              avg_cost = 0
              total_batch = int(mnist.train.num_examples / batch_size)

              for i in range(total_batch):
                # get minibatch
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                # Training
                sess.run(train_op, feed_dict={x:minibatch_x, y:minibatch_y})
                # 平均誤差計算のために誤差を足していってる
                avg_cost += sess.run(cost, feed_dict={x:minibatch_x,y:minibatch_y}) / total_batch

                # 表示・ログ関連
              if epoch % display_step == 0:
                print("Epoch: {:04d},cost: {:.9f}".format(epoch+1, avg_cost))
                accuracy = sess.run(eval_op,feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                print("Validation Error: {}".format(1 - accuracy))
                summary_str = sess.run(summary_op, feed_dict={x:minibatch_x, y:minibatch_y})
                summary_writer.add_summary(summary_str,sess.run(global_step))
                saver.save(sess, os.path.join("mlp_logs", "model-checkpoint"), global_step=global_step)

            print("Optimizer Finished!")
            accuracy = sess.run(eval_op,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Test accuracy: {}".format(accuracy))

