import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import shutil,os

mnist = input_data.read_data_sets("data", one_hot=True)

learning_rate = 0.01
epochs = 100
batch_size = 100
display_step = 1

# 2次元の畳み込み層
#
def conv2d(input, weight_shape, bias_shape):
  incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
  weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming)**0.5)
  W = tf.get_variable("W", weight_shape, initializer=weight_init)
  bias_init = tf.constant_initializer(value=0)
  b = tf.get_variable("b", bias_shape, initializer=bias_init)
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME'), b))

# 大きさがkのウィンドウを使って
# MaxPooling
def max_pool(input, k=2):
  return tf.nn.max_pool(input, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

# 全結合層
def layer(input, weight_shape, bias_shape):
  weight_init = tf.random_normal_initializer(stddev=(2.0 / weight_shape[0])**0.5)
  bias_init = tf.constant_initializer(value=0)

  W = tf.get_variable("W", weight_shape, initializer=weight_init)
  b = tf.get_variable("b", bias_shape, initializer=bias_init)

  return tf.nn.relu(tf.matmul(input, W) + b)

# 層の積み重ね
def inference(x, keep_prob):
  # 1次元の情報を(28*28)*Nの情報に変換
  x = tf.reshape(x, shape=[-1,28,28,1])
  # 畳み込み1層
  with tf.variable_scope("conv_1"):
    conv_1 = conv2d(x, [5,5,1,32], [32])
    pool_1 = max_pool(conv_1)
  # 畳み込み2層
  with tf.variable_scope("conv_2"):
    conv_2 = conv2d(pool_1, [5,5,32,64], [64])
    pool_2 = max_pool(conv_2)
  # 全結合層
  with tf.variable_scope("fc"):
    pool_2_flat = tf.reshape(pool_2, [-1,7*7*64])
    fc_1 = layer(pool_2_flat, [7*7*64,1024], [1024])
  # 出力層
  with tf.variable_scope("output"):
    output = layer(fc_1, [1024,10], [10])

  return output

# 損失関数
def loss(output, y):
  xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
  loss = tf.reduce_mean(xentropy)

  return loss

# トレーニング
def trainng(cost, global_step):
  tf.summary.scalar("cost", cost)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(cost, global_step=global_step)

  return train_op

def evaluate(output, y):
  correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar("validation error", (1.0 - accuracy))

  return accuracy

if __name__ == "__main__":

# 毎回，ログディレクトリを作り直す
  if os.path.exists("conv_mnist_logs"):
    shutil.rmtree("conv_mnist_logs")

  with tf.device("/gpu:1"):
    with tf.Graph().as_default():
      with tf.variable_scope("conv_model"):

        x = tf.placeholder("float", [None, 784])
        y = tf.placeholder("float", [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        output = inference(x, keep_prob)

        cost = loss(output, y)

        global_step = tf.Variable(0, name="global_step", trainable=False)

        train_op = trainng(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("conv_mnist_logs/", graph=sess.graph)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        for epoch in range(epochs):
          avg_cost = 0
          total_batch = int(mnist.train.num_examples / batch_size)

          for i in range(total_batch):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

            sess.run(train_op,feed_dict={x: minibatch_x,y: minibatch_y})

            avg_cost += sess.run(cost, feed_dict={x:minibatch_x, y:minibatch_y, keep_prob:0.5}) / total_batch

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

