'''
This is mix match of 
- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
- https://github.com/tensorflow/ecosystem/blob/master/docker/mnist.py
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("relu1", layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("relu2", layer_2)
    # Output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer

def do_train(sess, x, y, init, mnist, apply_grads, loss, acc, merged_summary_op, global_step):
    # Run the initializer
    #sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels},session=sess))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

def device_and_target():
  print("FLAGS.task_index %s" % FLAGS.task_index)
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
    raise ValueError("Must specify an explicit `ps_hosts`")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
    raise ValueError("Must specify an explicit `worker_hosts`")

  cluster_spec = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
  })

  server = tf.train.Server(
      cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  return (
      tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster_spec),
      server.target,
  )


# Parameters
learning_rate = 0.01
training_epochs = 25
#training_epochs = 4
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example/'

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Cluster: app args
flags = tf.app.flags
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", 0,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task the performs the variable "
                     "initialization")
flags.DEFINE_string("ps_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("train_dir", None,
                    "Directory for storing the checkpoints")
FLAGS = flags.FLAGS

def main(_):
    worker_device, target = device_and_target()
    print("worker info: device - %s, target -%s" % (worker_device, target)) 
    with tf.device(worker_device):
#    with tf.device("/job:ps/task:0"):
      mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
      
      # tf Graph Input
      # mnist data image of shape 28*28=784
      x = tf.placeholder(tf.float32, [None, 784], name='InputData')
      # 0-9 digits recognition => 10 classes
      y = tf.placeholder(tf.float32, [None, 10], name='LabelData')
      # cluster 
      global_step = tf.contrib.framework.get_or_create_global_step()
        
      # Store layers weight & bias
      weights = {
          'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
          'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
          'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
      }
      biases = {
          'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
          'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
          'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
      }
      
#    with tf.device(worker_device):
      # Encapsulating all ops into scopes, making Tensorboard's Graph
      # Visualization more convenient
      with tf.name_scope('Model'):
          # Build model
          pred = multilayer_perceptron(x, weights, biases)
      
      with tf.name_scope('Loss'):
          # Softmax Cross entropy (cost function)
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
      
      with tf.name_scope('SGD'):
          # Gradient Descent
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          # cluster 
#          num_workers = len(FLAGS.worker_hosts.split(","))
#          print("number of workers: %s" % num_workers)
#          optimizer = tf.train.SyncReplicasOptimizer(
#            optimizer,
#            replicas_to_aggregate=num_workers, 
#            total_num_replicas=num_workers,
#            name="mnist_sync_replicas"
#          )
          
          # Op to calculate every variable gradient
          grads = tf.gradients(loss, tf.trainable_variables())
          grads = list(zip(grads, tf.trainable_variables()))
          # Op to update all variables according to their gradient
          apply_grads = optimizer.apply_gradients(grads_and_vars=grads,global_step=global_step)
      
      with tf.name_scope('Accuracy'):
          # Accuracy
          acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
          acc = tf.reduce_mean(tf.cast(acc, tf.float32))
      
      # Initialize the variables (i.e. assign their default value)
      init = tf.global_variables_initializer()
      
      # Create a summary to monitor cost tensor
      tf.summary.scalar("loss", loss)
      # Create a summary to monitor accuracy tensor
      tf.summary.scalar("accuracy", acc)
      # Create summaries to visualize weights
      for var in tf.trainable_variables():
          tf.summary.histogram(var.name, var)
      # Summarize all gradients
      for grad, var in grads:
          tf.summary.histogram(var.name + '/gradient', grad)
      # Merge all summaries into a single op
      merged_summary_op = tf.summary.merge_all()
    
    # Start training
    #sess = tf.Session()
    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=(FLAGS.task_index == 0),
        checkpoint_dir=FLAGS.train_dir) as sess:
      while not sess.should_stop():
        do_train(sess, x, y, init, mnist, apply_grads, loss, acc, merged_summary_op, global_step)

if __name__ == "__main__":
    tf.app.run()
