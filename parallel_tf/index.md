# Parallel Tensorflow

This guide is intended as a in-depth guide on synchronized/asynchronized/custom synchronization for Tensorflow networks. The target audience is students and researchers who need to set up a fine-grained, controlled distributed Tensorflow environment. 

The motivation for this article is that Tensorflow is using a synchronization implementation which focuses more on stability and efficiency, and as a result becomes less controlable. 

## Example 1: Synchronous SGD

We start our discussion on synchronous training. Ideally, the synchronized sgd seeks to compute the sgd (Figure 1) via data parallelism. 
![Alt text](./sgd.png?raw=true "Figure 1")

Since it only distributes the data and averaging them for every batch of data, we expect to get exactly the same result for any number of workers, which looks like:
![Alt text](./sync_mnist.png?raw=true "Figure 2")
![Alt text](./sync_cifar10_naive_cnn.png?raw=true "Figure 3")

The official guide is vague on such setups and usages. In most of the settings, we could get somehow synchronized training, but we couldn't get the same results for arbitrary workers - or even worse, the results is non-repeatable even if we are running the same code and fixing the same random seed. Such characteristics are caused by the distributed protocol of Tensorflow, and in this tutorial we will try to build a controlled environment, such that our result will be fully repeatable and equivalent.

In the following excerpt of codes, I emphasizes how we implement the key points which controls the behavior of our distributed training session. (Different from the official guide I didn't use the make_session_run_hook. As in some [issues](https://github.com/tensorflow/tensorflow/issues/7970) described there's some race condition associated with it.) The hook is just a groups of ops that are needed in order to start the queues in SyncReplicaOptimizer. I recommend reading the source at Tensorflow's Github page for advanced information.

In short, we need to pay attention to following points:
0. Fix random seed before we import data.
1. Set up local init ops and recovery_wait_secs.
2. Chief needs to run those ops.
3. Don't shuffle data. (Different processes would mess up.)
4. Make sure you assign exactly same effective data for all the workers. e.g (100 * 1 = 50 * 2 = 25 * 4, for {1, 2, 4} workers accourdingly)
5. Set up a barrier to make sure that all workers are on the same batch. 



```

############ KEY PART 0 #################
# Fix random seeds. In order to repeat, #
# We need to fix three of them          #
# before we do anything related.        #
#########################################
import random
random.seed(0)
tf.set_random_seed(0)
np.random.seed(0)
############### END 0 ###################

def worker_train_func(job_name, task_index):
  cluster = ...
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
  if job_name == "ps":
    server.join()
  num_workers = len(worker_hosts)
  is_chief = (task_index == 0)
  worker_device = "/job:worker/task:%d/cpu:0" % task_index

  with tf.device(tf.train.replica_device_setter(
    worker_device=worker_device,
    ps_device="/job:ps/cpu:0",
    cluster=cluster)):
    loss = ...
    opt = tf.train.AdagradOptimizer(0.02)
    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_workers, total_num_replicas=num_workers)
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(processed_grads, global_step=global_step)
    init_op = tf.global_variables_initializer()

    ################ KEY PART 1 ###############
    # This part handles local init.           #
    # It is basically equivalent to the hook, #
    # but using a Supervisor is more          #
    # tunable, for reasons                    #
    # which we will find out later.           #
    # #########################################

    local_init_op = opt.local_step_init_op
    if is_chief:
      local_init_op = opt.chief_init_op
    ready_for_local_init_op = opt.ready_for_local_init_op
    chief_queue_runner = opt.get_chief_queue_runner()
    sync_init_op = opt.get_init_tokens_op()
    init_op = tf.global_variables_initializer()

    # Assigns ops to the local worker by default.
    # Create a "supervisor", which oversees the training process.

    ## IMPORTANT: recovery_wait_eecs=0.
    #  TF distributed protocol doesn't guarantee 
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./tmp/train_logs",
                             init_op=init_op,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             global_step=...,
                             recovery_wait_secs=0,
                             save_model_secs=600)

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_filters=["/job:ps", "/job:worker/task:0", "/job:worker/task:1"])
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    ################### END 1 #################



    ############# KEY PART 2 ##############
    # Since the sync replica optimizer is #
    # implemented via queue, we need to   #
    # run the queue_runner to initiate.   #
    #######################################

    if is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])
      test_writer = tf.summary.FileWriter('./tmp/train_logs_summary', sess.graph)

    ################## END 2 ###################

    while not sv.should_stop():

      ########### KEY PART 3 ############
      #    Shuffle need to be False.    #
      ###################################
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)

      ############ KEY PART 4 ############
      # Assign corresponding portions    #
      # to each thread.                  #
      ####################################

      for i in range(num_workers):
      if task_index == i:
        mini_batch_size = FLAGS.batch_size // num_workers
        start_index = int(mini_batch_size * i)
        end_index = int(mini_batch_size * (i+1))
        batch_xs = batch_xs[start_index:end_index]
        batch_ys = batch_ys[start_index:end_index]

      ############# END 4 #################

      
      train_feed = {x: batch_xs, y_: batch_ys}
      _ = sess.run([train_op], feed_dict=train_feed)

      ############## KEY PART 5 ###############
      # A barrier is required so that         #
      # all the threads are on the same page. #
      #########################################

      # This barrier needs to be setup and pass to each worker.
      # Depends on the worker type we may need various form of barrier.
      # See python Barrier implementation.

      barrier.wait()

      ############## END 5 ##########

```

## Example 2: Asynchornized SGD.

Async SGD is relatively easier to implement than sync sgd. It is faster too since we don't have to wait for all nodes to response. Usually it also demonstrates faster convergence than sync-SGD. However, since it goes to different gradient direction simultaneously, which directions cancel out, it usually fails to reach optimal results as synchronized version does, falling short of 1~2% of training accuracy on some network.

(Examplify: async training failed to converge to the performance level of sync training.)
![Alt text](./convergence_depth.png?raw=true "Title")

However, on networks with relatively simpler structure (which sometimes means there's less local minimum to trap the gradient descent, although not usually true), both asyn and async converges to the same performance.
![Alt text](./async_sync_group_conv.png?raw=true "Title")

Different from the synchronized setting, now we don't need to ensure simultaneous&sync updates for each batch. Refer to the code snippet before, we could remove the SyncReplicaOptimizer and then we get the asynchronized training model.

## Example 3: Process Group Training.
An attempt we made is mixing sync and async training model at the same time. According to a recent paper that Intel has published(https://arxiv.org/abs/1708.05256), the hybrid model would gain performance boost which is attributed from sync training, and speed boost which is attributed from async training.

(The hybrid model. Workers synchronize update on local parameter server, and local parameter servers send them to the central server.)
![Alt text](./hybrid.png?raw=true "Title")

In order to implement this architecture as in Tensorflow, we need to set up our own network structure accordingly. The codes are more involved, and I attached both a Cifar-10 example and MNIST example at the Github page. I would briefly mentions some key setup points here:

1. Define a central parameter server. This parameter server holds paramters to be updated asynchronously, and shall be visible by all workers.

2. Define several local parameter servers. Each server will receive gradients from its worker group. These server shall ONLY be visible to its workers and itself, for reasons we would mention later. (e.g use local_ps_id to control.) 
```
with tf.device("/job:ps/task:%d/cpu:0" % (local_ps_id, gpu_idx)):
```

3. Construct sync update procedure on each worker groups. Then, define some update operations, using the original optimizer to apply_gradients to the central ps asynchronously. (instead of using the sync one, since the sync one requires #num_workers updates and average them - it is not designed for individual updates anyway.)
```
def apply_gradient_to_ps(ps_server_id, average_grads, opt):
    server_vars = tf.get_collection(scope='ps_{0}'.format(ps_server_id),key=tf.GraphKeys.TRAINABLE_VARIABLES)
    average_grads = [tup for tup in average_grads if tup[0] is not None]
    # update_target_fn will be called everytime when new averaged_grads arrives at the ps.
    update_target_fn = []
    for tup in zip(average_grads, server_vars):
        grad, _ = tup[0]
        var = tup[1]
        grad_and_var = (grad, var)
        update_target_fn.append(grad_and_var)
    return opt.apply_gradients(update_target_fn)

def update_var(ps_id, ps_server_id):
    local_vars = tf.get_collection(scope="ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    ps_vars = tf.get_collection(scope="ps_{0}".format(ps_server_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)  

    # Fetch variable.
    fetch_ps_fn = []
    for target, source in zip(local_vars, ps_vars):
        fetch_ps_fn.append(tf.assign(target, source))
    # Group into functions
    fetch_ps_fn = tf.group(*fetch_ps_fn)
    return fetch_ps_fn

def train(....):
    # ...
    opt = tf.train.AdagradOptimizer(0.02)

    # ...
    sync_opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=group_num_replicas,
                           total_num_replicas=group_num_replicas)

    # Now use original optimizer.
    assign_to_ps_op = apply_gradient_to_ps(ps_server_id, processed_grads, opt)
    fetch_ps_op = update_var(ps_task_id, ps_server_id)
```
4. Initialization. For each group we need a chief! The Tensorflow documentation says that we could only define one chief because the chief will initialize all variables, but in our setting NONE of the workers could see all variables, and thus we need to initialize the variables separately on each process group. (In the example Github code, I set 4 processes, with workers [0,2] as group 1, workers [1,3] as group 2, and worker [0,1] as chief for supervisor. )

How does this perform? Well, refer to this early picture above, we could see that it has approximately the same performance as in the MNIST setting for synchronized training. This is definitely a good news, provided that this method requires less synchronization, which indicates higher efficiency.
![Alt text](./async_sync_group_conv.png?raw=true "Title")

This setup serves as an example on custom network training architecture using Tensorflow. There's still a lots of potential from customizing the training architecture, and associates/manages it via Kubernetes or similar cluster managers.

A possible next step for the research is to get a 4-GPU setting and experiment this method on Cifar-10, Cifar-100, or some bigger network with some famous architecture. 


PS:
Github Code page:
https://github.com/amirgholami/parallel_tf
This software has been developed and is maintained by the PALLAS group
at the ASPIRE Lab in the University of California, Berkeley.
More information please visit: 
http://aspire.eecs.berkeley.edu