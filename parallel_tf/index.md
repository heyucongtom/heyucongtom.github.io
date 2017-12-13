## Parallel Tensorflow

This guide is intended as a in-depth guide on synchronized/asynchronized/custom synchronization for Tensorflow networks. The target audience is students and researchers who need to set up a fine-grained, controlled distributed Tensorflow environment. 

The motivation for this article is that Tensorflow is using a synchronization implementation which focuses more on stability and efficiency, and as a result becomes less controlable. 

# Example 1: Synchronous SGD

We start our discussion on synchronous training. Ideally, the synchronized sgd seeks to compute the sgd (Figure 1) via data parallelism. 
![Alt text](./sgd.png?raw=true "Figure 1")

Since it only distributes the data and averaging them for every batch of data, we expect to get exactly the same result for any number of workers, which looks like:
![Alt text](./sync_mnist.png?raw=true "Figure 1")

The official guide is vague on such setups and usages. In the following excerpt of codes, I emphasizes how we implement the key points which controls the behavior of our distributed training session. (Different from the official guide I didn't use the make_session_run_hook. As in some issues [link](https://github.com/tensorflow/tensorflow/issues/7970) described there's some race condition associated with it.)

```
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
    # tunable, which we will find out later.  #
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
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)

      ############ KEY PART 3 ############
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

      ############# END 3 #################

      
      train_feed = {x: batch_xs, y_: batch_ys}
      _ = sess.run([train_op], feed_dict=train_feed)

      ############## KEY PART 4 ###############
      # A barrier is required so that         #
      # all the threads are on the same page. #
      #########################################

      # This barrier needs to be setup and pass to each worker.
      # Depends on the worker type we may need various form of barrier.

      barrier.wait()

      ############## END 4 ##########

```





# Attachments: 
See fc_models/synchronized_sgd.py for fully controlled sync:
  - barrier setup (to overcome tensorflow protocol)
  - random seed fixing
  - batch shuffle control
  - recovery_wait_secs
  
See customCifarInputs for sync cifar-10 training with multiply gpu. Similar to above.

See process_group experiment in each model's folder.

![Alt text](./sync_cifar10_naive_cnn.png?raw=true "Cifar-10 sync training")

![Alt text](./async_sync_group_conv.png?raw=true "Title")

Examplify: async training failed to converge to the depth of sync training
![Alt text](./sync_async_cifar10.png?raw=true "Title")

![Alt text](./convergence_depth.png?raw=true "Convergence level in detail")  
This software has been developed and is maintained by the PALLAS group  
at the ASPIRE Lab in the University of California, Berkeley.

More information please visit: 
http://aspire.eecs.berkeley.edu