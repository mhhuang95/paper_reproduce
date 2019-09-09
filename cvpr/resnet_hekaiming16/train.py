# Minhui huang
# Reproducing ResNet --cvpr16
#-----------------------------------------------------------------------------------


from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd


class Train(object):
    '''
    responsible for all training and validation
    '''

    def __init__(self):
        self.placeholders()

    def placeholders(self):
        '''
        five placeholders in total
        :return:
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def build_train_validation_graph(self):
        '''
        build the main train and validation graph at the same time
        :return:
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)


        regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_loss)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss, self.train_top1_error)

        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)


    def train(self):
        '''
        main function for training
        :return:
        '''
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        vali_data, vali_labels = read_validation_data()


        self.build_train_validation_graph()

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print('Restored from checkpoint...')
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        step_list = []
        train_error_list = []
        val_error_list = []

        print("start training!")

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                                     FLAGS.train_batch_size)

            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data, vali_labels,
                                                                                      FLAGS.validation_batch_size)

            if step % FLAGS.report_freq == 0:
                if FLAGS.is_full_validation:
                    validation_loss_value, validation_error_value = self.full_validation(
                        loss=self.vali_loss,
                        top1_error=self.vali_top1_error,
                        vali_data=vali_data,
                        vali_labels=vali_labels,
                        session=sess,
                        batch_data=train_batch_data,
                        batch_label=train_batch_labels
                    )
                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.image_placeholder: train_batch_data,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)

            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                  self.full_loss, self.train_top1_error],
                                                                 {self.image_placeholder: train_batch_data,
                                                                  self.label_placeholder: train_batch_labels,
                                                                  self.vali_image_placeholder: validation_batch_data,
                                                                  self.vali_label_placeholder: validation_batch_labels,
                                                                  self.lr_placeholder: FLAGS.init_lr})

            duration = time.time() - start_time

            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})

                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                              sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('Validation loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)


            if step%10000 == 0 or (step + 1) == FLAGS.train_steps:
                check_point_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, check_point_path, global_step=step)
                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self, test_image_array):
        '''
         evaluate the test data. Please finish pre-precessing in advance
        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''

        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print("%d test batches in total"%num_batches)

        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)


        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)

        for step in range(num_batches):
            if step%10 == 0:
                print('%i batches finished!' %step)

            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                              feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array


    def loss(self, logits, labels):
        '''
        cross entropy loss givin logits and labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return:
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, names='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k:
        :return:
        '''
        batch_size = predictions.get_shape().as_list()[0]
        