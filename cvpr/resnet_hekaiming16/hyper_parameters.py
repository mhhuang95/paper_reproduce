# Minhui huang
# Reproducing ResNet --cvpr16
#-----------------------------------------------------------------------------------


import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('version', 'test_110', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries ''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


#hyper parameters regards training
tf.app.flags.DEFINE_integer('train_steps', 80000, '''total training steps''')
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''training batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 250, '''validation batch size''')
tf.app.flags.DEFINE_integer('test_batch_size', 125, '''testing batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.1, '''initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_rate', 0.1, '''How much to decay the learning rate each time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')


tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''the number of residual blocks''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


tf.app.flags.DEFINE_integer('padding_size', 2, ''' layers of zero padding on each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'cache/logs_repeat20/model.ckpt-100000', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'