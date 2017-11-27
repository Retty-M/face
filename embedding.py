import os
import re
from tensorflow.python.platform import gfile
import nn4 as network
import tensorflow as tf

# Facenet embedding parameters

# Directory containing the graph definition and checkpoint files
model_dir = './data/model.ckpt-500000'
# Points to a module containing the definition of the inference graph
model_def = 'models.nn4'
# Image size (height, width) in pixels
image_size = 96
# The type of pooling to use for some of the inception layers {'MAX', 'L2'}
pool_type = 'MAX'
# Enables Local Response Normalization after the first layers of the inception network
use_lrn = False
# Random seed
seed = 42
# Number of images to process in a batch
batch_size = None


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


# Restore facenet model
print('Restore facenet embedding model')
tf.Graph().as_default()
sess = tf.Session()
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)

# ema = tf.train.ExponentialMovingAverage(1.0)
# saver = tf.train.Saver(ema.variables_to_restore())
# ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
# saver.restore(sess, ckpt.model_checkpoint_path)
saver = tf.train.import_meta_graph('./models/20170511-185253/model-20170511-185253.meta')
saver.restore(sess, './models/20170511-185253/model-20170511-185253.ckpt-80000')

# model_checkpoint_path = './models/model-20160506.ckpt-500000'
# ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
# model_checkpoint_path='model-20160506.ckpt-500000'

# saver.restore(sess, ckpt.model_checkpoint_path)
# saver.restore(sess, model_checkpoint_path)
# load_model('./models/20170511-185253')
print('Facenet embedding model restore success')
