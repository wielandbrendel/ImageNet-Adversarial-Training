import foolbox
from foolbox.models import TensorFlowModel
import argparse
import nets
from tensorpack import TowerContext
import tensorflow as tf
from foolbox import zoo

from tensorpack.tfutils import get_model_loader


def create():
    # fetch weights
    weights_path = zoo.fetch_weights(
        'https://github.com/facebookresearch/ImageNet-Adversarial-Training/releases/download/v0.1/R152-Denoise.npz',
        unzip=False
    )
    
    # model constructer expects an ArgumentParser as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=152, choices=[50, 101, 152])
    parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                        default='ResNetDenoise')
    args = parser.parse_args([])

    model = getattr(nets, args.arch + 'Model')(args)

    image = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))

    with TowerContext(tower_name='', is_training=False):
        logits = model.get_logits(image)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    with session.as_default():
         model = get_model_loader(weights_path).init(session)
            
         fmodel = TensorFlowModel(image, logits, channel_axis=1, bounds=[0, 255.], preprocessing=(127.5, 127.5))

    return fmodel

