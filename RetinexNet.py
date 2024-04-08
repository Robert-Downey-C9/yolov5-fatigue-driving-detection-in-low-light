from __future__ import print_function
import os
import argparse
from glob import glob
from PIL import Image
import tensorflow._api.v2.compat.v1 as tf
from models.RetinexNet.model import lowlight_enhance
from utils.RetinexNet.utils import *

tf.disable_v2_behavior()

# 命令行传递参数设置
'''
    命令行传递的参数如下：

    1. --use_gpu：是否使用GPU，默认是

    2. --gpu_idx：GPU号，默认为0

    3. --gpu_mem：显存利用率，默认为50%

    4. --phase：训练阶段或测试阶段，默认为训练阶段

    5. --epoch：训练次数，默认为100

    6. --batch_size：batch大小，默认为16

    7. --patch_size：patch大小，默认为48

    8. --start_lr：Adam的初始学习率，默认为0.001

    9. --eval_every_epoch：评估和保存模型的频率，默认为每训练20次评估和保存一次模型

    10.--checkpoint_dir：模型保存的目录，默认为./models/RetinexNet/checkpoint

    11.--sample_dir：评估结果保存的目录，默认为./results/sample

    12.--save_dir：测试结果保存的目录，默认为./results/test

    13.--test_dir：测试样本所在的目录，默认为./dataset/retinexnet/test/low

    14.--decom：是否需要保存分解结果，默认只保存最终结果
'''
parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='test', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='models/RetinexNet/checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./results/sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./results/test', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./dataset/retinexnet/test/low', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

# 模型训练函数
def lowlight_train(lowlight_enhance):

    # 检查和准备模型和评估结果保存的目录
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    # 学习率设置（训练20次之后的学习率均为初始学习率的0.1）
    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    # 顺序读取成对的低/正常光照训练样本
    train_low_data = []
    train_high_data = []

    train_low_data_names = glob('./dataset/retinexnet/our485/low/*.png') + glob('./dataset/retinexnet/syn/low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob('./dataset/retinexnet/our485/high/*.png') + glob('./dataset/retinexnet/syn/high/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    # 读取用于评估模型的样本（其中eval_high_data并没有用到）
    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('./dataset/retinexnet/eval/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    # 训练整个架构（先训练Decom-Net，再训练Enhance-Net）
    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), eval_every_epoch=args.eval_every_epoch, train_phase="Decom")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch, train_phase="Relight")

# 模型测试函数
def lowlight_test(lowlight_enhance):
    # 检查测试样本所在的目录是否存在，以及准备测试结果保存的目录
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 读取测试样本（其中test_high_data并没有用到）
    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)

    # 测试整个架构（其中test_high_data为无效参数）
    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()

