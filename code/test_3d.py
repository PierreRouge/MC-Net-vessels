import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MCNet', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet3d_v1', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 128, 128], help='Patch _size')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
test_save_path = "../model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = tuple(FLAGS.patch_size)
    with open(FLAGS.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/data.h5" for item in image_list]
    
elif FLAGS.dataset_name == "IXI":
    patch_size = tuple(FLAGS.patch_size)
    with open(FLAGS.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/data.h5" for item in image_list]

elif FLAGS.dataset_name == "Bullitt":
    patch_size = tuple(FLAGS.patch_size)
    with open(FLAGS.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/data.h5" for item in image_list]
    
elif FLAGS.dataset_name == "Liver":
    patch_size = tuple(FLAGS.patch_size)
    with open(FLAGS.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/data.h5" for item in image_list]
    
elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = tuple(FLAGS.patch_size)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas'
    with open(FLAGS.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type='vnet', in_chns=1, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, 'iter_15000.pth')
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=tuple(FLAGS.patch_size), stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=tuple(FLAGS.patch_size), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Bullitt":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=tuple(FLAGS.patch_size), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Liver":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=tuple(FLAGS.patch_size), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "IXI":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=tuple(FLAGS.patch_size), stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
