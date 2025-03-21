import argparse
import os


def set_stu_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    # 数据集
    parser.add_argument('--train_dataset', type=str, default='live',
                        help='Support datasets: pipal|livec|koniq-10k|bid|live|csiq|tid2013|kadid10k')
    parser.add_argument('--test_dataset', type=str, default='live',
                        help='Support datasets: pipal|livec|koniq-10k|bid|live|csiq|tid2013|kadid10k|spaq')
    # 超参数
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for training')
    # 权重路径
    parser.add_argument('--studentNet_model_path', type=str, default='/data/user/cwz/IQA/checkpoint_Student/models/Student_kadid10k_pretrain_saved_model.pth',
                        help='./model_zoo/NAR_student_cross_dataset.pth')

    # parser.add_argument('--test_dataset', type=str, default='live',
    #                     help='Support datasets: pipal|livec|koniq-10k|bid|live|csiq|tid2013|kadid10k')
    parser.add_argument('--patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--self_patch_num', type=int, default=10,
                        help='number of training & testing image self patches')
    parser.add_argument('--train_test_num', type=int, default=1, help='Train-Util times')
    parser.add_argument('--update_opt_epoch', type=int, default=30)
    parser.add_argument('--train_patch_num', type=int, default=1, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', type=int, default=1, help='Number of sample patches from testing image')

    parser.add_argument('--net_print', type=int, default=2000)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_Student/')
    parser.add_argument('--use_fitting_prcc_srcc', type=str2bool, default=True)
    parser.add_argument('--print_netC', type=str2bool, default=False)

    args = parser.parse_args()

    ref_dataset_path = '/home/dataset/DIV2K/'
    args.ref_train_dataset_path = ref_dataset_path + 'train_HR_Sample/'
    args.ref_test_dataset_path = ref_dataset_path + 'val_HR_Sample/'

    # checkpoint files
    args.model_checkpoint_dir = args.checkpoint_dir + 'models/'
    args.result_checkpoint_dir = args.checkpoint_dir + 'results/'
    args.log_checkpoint_dir = args.checkpoint_dir + 'log/'

    if os.path.exists(args.checkpoint_dir) and os.path.isfile(args.checkpoint_dir):
        raise IOError('Required dst path {} as a directory for checkpoint saving, got a file'.format(
            args.checkpoint_dir))
    elif not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print('%s created successfully!' % args.checkpoint_dir)

    if os.path.exists(args.model_checkpoint_dir) and os.path.isfile(args.model_checkpoint_dir):
        raise IOError('Required dst path {} as a directory for checkpoint model saving, got a file'.format(
            args.model_checkpoint_dir))
    elif not os.path.exists(args.model_checkpoint_dir):
        os.makedirs(args.model_checkpoint_dir)
        print('%s created successfully!' % args.model_checkpoint_dir)

    if os.path.exists(args.result_checkpoint_dir) and os.path.isfile(args.result_checkpoint_dir):
        raise IOError('Required dst path {} as a directory for checkpoint results saving, got a file'.format(
            args.result_checkpoint_dir))
    elif not os.path.exists(args.result_checkpoint_dir):
        os.makedirs(args.result_checkpoint_dir)
        print('%s created successfully!' % args.result_checkpoint_dir)

    if os.path.exists(args.log_checkpoint_dir) and os.path.isfile(args.log_checkpoint_dir):
        raise IOError('Required dst path {} as a directory for checkpoint log saving, got a file'.format(
            args.log_checkpoint_dir))
    elif not os.path.exists(args.log_checkpoint_dir):
        os.makedirs(args.log_checkpoint_dir)
        print('%s created successfully!' % args.log_checkpoint_dir)

    return args


def check_stu_args(args, rank=0):
    if rank == 0:
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            print('%s: %s' % (str(k), str(v)))
        print('------------ End -------------')
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    args = set_stu_args()
