import os
import argparse
from solver import Solver
from data_adience import get_loader
from torch.backends import cudnn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    rafd_loader=None


    adience_loader = get_loader(config.image_dir,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   config.dataset, config.mode, config.num_workers)


    # Solver for training and testing StarGAN.
    celeba_loader=None
    solver = Solver(adience_loader, celeba_loader,rafd_loader, config)

    if config.mode == 'train':

            solver.train()

    elif config.mode == 'test':

            solver.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=8, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='adience', choices=['CelebA', 'RaFD', 'Both','adience'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=50000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000
                        , help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=["(4,","(25,","(0,","(8,","(15,","(38,","(48,","(60,"])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='/home/smhe/Adience/aligned')
    parser.add_argument('--user_deconv', type=str2bool, default=True)
    parser.add_argument('--project_name' ,type=str, default='stargan_adience')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=500)

    config = parser.parse_args()
    config.log_dir=config.project_name+"/logs"
    config.model_save_dir=config.project_name+"/models"
    config.sample_dir = config.project_name + "/samples"
    config.result_dir = config.project_name + "/results"

    def mkdir(path):
        """create a single empty directory if it didn't exist

        Parameters:
            path (str) -- a single directory path
        """
        if not os.path.exists(path):
            os.makedirs(path)


    def mkdirs(paths):
        """create empty directories if they don't exist

        Parameters:
            paths (str list) -- a list of directory paths
        """
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                mkdir(path)
        else:
            mkdir(paths)


    def print_options( opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            # comment = ''
            # default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        mkdirs(opt.log_dir)
        file_name = os.path.join(opt.log_dir, '{}_opt.txt'.format(opt.mode))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    print(config)
    print_options(config)
    main(config)