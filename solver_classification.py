from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
from model_agecomparison import Classificationmodel, getCossloss, getKLloss
import os
import time
import datetime
from torch import nn


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, classification_loader,test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.classification_loader = classification_loader
        self.test_loader=test_loader



        self.image_size = config.image_size


        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.classification_lr = config.classification_lr

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.vgg_type=config.vgg_type

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.classification_modle=Classificationmodel(self.vgg_type)


        self.classification_optimizer = torch.optim.Adam(self.classification_modle.parameters(), self.classification_lr, [self.beta1, self.beta2])
        self.print_network(self.classification_modle, 'G')

        self.classification_modle.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("Total number of parameters : %.3f M' " % (num_params / 1e6))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        classfication_path = os.path.join(self.model_save_dir, '{}-classfication.ckpt'.format(resume_iters))
        self.classification_modle.load_state_dict(torch.load(classfication_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.classification_modle.param_groups:
            param_group['lr'] = g_lr


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.classification_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def createage_labels(self, c_org, c_dim=5, selected_attrs=None):

        age_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ["(4,", "(25,", "(0,", "(8,", "(15,", "(38,", "(48,", "(60,"]:
                age_color_indices.append(i)
        c_trg_list = []
        for i in range(c_dim):

            c_trg = c_org.clone()
            if i in age_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in age_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA' or dataset == 'adience':
            return F.cross_entropy(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        # if self.dataset == 'CelebA':
        #     data_loader = self.celeba_loader
        # elif self.dataset == 'RaFD':
        #     data_loader = self.rafd_loader
        # elif self.dataset == 'adience':
        #     data_loader = self.adience_loader
        data_loader=self.classification_loader
        test_loader=self.test_loader
        self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
        # self.transform(image), torch.FloatTensor(one), torch.FloatTensor(cost_one), torch.FloatTensor(y_sig01)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        # Fetch fixed inputs for debugging.


        # Learning rate cache for decaying.
        classification_lr = self.classification_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                image, one,cost_one,y_sig01 = next(data_iter)
            except:
                data_iter = iter(data_loader)
                image, one,cost_one,y_sig01= next(data_iter)



            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            class_fc1, class_fc2=self.classification_modle(image)

            Cossloss=getCossloss(class_fc1, one, cost_one)
            KLloss=getKLloss(class_fc2, y_sig01)

            outloss=Cossloss+KLloss
            self.reset_grad()
            outloss.backward()
            self.classification_optimizer.step()


            # Logging.
            loss = {}
            loss['classfication/Cossloss'] = Cossloss.item()
            loss['classfication/KLloss'] = KLloss.item()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")

                log = "{}, Elapsed [{}], Iteration [{}/{}]".format(log_time, et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                with open(self.log_name, "a") as log_file:
                    log_file.write('%s\n' % log)  # save the message

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # # Translate fixed images for debugging.
            # if (i + 1) % self.sample_step == 0:
            #     with torch.no_grad():
            #         x_fake_list = [x_fixed]
            #         for c_fixed in c_fixed_list:
            #             x_fake_list.append(self.G(x_fixed, c_fixed))
            #         x_concat = torch.cat(x_fake_list, dim=3)
            #         sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
            #         save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            #         print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                classification_path = os.path.join(self.model_save_dir, '{}-classification.ckpt'.format(i + 1))
                torch.save(self.classification_modle.state_dict(), classification_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                classification_lr -= (self.classification_lr / float(self.num_iters_decay))

                self.update_lr(classification_lr)
                lr_str = 'Decayed learning rates, classification_lr: {}.'.format(classification_lr)
                with open(self.log_name, "a") as log_file:
                    log_file.write('%s\n' % lr_str)  # save the message

            if (i + 1) % self.sample_step == 0:
                CA3_sum=0
                CA5_sum=0
                for i_train_batch, train_batch in enumerate(test_loader):
                    image, one, cost_one, y_sig01=train_batch
                    class_fc1, class_fc2 = self.classification_modle(image)
                    result_index=torch.argmax(class_fc2,dim=-1)
                    CA3=torch.abs(result_index-y_sig01)<=3
                    CA5=torch.abs(result_index-y_sig01)<=5
                    CA3_sum+=torch.sum(CA3)
                    CA5_sum+=torch.sum(CA5)
                CA3_prescision=CA3_sum/torch.float(len(test_loader))
                CA5_prescision=CA5_sum/torch.float(len(test_loader))
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))



