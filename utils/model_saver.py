import os.path

import torch


class ModelSaver:
    def __init__(self,
                 model,
                 batch_size,
                 save_freq=1,
                 start_epoch=0,
                 loss=None,
                 optimizer=None
                 ):
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.start_epoch = start_epoch

        self.model_name = str(type(model))
        self.model_name = self.model_name[self.model_name.rfind('.') + 1:-2]
        self.path = f'../../models/weights/{self.model_name}/'

        if loss:
            self.loss_name = str(type(loss))
            self.loss_name = self.loss_name[self.loss_name.rfind('.') + 1:-2]
            self.path += self.loss_name + '/'

        if optimizer:
            opt_param_excl = ['params', 'differentiable', 'maximize', 'foreach', 'weight_decay']
            self.opt_name = str(type(optimizer))
            self.opt_name = self.opt_name[self.opt_name.rfind('.') + 1:-2]
            self.opt_params = optimizer.state_dict()['param_groups'][0]
            self.opt_params = '_'.join(
                [f'{k}_{v}' for k, v in self.opt_params.items() if k not in opt_param_excl and not isinstance(v, bool)]
            ).replace('.', 'o').replace(',', '_').replace(' ', '_').replace('__', '_')
            self.path += self.opt_name + '_' + self.opt_params + '/'

        print("Model's weights will be saved to:", self.path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self,
             model,
             epoch
             ):
        if not (epoch + 1) % self.save_freq:
            torch.save(model.state_dict(),
                       (f'{self.path}{self.model_name}'
                        f'_E{epoch + self.start_epoch}'
                        f'_B{self.batch_size}'
                        '.pth'))
