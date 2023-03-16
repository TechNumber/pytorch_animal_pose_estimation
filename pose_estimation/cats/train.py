import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer

from conf.config_dataclasses import Config

# -------BEFORE PYTORCH LIGHTNING-------

# def train(model,
#           data_train,
#           data_test,
#           loss,
#           optimizer,
#           epochs,
#           logger=None,
#           model_saver=None,
#           device=torch.device('cpu')):
#     model.to(device)
#
#     if logger:
#         logger.watch(model)
#
#     min_test_loss = 0
#     min_test_loss_epoch = 0
#
#     model.train()
#     for epoch in range(epochs):
#         batch_train_loss_list = []
#         for batch, batch_data in enumerate(data_train):
#             model.zero_grad()
#
#             train_img = batch_data['image'].to(device)
#             train_hmap = batch_data['heatmap'].to(device)
#
#             pred_hmaps = model(train_img)
#             loss_value = loss(pred_hmaps, train_hmap.unsqueeze(1))
#
#             batch_train_loss_list.append(loss_value.item())
#             loss_value.backward()
#
#             optimizer.step()
#
#         test_loss_value = test(model, loss, data_test, logger, device)
#         if epoch == 0 or test_loss_value < min_test_loss:
#             min_test_loss = test_loss_value
#             min_test_loss_epoch = epoch
#
#         if logger:
#             logger.log({
#                 'train/loss': np.average(batch_train_loss_list),
#                 'test/loss': test_loss_value,
#                 'test/min_test_loss': min_test_loss,
#                 'epoch': epoch
#             })
#
#         # print(f'Train loss: {epoch_train_loss_list[-1]}, Epoch: {epoch}')
#         # print(f'Test loss: {test_loss_value}, Epoch: {epoch}')
#         # if not epoch % logging_step:
#         # plt.figure()
#         # plt.ylim((0, 10))
#         # plt.plot(epoch_train_loss_list)
#         # plt.plot(epoch_test_loss_list, c='orange')
#         # plt.plot(min_epoch_test_loss_list, c='red')
#         # plt.show()
#         if model_saver:
#             model_saver.save(model, epoch)
#
#     if model_saver:
#         model_saver.save(model, epochs)
#
#     if logger:
#         logger.run.summary['min_test_loss'] = min_test_loss
#         logger.run.summary['min_test_loss_epoch'] = min_test_loss_epoch
#         logger.finish()


def train(cfg: Config):
    seed_everything(cfg.seed, workers=True)

    module = instantiate(cfg.lit_module, cfg=cfg)
    dataset = instantiate(cfg.dataset.init, cfg=cfg)
    callbacks = cfg.callbacks and list(instantiate(cfg.callbacks).values())
    logger = instantiate(cfg.logger)
    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(module, dataset)
    trainer.test(module, dataset)


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def train_model(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=False))
    train(cfg)


if __name__ == '__main__':
    train_model()

    # -------BEFORE PYTORCH LIGHTNING-------

    # INIT_WEIGHT_PATH = '../../models/weights/ConvolutionalPoseMachines___4_stages/HMapsMSELoss/Adam_lr_1e-05___betas_(0o9_0o999)_eps_1e-08/ConvolutionalPoseMachines_E899_B5.pth'
    # ALPHA = 0.00001
    # IMAGE_SIZE = (368, 368)
    # EPOCHS = 7
    # TRAIN_BATCH_SIZE = 5
    # TEST_BATCH_SIZE = 5
    # # LOG_STEP = 30
    # N_SUBSTAGES = 3
    # SAVE_MODEL_STEP = 90
    # START_EPOCH = 0

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # all_tform = transforms.Compose([
    #     RandomFlip(0.5, 0.5),
    #     RandomRatioCrop(0.1, 0.1, 0.9, 0.9),
    #     RandomRotation((-30, 30)),
    # ])
    #
    # img_tform = transforms.Compose([
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.ToTensor(),
    # ])
    #
    # data_train = AKD(
    #     json_file_path='../../datasets/cats_utils/train/keypoints_annotations.json',
    #     image_dir='../../datasets/cats_utils/train/labeled/',
    #     transform={'all': all_tform,
    #                'image': img_tform,
    #                'keypoints': transforms.ToTensor()},
    #     heatmap=True)
    # data_train_loader = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=3)
    #
    # data_test = AKD(
    #     json_file_path='../../datasets/cats_utils/test/keypoints_annotations.json',
    #     image_dir='../../datasets/cats_utils/test/labeled/',
    #     transform={'all': all_tform,
    #                'image': img_tform,
    #                'keypoints': transforms.ToTensor()},
    #     heatmap=True)
    # data_test_loader = DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=3)
    #
    # model = ConvolutionalPoseMachines(
    #     n_keypoints=16,
    #     n_substages=N_SUBSTAGES,
    #     n_base_ch=80,
    #     img_feat_ch=20
    # ).to(device)
    # if os.path.isfile(INIT_WEIGHT_PATH):
    #     model.load_state_dict(torch.load(INIT_WEIGHT_PATH))
    # else:
    #     print("Weights not found.")
    # optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
    # loss = HMapsMSELoss().to(device)
    #
    # model_saver = ModelSaver(
    #     model,
    #     TRAIN_BATCH_SIZE,
    #     save_freq=SAVE_MODEL_STEP,
    #     start_epoch=START_EPOCH,
    #     loss=loss,
    #     optimizer=optimizer
    # )
    #
    # logger = Logger(
    #     model,
    #     IMAGE_SIZE,
    #     EPOCHS,
    #     TRAIN_BATCH_SIZE,
    #     loss=loss,
    #     optimizer=optimizer,
    #     n_substages=N_SUBSTAGES,
    #     dataset='cats_utils',
    #     start_epoch=START_EPOCH
    # )
    #
    # train(
    #     model=model,
    #     data_train=data_train_loader,
    #     data_test=data_test_loader,
    #     loss=loss,
    #     optimizer=optimizer,
    #     epochs=EPOCHS,
    #     logger=logger,
    #     model_saver=model_saver,
    #     device=device
    # )
