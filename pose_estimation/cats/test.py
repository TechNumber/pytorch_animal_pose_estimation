import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer

from conf.config_dataclasses import Config


def test(cfg: Config):
    seed_everything(cfg.seed, workers=True)

    PATH = 'checkpoints/epoch=45-step=414.ckpt'
    module = hydra.utils.get_class(cfg.lit_module._target_).load_from_checkpoint(
        checkpoint_path=PATH
    )
    dataset = instantiate(cfg.dataset.init, cfg=cfg)
    callbacks = cfg.callbacks and list(instantiate(cfg.callbacks).values())
    logger = instantiate(cfg.logger)
    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.test(module, dataset)


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def test_model(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=False))
    test(cfg)


if __name__ == '__main__':
    test_model()
