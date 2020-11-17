import argparse
import os
import numpy as np
from collections import OrderedDict
import warnings
import shutil

from mypath import Path
from dataloaders import make_data_loader
from utils.calculate_weights import calculate_weigths_labels
from utils.metrics import Evaluator
from dataloaders.utils import decode_seg_map_sequence

import torch
from torch.nn import functional as F

from torchvision.utils import make_grid

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from efficientunet import get_efficientunet_b4


class EfficientUnetModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        kwargs = {'num_workers': hparams.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(hparams, **kwargs)
        self.num_img_tr = len(self.train_loader)
        self.pretrained_net = get_efficientunet_b4(out_channels=self.nclass, concat_input=True, pretrained=True)
        if hparams.use_balanced_weights:
            parameters_dir = "/work/scratch/lei/MyProject/t_chucai/models_and_parameters/parameters/classes_weights"
            classes_weights_path = os.path.join(parameters_dir, hparams.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(hparams.dataset, self.train_loader, self.nclass)
            self.weight = torch.from_numpy(weight.astype(np.float32))
        else:
            self.weight = None
        self.evaluator = Evaluator(self.nclass)

    def forward(self, X):
        return self.pretrained_net(X)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["label"]
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks.long(), weight=self.weight.to(self.device), ignore_index=255)
        tensorboard_logs = {'loss/train': loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tensorboard_logs, "log": tensorboard_logs}
        )
        if batch_idx % (self.num_img_tr // 10) == 0:
            global_step = batch_idx + self.num_img_tr * self.current_epoch
            self.visualize_image(self.hparams.dataset, images, masks, outputs, global_step)
        return output

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["label"]
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks.long(), weight=self.weight.to(self.device), ignore_index=255)
        pred = outputs.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        masks = masks.cpu().numpy()
        self.evaluator.add_batch(masks, pred)
        return {"loss/val": loss}

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}
        tensorboard_logs["loss/val"] = torch.tensor(
            [output["loss/val"] for output in outputs]
        ).mean()
        tensorboard_logs["val/Acc"] = self.evaluator.Pixel_Accuracy()
        tensorboard_logs["val/Acc_class"] = self.evaluator.Pixel_Accuracy_Class()
        tensorboard_logs["val/mIoU"] = self.evaluator.Mean_Intersection_over_Union()
        tensorboard_logs["val/fwIoU"] = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.evaluator.reset()
        return {
            "progress_bar": tensorboard_logs,
            "log": tensorboard_logs,
            "loss/val": tensorboard_logs["loss/val"],
            "val/Acc": tensorboard_logs["val/Acc"],
            "val/Acc_class": tensorboard_logs["val/Acc_class"],
            "val/mIoU": tensorboard_logs["val/mIoU"],
            "val/fwIoU": tensorboard_logs["val/fwIoU"],
        }

    def visualize_image(self, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        self.logger.experiment.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        self.logger.experiment.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        self.logger.experiment.add_image('Groundtruth label', grid_image, global_step)


def new_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: coco)')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: 4)')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--min_epochs', type=int, default=1,
                        help="Minimum number of epochs to train for")
    parser.add_argument('--max_epochs', type=int, default=50,
                        help="Maximum number of epochs to train for")
    parser.add_argument('--eval_freq', type=int, default=1,
                        help="Validate every --eval_freq epochs")
    parser.add_argument('--save_top_k', type=int, default=1,
                        help="Save top k models. -1 = save all models")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="adam: learning rate")
    parser.add_argument('--logpath', type=str, default=os.getcwd(),
                        help="The path where logs should go")
    parser.add_argument('--version', type=int, default=None,
                        help="Use a deterministic version for logging")
    return parser


def main():
    parser = new_parser()
    hparams = parser.parse_args()
    model = EfficientUnetModel(hparams)
    logger = TestTubeLogger(
        save_dir=hparams.logpath,
        name="lightning_logs",
        version=hparams.version,
    )
    monitor = "val/mIoU"
    mode = "max"
    # Construct the default ModelCheckpoint in order to adjust it.
    version_path = '{}/{}/version_{}'.format(
        hparams.logpath,
        logger.experiment.name,
        logger.experiment.version)
    ckpt_path = '{}/{}/{}'.format(version_path, 'checkpoints', '{epoch:03d}')
    checkpoint_callback = ModelCheckpoint(filepath=ckpt_path,
                                          monitor=monitor,
                                          save_top_k=hparams.save_top_k,
                                          mode=mode)  # save best model
    ckpts = []
    for el in os.scandir(checkpoint_callback.dirpath):
        if el.name.endswith('.ckpt'):
            ckpts.append(el.path)
    if len(ckpts) > 1:
        # should be able to remove this as ModelCheckpoint should be fixed
        # in pytorch_lightning now
        warnings.warn("more than 1 checkpoint found, assuming latest = best",
                      RuntimeWarning)
        all_list = [int(ckpt.split('=')[-1].strip('.ckpt')) for ckpt in ckpts]
        index = np.argmax(np.array(all_list))
        warnings.warn("latest ckpt was {}".format(all_list[index]),
                      RuntimeWarning)
        ckpt = ckpts[index]
    elif len(ckpts) == 1:
        ckpt = ckpts[0]
    else:
        ckpt = None
    trainer = Trainer(
        logger=logger,
        default_root_dir=hparams.logpath,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        min_epochs=hparams.min_epochs,
        check_val_every_n_epoch=hparams.eval_freq,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=20,
        resume_from_checkpoint=ckpt
    )

    try:
        trainer.fit(model)
    except RuntimeError as e:
        if not ('corrupted' in str(e) or 'in loading state_dict' in str(e)):
            raise  # Something else is wrong.
        print(e)
        # If the job is put back into queue when writing weights, the file will
        # be corrupted.
        # In this case, there is nothing we can do except to start fresh.
        print("Removing broken checkpoint: {}".format(version_path))
        shutil.rmtree(version_path, ignore_errors=True)
        # and restart training
        trainer.fit(model)


if __name__ == "__main__":
    main()
