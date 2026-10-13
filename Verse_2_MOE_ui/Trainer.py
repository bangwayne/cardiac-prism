import os
import sys
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import torch.multiprocessing as mp
import shutil

from matplotlib.testing.jpl_units import Epoch

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from Verse.Verse_model import Verse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from config.config import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
import torch.distributed as dist
from .Trainwrapper import VerseTrainWrapper
import random
import time


class Trainer:
    def __init__(self, model_cfg, train_cfg, log, run_path):
        self.exp_id = cfg['exp_id']
        self.stage = stage_cfg['name']
        self.use_amp = stage_cfg.amp

        local_rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        # setting up the model
        self.verse = nn.parallel.DistributedDataParallel(VerseTrainWrapper(model_cfg, train_cfg).cuda(),
                                                         device_ids=[gpu],
                                                         output_device=gpu,
                                                         gradient_as_bucket_view=True,
                                                         find_unused_parameters=True,
                                                         bucket_cap_mb=train_cfg.training.bucket_cap_md)

        self.train()
        parameter_groups = filter(lambda p: p.requires_grad, self.verse.parameters())
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=train_cfg.solver.base_lr,
            weight_decay=train_cfg.solver.weight_decay
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=cfg.solver.warmup_epochs,
                                                       max_epochs=train_cfg.training.epochs)
        self.train_dataloader = get_loader(train_cfg)

        ######
        self.size = stage_cfg['crop_size']

        # setting up logging
        self.log = log
        self.run_path = run_path
        self.log.log_string('model_size',
                            str(sum([param.nelement() for param in self.verse.parameters()])))
        self.log.log_string(
            'number_of_parameters_that_requires_gradient',
            str(
                sum([
                    param.nelement()
                    for param in filter(lambda p: p.requires_grad, self.verse.parameters())
                ])))
        self.log.log_string('torch version', torch.__version__)
        self.log.log_string('PIL version', PIL.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)

    def train(self):
        self._is_train = True
        self.integrator = self.train_integrator
        self.verse.train()
        return self

    def train_epoch(self, cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, gpu,
                    train_iter_num):

        rank = self.local_rank
        epoch_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True)
        # atexit.register(cleanup)
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
                # torch.distributed.barrier()
        model.train()

        for batch in epoch_iterator:
            for batch_data in batch:
                batch_data['epoch'] = epoch

            loss_dict = model(batch)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                print("loss_dict is a tensor!")
            else:
                losses = sum(loss_dict.values())

            if torch.isnan(losses).any():
                print("NaN detected in loss. Stopping training.")
                assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

            optimizer.zero_grad()
            losses.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            optimizer.step()
            print(f'[RANK {rank}: GPU {gpu}] EPOCH-{epoch} ITER-{train_iter_num} --- loss {losses.item()}')
            train_iter_num += 1
            if rank == 0:
                writer.add_scalar('train_iter/loss', losses, train_iter_num)
            epoch_loss += losses.item()

        scheduler.step()

        if cfg.TRAINING.DIST:
            epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()

        epoch_loss /= len(train_dataloader) + 1e-12

        print(f'{cfg.TRAINING.MODEL_SAVE_PATH} ==> [GPU {gpu}] ',
              'epoch_loss: {}'.format(epoch_loss))
        if rank == 0:
            writer.add_scalar('train/loss', epoch_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
        return epoch_loss, train_iter_num

    def save_checkpoint(self, it, epoch, save_copy=False):
        if self.local_rank != 0:
            return
        checkpoint = {
            'it': it,
            'epoch': epoch,
            'weights': self.verse.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        model_save_name = "verse_model_epoch_latest.pth"
        model_path = os.path.join(cfg.training.model_save_path, model_save_name)
        torch.save(checkpoint, model_path)
        self.log.info(f'Latest Checkpoint saved to {model_path}.')

        if save_copy:
            model_save_name = f"verse_model_epoch_{epoch}.pth"
            model_path = os.path.join(cfg.training.model_save_path, model_save_name)
            torch.save(checkpoint, model_path)
            self.log.info(f'{epoch}th epoch Checkpoint saved to {model_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        print("=> loading checkpoint '{}'".format(path))
        # loc = 'cuda:{}'.format(gpu)
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})
        print("Load Checkpoint from {}".format(path))
        print("Checkpoint keys:", checkpoint.keys())
        it = checkpoint['it']
        epoch = checkpoint['epoch']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.verse.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info('Network weights, optimizer states, and scheduler states loaded.')

        return it, epoch

def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--config_file", type=str, default='')
    # config
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('-num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for numpy
    np.random.seed(seed)
    # Set the seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For the cuDNN backend, set the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    """ Kill all child processes and destroy distributed group. """
    try:
        torch.distributed.destroy_process_group()
    except:
        pass
    os.system("pkill -f muti_gpu_train_Verse_Cardiac.py")  # Kill all processes related to training script
    sys.exit(0)


# Define a function to save cfg to a YAML file
def save_cfg_to_yaml(cfg, output_filename):
    with open(output_filename, "w") as f:
        f.write(cfg.dump())


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.stdout = open(log_file, 'a')
    sys.stderr = open(log_file, 'a')


def muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu, train_iter_num):
    epoch_loss = 0

    epoch_iterator = tqdm(
        train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True
    )
    # atexit.register(cleanup)
    if cfg.TRAINING.DIST:
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
            # torch.distributed.barrier()

    model.train()

    for batch in epoch_iterator:

        for batch_data in batch:
            batch_data['epoch'] = epoch

        loss_dict = model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            print("loss_dict is a tensor!")
        else:
            losses = sum(loss_dict.values())

        if torch.isnan(losses).any():
            print("NaN detected in loss. Stopping training.")
            assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

        optimizer.zero_grad()
        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        optimizer.step()
        print(f'[RANK {rank}: GPU {gpu}] EPOCH-{epoch} ITER-{train_iter_num} --- loss {losses.item()}')
        train_iter_num += 1
        if rank == 0:
            writer.add_scalar('train_iter/loss', losses, train_iter_num)
        epoch_loss += losses.item()

    scheduler.step()

    if cfg.TRAINING.DIST:
        epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()

    epoch_loss /= len(train_dataloader) + 1e-12

    print(f'{cfg.TRAINING.MODEL_SAVE_PATH} ==> [GPU {gpu}] ',
          'epoch_loss: {}'.format(epoch_loss))
    if rank == 0:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, train_iter_num


def muti_gpu_main_worker(gpu, ngpus_per_node, cfg, args):
    node_rank = int(cfg.TRAINING.NODE_RANK)
    rank = node_rank * ngpus_per_node + gpu
    # world_size = ngpus_per_node  # args.world_size
    world_size = ngpus_per_node * cfg.TRAINING.NUM_NODES
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.TRAINING.INIT_METHOD,
        rank=rank,
        world_size=world_size,
    )
    print('init_process_group finished')

    model = Verse(cfg).to(gpu)  # checkpoint for pretrained vit
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS, resume=False
    # )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=cfg.TRAINING.BUCKET_CAP_MD
    )

    if os.path.isfile(args.resume):
        # Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cfg_solver = cfg.SOLVER
    if cfg_solver.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_solver.BASE_LR,
            weight_decay=cfg_solver.WEIGHT_DECAY
        )

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=cfg_solver.WARMUP_EPOCH,
                                              max_epochs=150)
    num_epochs = cfg.TRAINING.NUM_EPOCH
    iter_num = 0

    train_dataloader = get_loader(cfg)

    start_epoch = 0
    if cfg.TRAINING.RESUME is not None:
        if os.path.isfile(cfg.TRAINING.RESUME):
            print("=> loading checkpoint '{}'".format(cfg.TRAINING.RESUME))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(cfg.TRAINING.RESUME, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAINING.RESUME, checkpoint['epoch']))

    if rank == 0:
        writer = SummaryWriter(log_dir='./tb_log/' + cfg.TRAINING.RUN_ID)
        print('Writing Tensorboard logs to ', './tb_log/' + cfg.TRAINING.RUN_ID)
    else:
        writer = None

    for epoch in range(start_epoch, num_epochs):
        with model.join():
            epoch_loss, iter_num = muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer,
                                                        epoch, rank,
                                                        gpu, iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host and epoch == 9:
            os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
            shutil.copyfile(__file__, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH,
                                                   cfg.TRAINING.RUN_ID + '_' + os.path.basename(__file__)))
        if is_main_host and epoch == 2:
            os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
            cfg_save_path = os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, 'config.json')
            with open(cfg_save_path, 'w') as cfg_file:
                json.dump(cfg, cfg_file, indent=4)

        if is_main_host and epoch > 50:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, 'latest_medsam_model.pth'))

        torch.distributed.barrier()


def main(cfg):
    # set seeds
    set_seed(2024)
    torch.cuda.empty_cache()
    if cfg.TRAINING.DIST:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9999'
        # ngpus_per_node = torch.cuda.device_count()
        ngpus_per_node = 4
        specific_gpus = [0, 1, 2, 3]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, specific_gpus))

        cfg.TRAINING.NGPUS = ngpus_per_node
        print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        mp.set_start_method('spawn')
        # mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
        try:
            mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
        except KeyboardInterrupt:
            print("\n[MAIN] Training interrupted. Cleaning up...")
            cleanup()
    else:
        # torch.multiprocessing.set_sharing_strategy('file_system')
        gpu = 0
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH,
                                               cfg.TRAINING.RUN_ID + '_' + os.path.basename(__file__)))
        main_worker(gpu, cfg)


if __name__ == "__main__":
    args = set_parse()
    cfg = setup(args)
    main(cfg=cfg)
    sys.exit(0)
