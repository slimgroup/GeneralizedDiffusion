# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#Z
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import re
import json
import click
import torch 
import dnnlib
from torch_utils import distributed as dist
import dnnlib
from training import training_loop
import wandb
import warnings
import torch._dynamo
torch._dynamo.config.suppress_errors = True

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, default="")
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default = "")
@click.option('--dataset_main_name', help='Path to the dataset main folder', metavar='ZIP|DIR',     type=str, default = "")
@click.option('--dataset_main_name_cond', help='Path to the dataset main folder', metavar='ZIP|DIR',     type=str, default = "")
@click.option('--dataset_main_name_back', help='Path to the dataset main folder', metavar='ZIP|DIR',     type=str, default = "")

@click.option('--num_offsets',     help='number of offsets in positive and negative so must be multiplied by two for total non-zero offsets.', metavar='INT',   type=click.IntRange(min=0), default=0)
@click.option('--num_offsets',     help='number of offsets in positive and negative so must be multiplied by two for total non-zero offsets.', metavar='INT',   type=click.IntRange(min=0), default=0)
@click.option('--use_offsets',         help='Enable offsets', metavar='BOOL',                     type=bool, default=False, show_default=True)


@click.option('--cond_norm',       help='cond_norm', metavar='FLOAT',                       type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--gt_norm',       help='gt_norm', metavar='FLOAT',                       type=click.FloatRange(min=0), default=1.0, show_default=True)

@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True) 

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list, default=[1,2,2,2]  )
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0), default=2e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--max_grad_norm', help='Max norm for gradients.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--weight_decay', help='Value of weight decay. Set to 0. to disable.', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)

# Ambient diffusion
@click.option('--norm', help='Norm for loss', default=2, show_default=True)
@click.option('--gated', help='Whether to use gated convolutions', metavar='BOOL', default=True, show_default=True)

@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=30, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=30, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT', default=5, type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--wandb_id', help='Id of wandb run to resume', type=str, default='')
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# wandb
@click.option('--experiment_name', help='Name for the experiment to run', type=str, default="", required=False, show_default=True)
@click.option('--project_name', help='Name for the project (one project for many experiments) to run', type=str, default="conditional_diffusion", required=False, show_default=True)


def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

       
    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.update(max_grad_norm=opts.max_grad_norm)
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, dataset_main_name = opts.dataset_main_name,dataset_main_name_cond = opts.dataset_main_name_cond,dataset_main_name_back = opts.dataset_main_name_back,cond_norm = opts.cond_norm,gt_norm = opts.gt_norm, use_offsets = opts.use_offsets, xflip=opts.xflip, cache=opts.cache,                                 )


    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()

    if opts.weight_decay == 0.:
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    else:
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.weight_decay)

    # Network architecture.
    c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
    c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=64, channel_mult=[2,2,2], gated=opts.gated)
   
    # Preconditioning & loss function.
    c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    c.loss_kwargs.class_name = 'training.loss.ConditionalLoss'
    c.loss_kwargs.norm = opts.norm
    
    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres

    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not dnnlib.util.is_file(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'gpus{dist.get_world_size():d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    desc += f'-offsets{opts.use_offsets}'

    if dist.get_rank() == 0:
        wandb.init(project=opts.project_name,config=kwargs,name=desc)
   

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if dnnlib.util.is_dir(opts.outdir):
            prev_run_dirs = [x.split('/')[-1] for x in dnnlib.util.list_dir(opts.outdir)]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()


    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        dnnlib.util.create_dir(c.run_dir)

        with dnnlib.util.open_url(os.path.join(c.run_dir, 'training_options.json'), read_mode='wt') as f:
            json.dump(c, f, indent=2)
    
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
