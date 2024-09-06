import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_utils import distributed as dist
import dnnlib
from training import dataset
from torch_utils.misc import StackedRandomGenerator
import json
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import argparse
import colorcet as cc
import pdb

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def ambient_sampler(
    net, latents, randn_like=torch.randn_like,
    num_steps=10, sigma_min=0.1, sigma_max=80, rho=7,
    S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=10,
    cond_loc = "",
    image_dir = "",
    cond=None
    ):
   
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0    

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            net_input = torch.cat([x_next, cond], dim=1)
            denoised = net(net_input, t_hat).to(torch.float64)[:, :1]
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                net_input = torch.cat([x_next, cond], dim=1)
                denoised = net(net_input, t_next).to(torch.float64)[:, :1]
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc, gt_loc, device=torch.device('cuda'),  **sampler_kwargs):

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    seeds = seeds[:num_generate]
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    print(num_batches)
    dist.print0(f"The algorithm will run for {num_batches} batches --  {len(seeds)} images of batch size {max_batch_size}")
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # the following has for each batch size allocated to this GPU, the indexes of the corresponding images.
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    batches_per_process = len(rank_batches)
    dist.print0(f"This process will get {len(rank_batches)} batches.")

    # load training options
    with dnnlib.util.open_url(training_options_loc, verbose=(dist.get_rank() == 0)) as f:
        training_options = json.load(f)

    if training_options['dataset_kwargs']['use_labels']:
        assert num_classes > 0, "If the network is class conditional, the number of classes must be positive."
        label_dim = num_classes
    else:
        label_dim = 0

    interface_kwargs = dict(img_resolution=training_options['dataset_kwargs']['resolution'], label_dim=label_dim, img_channels=2)
    network_kwargs = training_options['network_kwargs']
    model_to_be_initialized = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module

    # find all *.pkl files in the folder network_loc and sort them
    files = dnnlib.util.list_dir(network_loc)
    # Filter the list to include only "*.pkl" files
    pkl_files = [f for f in files if f.endswith('.pkl')]

    # Sort the list of "*.pkl" files
    sorted_pkl_files = sorted(pkl_files)
    sorted_pkl_files = [sorted_pkl_files[-1]] # use only the most recent network

    checkpoint_numbers = []
    for curr_file in sorted_pkl_files:
        checkpoint_numbers.append(int(curr_file.split('-')[-1].split('.')[0]))
    checkpoint_numbers = np.array(checkpoint_numbers)
    
    for checkpoint_number, checkpoint in zip(checkpoint_numbers, sorted_pkl_files):
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        network_pkl = os.path.join(network_loc, f'network-snapshot-{checkpoint_number:06d}.pkl')
        # Load network.
        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            loaded_obj = pickle.load(f)['ema']
        
        if type(loaded_obj) == OrderedDict:
            COMPILE = False
            if COMPILE:
                net = torch.compile(model_to_be_initialized)
                net.load_state_dict(loaded_obj)
            else:
                modified_dict = OrderedDict({key.replace('_orig_mod.', ''):val for key, val in loaded_obj.items()})
                net = model_to_be_initialized
                net.load_state_dict(modified_dict)
        else:
            # ensures backward compatibility for times where net is a model pkl file
            net = loaded_obj
        net = net.to(device)
        dist.print0(f'Network loaded!')

        #pdb.set_trace()

        image_dir = os.path.join(outdir, str(checkpoint_number) + "/" + cond_loc[-12:-4])
        os.makedirs(image_dir, exist_ok=True)

        cond = np.load(cond_loc)
        cond = torch.from_numpy(cond) 
        cond = cond.repeat(1,1,1,1).to((device))

        cond[0,0,0:16,:] = 0
       
        a = np.quantile(np.absolute(cond.cpu()),0.98)
        plt.figure(); plt.title("Condition")
        plt.imshow(cond[0,0,:,:].cpu(), vmin=-a,vmax=a, cmap = "gray")
        plt.axis("off")
        cb = plt.colorbar(fraction=0.0235, pad=0.04); 
        plt.savefig(os.path.join(image_dir, "actual_condition.png"),bbox_inches = "tight",dpi=300)

        gt = np.load(gt_loc) 
        vmin_gt = 1.5
        vmax_gt = 3.7#4.5
        cmap_gt = cc.cm['rainbow4']

        plt.figure();  plt.title("Ground truth")
        plt.imshow(gt, vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
        plt.axis("off")
        cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
        plt.savefig(os.path.join(image_dir, "original_velocity.png"),bbox_inches = "tight",dpi=300)

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
        batch_count = 1
        images_np_stack = np.zeros((len(seeds),1,*gt.shape))
        for batch_seeds in tqdm.tqdm(rank_batches, disable=dist.get_rank() != 0):
            dist.print0(f"Waiting for the green light to start generation for {batch_count}/{batches_per_process}")
            # don't move to the next batch until all nodes have finished their current batch
            torch.distributed.barrier()
            dist.print0("Others finished. Good to go!")
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, 1, gt.shape[0], gt.shape[1]], device=device)
           
            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            images = ambient_sampler(net, latents,num_steps=num_steps, randn_like=rnd.randn_like,
                cond=cond, image_dir=image_dir, **sampler_kwargs)
            
            # Save Images
            images_np = images.cpu().detach().numpy()
            for seed, one_image in zip(batch_seeds, images_np):
                dist.print0(f"Saving loc: {image_dir}")
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, "steps_"+str(num_steps)+"_"+f'{seed:04d}.png')

                plt.figure(); plt.title("Posterior Sample")
                plt.imshow(one_image[0, :, :],   vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
                plt.axis("off")
                cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
                plt.savefig(image_path, bbox_inches = "tight",dpi=300)
                plt.close()
                os.makedirs(os.path.join(image_dir, f'saved/'), exist_ok=True)
                np.save(os.path.join(image_dir, f'saved/{seed:06d}')+ ".npy", one_image[0, :, :])
            images_np_stack[batch_count-1,0,:,:] = one_image
            batch_count += 1

           # plot posterior statistics
        post_mean = np.mean(images_np_stack,axis=0)[0,:,:]
        ssim_t = ssim(gt,post_mean, data_range=np.max(gt) - np.min(gt))

        plt.figure(); plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
        plt.imshow(post_mean,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt)
        plt.axis("off"); 
        cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_mean.png"),bbox_inches = "tight",dpi=300); plt.close()

        plt.figure(); plt.title("Stdev")
        plt.imshow(np.std(images_np_stack,axis=0)[0,:,:],  vmin=0, vmax=None,   cmap = "magma")
        plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"std.png"),bbox_inches = "tight",dpi=300); plt.close()
            
        rmse_t = np.sqrt(mean_squared_error(gt, post_mean))
        plt.figure(); plt.title("Error RMSE:"+str(round(rmse_t,4)))
        plt.imshow(np.abs(post_mean-gt), vmin=0, vmax=None, cmap = "magma")
        plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_error.png"),bbox_inches = "tight",dpi=300); plt.close()

        dist.print0(f"Node finished generation for {checkpoint_number}")
        dist.print0("waiting for others to finish..")

        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        dist.print0("Everyone finished.. Starting calculation..")
    
if __name__ == "__main__":
   
    seeds = [i for i in range(0, 100)]
    max_batch_size = 1
    num_generate = 2
    num_steps = 10

    device = torch.device('cuda')
    #device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_loc', type=str, default="")
    parser.add_argument('--network_loc', type=str, default="")
    parser.add_argument('--gt_loc', type=str, default="")

    args = parser.parse_args()
    cond_loc = args.cond_loc
    vel_loc = args.gt_loc
    network_loc = args.network_loc

    training_options_loc = network_loc+"/training_options.json"
    outdir = "sampling/"

    main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc, vel_loc, device)
    