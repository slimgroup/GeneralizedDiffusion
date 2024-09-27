import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from torch_utils import distributed as dist
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
    cond=None,
    gt_norm=1
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
            net_input = torch.cat([x_hat, cond], dim=1)
            denoised = net(net_input, t_hat).to(torch.float64)[:, :1]
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                net_input = torch.cat([x_next, cond], dim=1)
                denoised = net(net_input, t_next).to(torch.float64)[:, :1]
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return gt_norm*x_next

def main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_base, back_base, gt_base, gt_norm, cond_norm,use_offsets,out_chan, device=torch.device('cuda'),  **sampler_kwargs):

    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    seeds = seeds[:num_generate]
    num_batches = ((len(seeds) - 1) // (max_batch_size * 1) + 1) *1
    print(num_batches)
    #dist.print0(f"The algorithm will run for {num_batches} batches --  {len(seeds)} images of batch size {max_batch_size}")
    rank_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # the following has for each batch size allocated to this GPU, the indexes of the corresponding images.

    # load training options
    with dnnlib.util.open_url(training_options_loc, verbose=(True)) as f:
        training_options = json.load(f)

    label_dim = 0

    #load in condition
    files_cond = dnnlib.util.list_dir(cond_base)
    cond_loc = cond_base+files_cond[0]

    cond = np.load(cond_loc) / cond_norm
    print(use_offsets)
    if not use_offsets:
        print("use only zero offset")
        cond = cond[12,:,:]
        #cond = cond[np.newaxis,...]

    cond = torch.from_numpy(cond) 
    cond = cond.repeat(1,1,1,1).to((device))
    print(cond.shape)



    interface_kwargs = dict(img_resolution=cond.shape[2], label_dim=0, img_channels=cond.shape[1]+2, out_channels=out_chan)
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

        network_pkl = os.path.join(network_loc, f'network-snapshot-{checkpoint_number:06d}.pkl')
        # Load network.
        #dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=True) as f:
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
        #dist.print0(f'Network loaded!')

        ###loop here 
        files_cond = dnnlib.util.list_dir(cond_base)

        ssims = []
        rmses = []
        print(files_cond[0::4])
        for i_str in files_cond[0::4]:

            cond_loc = cond_base+i_str
            gt_loc = gt_base+"gt_"+i_str[-8:]
            back_loc = back_base+"gt0_"+i_str[-8:]
            print(cond_loc)
            print(gt_loc)

            cond = np.load(cond_loc) / cond_norm
            if not use_offsets:
                print("use only zero offset")
                cond = cond[12,:,:]
                #cond = cond[np.newaxis,...]

            cond = torch.from_numpy(cond) 
            cond = cond.repeat(1,1,1,1).to((device))

            background = np.load(back_loc)
            background = torch.from_numpy(background)  / gt_norm
            background = background.repeat(1,1,1,1).to((device))

            cond = torch.cat([cond, background], axis=1)

        #pdb.set_trace()

            image_dir = os.path.join(outdir, str(network_loc.split("/")[1]) + str(checkpoint_number) + "/" + cond_loc[-12:-4])
            os.makedirs(image_dir, exist_ok=True)
            
           
            a = np.quantile(np.absolute(cond.cpu()),0.98)
            plt.figure(); plt.title("Condition rtm")
            plt.imshow(cond[0,0,:,:].cpu(), vmin=-a,vmax=a, cmap = "gray")
            plt.axis("off")
            cb = plt.colorbar(fraction=0.0235, pad=0.04); 
            plt.savefig(os.path.join(image_dir, "rtm_condition.png"),bbox_inches = "tight",dpi=300)

            gt = np.load(gt_loc) 
            vmin_gt = 1.5
            vmax_gt = np.max(gt)
            cmap_gt = cc.cm['rainbow4']

            plt.figure(); plt.title("Condition back")
            plt.imshow(gt_norm*cond[0,-1,:,:].cpu(), vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
            plt.axis("off")
            cb = plt.colorbar(fraction=0.0235, pad=0.04); 
            plt.savefig(os.path.join(image_dir, "back_condition.png"),bbox_inches = "tight",dpi=300)

            plt.figure();  plt.title("Ground truth")
            plt.imshow(gt, vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
            plt.axis("off")
            cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
            plt.savefig(os.path.join(image_dir, "original_velocity.png"),bbox_inches = "tight",dpi=300)


            # Loop over batches.
            #dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
            batch_count = 1
            images_np_stack = np.zeros((len(seeds),1,*gt.shape))
            for batch_seeds in tqdm.tqdm(rank_batches):
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_seeds)
                latents = rnd.randn([batch_size, 1, gt.shape[0], gt.shape[1]], device=device)
               
                # Generate images.
                sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                images = ambient_sampler(net, latents,num_steps=num_steps, randn_like=rnd.randn_like,
                    cond=cond, image_dir=image_dir,gt_norm=gt_norm, **sampler_kwargs)
                
                # Save Images
                images_np = images.cpu().detach().numpy()
                for seed, one_image in zip(batch_seeds, images_np):
                    #dist.print0(f"Saving loc: {image_dir}")
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
            plt.imshow(np.std(images_np_stack,axis=0)[0,:,:],  vmin=0, vmax=0.5,   cmap = "magma")
            plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
            plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"std.png"),bbox_inches = "tight",dpi=300); plt.close()
                
            rmse_t = np.sqrt(mean_squared_error(gt, post_mean))
            plt.figure(); plt.title("Error RMSE:"+str(round(rmse_t,4)))
            plt.imshow(np.abs(post_mean-gt), vmin=0, vmax=0.5, cmap = "magma")
            plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
            plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_error.png"),bbox_inches = "tight",dpi=300); plt.close()

            #dist.print0(f"Node finished generation for {checkpoint_number}")

            #dist.print0("Everyone finished.. Starting calculation..")
            ssims.append(ssim_t)
            rmses.append(rmse_t)

            print("SSIM:"+str(ssim_t))
            print("rmses:"+str(rmse_t))
            
            np.save(os.path.join(outdir,"metrics/",str(network_loc.split("/")[1]) +str(checkpoint_number) + f'{seed:06d}')+ "_ssims.npy", ssims)
            np.save(os.path.join(outdir,"metrics/",str(network_loc.split("/")[1]) +str(checkpoint_number) + f'{seed:06d}')+ "_rmses.npy", rmses)

        np.save(os.path.join(outdir,"metrics/",str(network_loc.split("/")[1]) +str(checkpoint_number) + f'{seed:06d}')+ "_ssims.npy", ssims)
        np.save(os.path.join(outdir,"metrics/",str(network_loc.split("/")[1]) +str(checkpoint_number) + f'{seed:06d}')+ "_rmses.npy", rmses)
    
if __name__ == "__main__":
   
    seeds = [i for i in range(0, 100)]
    max_batch_size = 1
    num_generate = 16
    num_steps = 10

    device = torch.device('cuda')
    #device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_loc', type=str, default="")
    parser.add_argument('--back_loc', type=str, default="")
    parser.add_argument('--network_loc', type=str, default="")
    parser.add_argument('--gt_loc', type=str, default="")
    parser.add_argument('--gt_norm', type=float, default=1.0)
    parser.add_argument('--cond_norm', type=float, default=1.0)
    parser.add_argument('--out_chan', type=int, default=1)
    parser.add_argument('--use_offsets', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    cond_loc = args.cond_loc
    back_loc = args.back_loc
    vel_loc = args.gt_loc
    network_loc = args.network_loc
    gt_norm = args.gt_norm
    cond_norm = args.cond_norm
    use_offsets = args.use_offsets
    out_chan = args.out_chan
    print(use_offsets)

    training_options_loc = network_loc+"/training_options.json"
    outdir = "sampling/"

    main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc,back_loc, vel_loc, gt_norm, cond_norm, use_offsets,out_chan,device)
    