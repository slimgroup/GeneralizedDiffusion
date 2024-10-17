#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import os
import matplotlib.pyplot as plt
import dnnlib
import numpy as np
import colorcet as cc

plt.rcParams["font.family"] = "serif"

#plot plosterior samples and statistics

cmap_gt = cc.cm['rainbow4']
image_dir = "sampling/final_plots/galactic/"

# #i_str ="0981"
# #i_str ="0873"
# #i_str ="0988"
# #i_str ="0922"
# i_str ="0913"

#gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_syntho_ext_test/gt_"+i_str+".npy")
# gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/gt0s_syntho_ext_test/gt0_"+i_str+".npy")
# rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/rtms_syntho_ext_test/rtm_"+i_str+".npy")[12,:,:]

# vmin_gt = 1.5
# vmax_gt = np.max(gt)



# plt.figure(figsize=(7,3));  
# plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir+"gt_"+i_str+".png"), bbox_inches = "tight", dpi=300)


# plt.figure(figsize=(7,3)); 
# plt.imshow(gt0, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir+"gt0_"+i_str+".png"), bbox_inches = "tight", dpi=300)

# a = np.quantile(np.absolute(rtm),0.95)
# plt.figure(figsize=(7,3));  
# plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir+"rtm_"+i_str+".png"), bbox_inches = "tight", dpi=300)


net_name = "synth"
#net_name = "newloss_cont-offsetsTrue751"
#path = "sampling/00141-gpus2-batch10-synth_ext-offsetsFalse720/rtm_"+i_str+"/saved/"
path = "/slimdata/rafaeldata/seam_rtm_diffusion/sampling/galactic_back_1001/wi_final/saved/"

files_rtm = dnnlib.util.list_dir(path)

first = np.load(path+"000000.npy")

num_post_samples = len(files_rtm)  # Assuming num_expected is defined
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1



velocity = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_galactic/gt_xwi_final.npy")#[:,0:size_gen]
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/rtms_galactic/rtm_xwi_final.npy")#[:,0:size_gen]
vmin_gt = np.min(velocity)#1.5
vmax_gt = np.max(velocity)#3.7#4.5
      


d = 0.0125
extent = (0,d*velocity.shape[1],d*velocity.shape[0],0)

plt.figure(figsize=(15,6));  
plt.imshow(velocity, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=6, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"gt0_.png"), bbox_inches = "tight", dpi=300)


a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(15,6));  
plt.imshow(-rtm, cmap="gray", vmin=-a, vmax=a, aspect=6, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"rtm_.png"), bbox_inches = "tight", dpi=300)

post_mean = np.mean(images_np_stack,axis=0)[0,:,:]

plt.figure(figsize=(15,6)); #plt.title("Posterior mean")
plt.imshow(post_mean,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,aspect=6, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir,"_p"+str(num_post_samples)+"wnext_mean.png"),bbox_inches = "tight",dpi=300); plt.close()


plt.figure(figsize=(15,6)); #plt.title("Uncertainty")
plt.imshow(np.std(images_np_stack,axis=0)[0,:,:],  vmin=0, vmax=0.15,  cmap = "magma",aspect=6, extent=extent)
#cb = plt.colorbar(fraction=0.0235, pad=0.01); cb.set_label('[Km/s]')
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"wnextstd.png"),bbox_inches = "tight",dpi=300); plt.close()






well = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_galactic/well_data_interp_norm.npy")
trace_ind = 475 

first_ind = 150
up_to = 350
range_depth = [i*d for i in range(first_ind,up_to)]

plt.figure(figsize=(7,3)); #plt.title("Trace at well location")
for i in range(1,num_post_samples):
    plt.plot(range_depth,images_np_stack[i,0,first_ind:up_to,trace_ind], linewidth=0.4, alpha=0.2, color="red")

plt.plot(range_depth,images_np_stack[0,0,first_ind:up_to,trace_ind], linewidth=0.8, color="red",  alpha=0.5,label="Posterior sample")
plt.plot([i*d for i in range(178,343)] ,well[178:343], linewidth=0.8,color="black",  label="Borehole Well")
#plt.plot(images_np_stack[2,0,:,trace_ind], linewidth=0.8,color="red", linestyle="--", label="Posterior sample")
#plt.plot(post_mean[:,trace_ind], linewidth=0.8,color="black", linestyle="--", label="Posterior mean")
#plt.plot(velocity[:,trace_ind], linewidth=0.8,color="black", label="Migration Velocity")
#plt.set_ylim(0,512)
plt.ylabel("Km/s")
plt.xlabel("Depth [Km]")
plt.legend()
plt.savefig(os.path.join(image_dir, "_trace.png"),bbox_inches = "tight",dpi=300); plt.close()





plt.figure(figsize=(3,7)); #plt.title("Trace at well location")
for i in range(1,num_post_samples):
    plt.plot(images_np_stack[i,0,first_ind:up_to,trace_ind],range_depth, linewidth=0.4, alpha=0.2, color="red")

plt.plot(images_np_stack[0,0,first_ind:up_to,trace_ind],range_depth, linewidth=0.8, color="red",  alpha=0.5,label="Posterior sample")
plt.plot(well[178:343],range(178,343), linewidth=0.8,color="black",  label="Borehole Well")
#plt.plot(images_np_stack[2,0,:,trace_ind], linewidth=0.8,color="red", linestyle="--", label="Posterior sample")
#plt.plot(post_mean[:,trace_ind], linewidth=0.8,color="black", linestyle="--", label="Posterior mean")
#plt.plot(velocity[:,trace_ind], linewidth=0.8,color="black", label="Migration Velocity")
plt.set_ylim(512,0)
plt.gca().invert_yaxis()
plt.xlabel("Km/s")
plt.ylabel("Depth [grid point]")
plt.legend()
plt.savefig(os.path.join(image_dir, "_trace.png"),bbox_inches = "tight",dpi=300); plt.close()



# plt.figure(figsize=(7,3)); 
# plt.imshow(post_mean, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir, net_name+"_p"+str(num_post_samples)+"_mean_"+i_str+".png"),bbox_inches = "tight",dpi=300); plt.close()


# plt.figure(figsize=(7,3)); 
# plt.imshow(images_np_stack[0,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir, net_name+"_p"+str(num_post_samples)+"_post_"+i_str+".png"),bbox_inches = "tight",dpi=300); plt.close()

# #cmap_error = cc.cm['CET_L3']
# cmap_error = "magma"


# post_std = np.std(images_np_stack,axis=0)[0,:,:]
# plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
# plt.imshow(post_std,  vmin=0, vmax=0.5,   cmap = cmap_error, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
# #plt.colorbar(fraction=0.0235, pad=0.04)
# plt.savefig(os.path.join(image_dir, net_name+"_p"+str(num_post_samples)+"std_"+i_str+".png"),bbox_inches = "tight",dpi=300); plt.close()
#  