#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import os
import matplotlib.pyplot as plt
import dnnlib
import numpy as np
import colorcet as cc

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error




def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    #device = errors.device
    if range == None:
        bin_boundaries = np.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1)
    else:
        bin_boundaries = np.linspace(range[0], range[1], n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []
    uce = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |uncertainty - error| in each bin
        in_bin = (uncert > (bin_lower.item())) * (uncert < (bin_upper.item()))
        prop_in_bin = in_bin.mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].mean()  # err()
            avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
            uce += np.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin
            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)
    err_in_bin = errors_in_bin_list
    avg_uncert_in_bin = avg_uncert_in_bin_list
    prop_in_bin = prop_in_bin_list
    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin


plt.rcParams["font.family"] = "serif"


vmin_gt = 1.5
vmax_gt = 4.75
cmap_gt = cc.cm['rainbow4']


#for i_str in ["0008"]:
i_str = "0004"
gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gts_seam_filter_test/gt_"+i_str+".npy")
gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gt0s_seam_filter_test/gt0_"+i_str+".npy")
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_filter_test/rtm_"+i_str+".npy")

d = 0.02
extent = (0,d*gt.shape[1],d*gt.shape[0],0)


plt.figure(figsize=(12,5));  
plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, extent=extent,aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt_seam.png"), bbox_inches = "tight", dpi=300)

plt.figure(figsize=(12,5));  
plt.imshow(gt0, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt0_seam.png"), bbox_inches = "tight", dpi=300)
a = np.quantile(np.absolute(rtm),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_0.png"), bbox_inches = "tight", dpi=300)


rtm1 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v1_0005.npy")
a = np.quantile(np.absolute(rtm1),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm1, cmap="seismic", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_1_seismic.png"), bbox_inches = "tight", dpi=300)




rtmgt = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_gt_0005.npy")
a = np.quantile(np.absolute(rtmgt),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtmgt, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_gt_noaxis.png"), bbox_inches = "tight", dpi=300)




rtm2 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_0005.npy")
a = np.quantile(np.absolute(rtm2),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm2, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_noaxis.png"), bbox_inches = "tight", dpi=300)




image_dir = "sampling/final_plots/seam/"
#path = "sampling/120/rtm_0001/saved/"

path = "sampling/00179-gpus2-batch10-seam_filter-offsetsFalse210/rtm_"+i_str+"/saved/"
files_rtm = dnnlib.util.list_dir(path)


first = np.load(path+"000000.npy")
num_post_samples = 16  # Assuming num_expected is defined
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1


post_mean_1 = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t = ssim(gt,post_mean_1, data_range=np.max(gt) - np.min(gt))


#plot some posterior statistics
plt.figure(figsize=(12,5));  # plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
plt.imshow(post_mean_1,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_mean_1.png"),bbox_inches = "tight",dpi=300); plt.close()


#import colorcet as cc;cmap_error = cc.cm['CET_L3']
cmap_error = "magma"
post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(12,5));   #plt.title("Posterior deviation")
plt.imshow(2*post_std,  vmin=0, vmax=0.5,   cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_std_1.png"),bbox_inches = "tight",dpi=300); plt.close()
    

rmse_t = np.sqrt(mean_squared_error(gt, post_mean_1))
print(rmse_t)
print(ssim_t)
post_error = np.abs(post_mean_1-gt)
plt.figure(figsize=(12,5)); #plt.title("Error RMSE:"+str(round(rmse_t,4)))
plt.imshow(post_error, vmin=0, vmax=0.5, cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_error_1.png"),bbox_inches = "tight",dpi=300); plt.close()


cmap_error_gray = cc.cm['CET_L1']
cmap_error_gray.set_over('red')
threshold = 2

support = post_error / (post_std+1e-1)
perc_256 = np.mean((support) > threshold)*100

# >>> perc_256
# 27.57110595703125
# >>> perc_256
# 25.742949039564223

plt.figure(figsize=(12,5));    #plt.title("Posterior deviation")
plt.imshow(support,  vmin=0, vmax=threshold,   cmap = cmap_error_gray,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]");   #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+i_str+"_seam_bouman_1.png"),bbox_inches = "tight",dpi=300); plt.close()
  


for i in range(1,10):
    plt.figure(figsize=(12,5));  
    plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1)
    plt.ylabel("Z [grid]"); plt.xlabel("X [grid]"); 
    #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
    plt.savefig(os.path.join(image_dir, "posterior_"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/posterior_*.png sampling/final_plots/seam/samples_movie.gif
# echo "Experiment done!"

range_km = [d*i for i in range(0,512)]
trace_ind = 625 
plt.figure(figsize=(8,3)); #plt.title("Vertical trace at X grid point "+str(trace_ind))
for i in range(1,num_post_samples):
    plt.plot(range_km,images_np_stack[i,0,:,trace_ind], linewidth=0.4, alpha=0.3, color="red")

plt.plot(range_km,images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(range_km,gt[:,trace_ind], linewidth=0.8,color="black", label="Ground truth ")
#plt.ylim(1.2,to5.5)
plt.ylabel("Velocity [Km/s]")
#plt.xlabel("Depth [grid point]")
plt.xlabel("Depth [Km]")
plt.legend()
plt.savefig(os.path.join(image_dir,"_p"+str(num_post_samples)+"_traces_seam_1.png"),bbox_inches = "tight",dpi=300); plt.close()


range_depth = range(512)
trace_ind = 256 
plt.figure(figsize=(3,7)); plt.title("Vertical trace at X grid point "+str(trace_ind))
for i in range(1,num_post_samples):
    plt.plot(images_np_stack[i,0,:,trace_ind],range_depth, linewidth=0.4, alpha=0.3, color="red")

plt.plot(images_np_stack[15,0,:,trace_ind],range_depth, linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(gt[:,trace_ind],range_depth, linewidth=0.8,color="black", label="Ground truth ")
#plt.ylim(1.2,to5.5)
plt.set_ylim(512,0)
plt.gca().invert_yaxis()
plt.xlabel("Velocity [Km/s]")
plt.ylabel("Depth [grid point]")
plt.legend()
plt.savefig(os.path.join("sampling/final_plots/seam/_p"+str(num_post_samples)+"trace_vert.png"),bbox_inches = "tight",dpi=300); plt.close()




###################################




i_str = "0005"
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_filter_2_test/rtm_"+i_str+".npy")
gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gt0s_seam_filter_2_test/gt0_"+i_str+".npy")


plt.figure(figsize=(12,5));  
plt.imshow(gt0, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt0_seam_1.png"), bbox_inches = "tight", dpi=300)

a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_1.png"), bbox_inches = "tight", dpi=300)


dataset_name = "seam"

i_str = "0004"
#path = "sampling/00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue570/rtm_"+i_str+"/saved/"
net_name_path = "00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue570"
#net_name_path = "00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue300"
net_name = net_name_path[-3:]+net_name_path[-7:-3]

path = "sampling/"+net_name_path+"/rtm_"+i_str+"/saved/"

files_rtm = dnnlib.util.list_dir(path)


first = np.load(path+"000000.npy")
num_post_samples = len(files_rtm)  # Assuming num_expected is defined
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1


post_mean_2 = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t = ssim(gt,post_mean_2, data_range=np.max(gt) - np.min(gt))


#plot some posterior statistics
plt.figure(figsize=(12,5));  # plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
plt.imshow(post_mean_2,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_mean_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


#import colorcet as cc;cmap_error = cc.cm['CET_L3']
cmap_error = "magma"
post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(12,5));   #plt.title("Posterior deviation")
plt.imshow(2*post_std,  vmin=0, vmax=0.5,   cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_std_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()
   

rmse_t = np.sqrt(mean_squared_error(gt, post_mean_2))
rmsstd = np.sqrt(np.mean(post_std**2))

post_error = np.abs(post_mean_2-gt)
plt.figure(figsize=(12,5)); #plt.title("Error RMSE:"+str(round(rmse_t,4)))
plt.imshow(post_error, vmin=0, vmax=0.5, cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_error_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()

##################################################################
threshold = 2
support = post_error / (2*post_std+1e-1)
perc_256 = np.mean((support) > threshold)*100

cmap_error_gray = cc.cm['CET_L1']
cmap_error_gray.set_over('red')

plt.figure(figsize=(12,5));    #plt.title("Posterior deviation")
plt.imshow(support,  vmin=0, vmax=threshold,   cmap = cmap_error_gray,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]");   #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_bouman_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()
  
#################################################################
lower_percentile=1
upper_percentile=99
lower_bound = np.percentile(images_np_stack[:,0,:,:], lower_percentile, axis=0)
upper_bound = np.percentile(images_np_stack[:,0,:,:], upper_percentile, axis=0)
# Create a mask where the ground truth is within the credible interval
coverage_mask = (gt >= lower_bound) & (gt <= upper_bound)
# Calculate the coverage as the percentage of pixels inside the credible interval
coverage = np.mean(coverage_mask) * 100 


# for i in range(1,10)
#     plt.figure(figsize=(12,5));  
#     plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1)
#     plt.ylabel("Z [grid]"); plt.xlabel("X [grid]"); 
#     #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
#     plt.savefig(os.path.join(image_dir, "posterior_2"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/posterior_*.png sampling/final_plots/seam/samples_movie.gif
# echo "Experiment done!"

range_km = [d*i for i in range(0,512)]
trace_ind = 625 
plt.figure(figsize=(8,3)); #plt.title("Vertical trace at X grid point "+str(trace_ind))
for i in range(1,num_post_samples):
    plt.plot(range_km,images_np_stack[i,0,:,trace_ind], linewidth=0.4, alpha=0.3, color="red")

plt.plot(range_km,images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(range_km,gt[:,trace_ind], linewidth=0.8,color="black", label="Ground truth ")
#plt.ylim(1.2,to5.5)
plt.ylabel("Velocity [Km/s]")
#plt.xlabel("Depth [grid point]")
plt.xlabel("Depth [Km]")
plt.legend()
plt.savefig(os.path.join(image_dir,str(num_post_samples)+i_str+net_name+"_traces_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


uce, err_in_bin, avg_uncert_in_bin, prop_in_bin= uceloss(post_error, post_std, n_bins=20, outlier=0.0, range=None)


fig, ax  = plt.subplots(1, 1, figsize=(4, 4))
#ax.plot([0, 0], [1, 1], 'k--')
#plt.plot([0, 0], [1, 1], 'k--',color="black")
ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle="--",color="black",label="Perfect calibration")
plt.plot(avg_uncert_in_bin,err_in_bin,color="red",label="UCE="+str(round(uce[0],4)))
plt.xlim(0,2); plt.ylim(0,2);
plt.ylabel("Error [Km/s]")
plt.xlabel("Uncertainty [Km/s]")
ax.set_aspect(1)
plt.legend()
plt.savefig(os.path.join(image_dir,str(num_post_samples)+i_str+net_name+"_calibration_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


print(net_name)
print(uce)
print(coverage)
print(perc_256)
print(rmse_t)
print(ssim_t)
print(rmsstd)




# >>> print(net_name)
# 570True
# >>> print(uce)
# [0.06110951]
# >>> print(coverage)
# 38.73794975630734
# >>> print(perc_256)
# 3.313481042144496
# >>> print(rmse_t)
# 0.22695310685632766
# >>> print(ssim_t)
# 0.6822860075856408
# >>> print(rmsstd)
# 0.11688961814702988

# >>> print(net_name)
# 570True
# >>> print(uce)
# [0.05571461]
# >>> print(coverage)
# 54.395427178899084
# >>> print(perc_256)
# 2.94547824684633
# >>> print(rmse_t)
# 0.22539922559582148
# >>> print(ssim_t)
# 0.6876778067152624
# >>> print(rmsstd)
# 0.1208050851316578

# >>> print(net_name)
# 300True
# >>> print(uce)
# [0.04459107]
# >>> print(coverage)
# 60.87276913704128
# >>> print(perc_256)
# 1.9987053827408259
# >>> print(rmse_t)
# 0.23276157420462226
# >>> print(ssim_t)
# 0.689757217537163
# >>> print(rmsstd)
# 0.1406542638361207




###############################


i_str = "0005"
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_filter_shift_prevstack_3_test/rtm_"+i_str+".npy")
#gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gt0s_seam_filter_3_test/gt0_"+i_str+".npy")

a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_2.png"), bbox_inches = "tight", dpi=300)







i_str = "0005"
path = "sampling/00185-gpus2-batch10-seam_filter_3-offsetsFalse480/rtm_"+i_str+"/saved/"
files_rtm = dnnlib.util.list_dir(path)


first = np.load(path+"000000.npy")
num_post_samples = len(files_rtm)#16  # Assuming num_expected is defined
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1


post_mean_3 = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t_3 = ssim(gt,post_mean_3, data_range=np.max(gt) - np.min(gt))


#plot some posterior statistics
plt.figure(figsize=(12,5));  # plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
plt.imshow(post_mean_3,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_mean_3.png"),bbox_inches = "tight",dpi=300); plt.close()


#import colorcet as cc;cmap_error = cc.cm['CET_L3']
cmap_error = "magma"
post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(12,5));   #plt.title("Posterior deviation")
plt.imshow(2*post_std,  vmin=0, vmax=0.5,   cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_std_3.png"),bbox_inches = "tight",dpi=300); plt.close()
    

rmse_t = np.sqrt(mean_squared_error(gt, post_mean_2))
print(rmse_t)
print(ssim_t)
post_error = np.abs(post_mean_3-gt)
plt.figure(figsize=(12,5)); #plt.title("Error RMSE:"+str(round(rmse_t,4)))
plt.imshow(post_error, vmin=0, vmax=0.5, cmap = cmap_error,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+"_seam_error_3.png"),bbox_inches = "tight",dpi=300); plt.close()


cmap_error_gray = cc.cm['CET_L1']
cmap_error_gray.set_over('red')
threshold = 2

support = post_error / (post_std+1e-1)
perc_256 = np.mean((support) > threshold)*100


# >>> perc_256
# 25.742949039564223

# >>> perc_256
# 5.845488102064221

# >>> perc_256
# 9.70145409260321

plt.figure(figsize=(12,5));    #plt.title("Posterior deviation")
plt.imshow(support,  vmin=0, vmax=threshold,   cmap = cmap_error_gray,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]");   #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+i_str+"_seam_bouman_3.png"),bbox_inches = "tight",dpi=300); plt.close()
  


range_km = [d*i for i in range(0,512)]
trace_ind = 625 
plt.figure(figsize=(8,3)); #plt.title("Vertical trace at X grid point "+str(trace_ind))
for i in range(1,num_post_samples):
    plt.plot(range_km,images_np_stack[i,0,:,trace_ind], linewidth=0.4, alpha=0.3, color="red")

plt.plot(range_km,images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(range_km,gt[:,trace_ind], linewidth=0.8,color="black", label="Ground truth ")
#plt.ylim(1.2,to5.5)
plt.ylabel("Velocity [Km/s]")
#plt.xlabel("Depth [grid point]")
plt.xlabel("Depth [Km]")
plt.legend()
plt.savefig(os.path.join(image_dir,"_p"+str(num_post_samples)+"_traces_seam_3.png"),bbox_inches = "tight",dpi=300); plt.close()
