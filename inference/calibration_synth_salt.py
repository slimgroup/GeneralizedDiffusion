#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 
import os
import matplotlib.pyplot as plt
import dnnlib
import numpy as np
import colorcet as cc


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
cmap_gt = cc.cm['rainbow4']

image_dir = "sampling/final_plots/synthoseis_salt/"

#i_str ="0981"
#i_str ="0873"
#i_str ="0988"
#i_str ="0922"
#i_str ="0850"
i_str ="0841"
d = 0.0125
gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_syntho_salt_test/gt_"+i_str+".npy")
gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/gt0s_syntho_salt_test/gt0_"+i_str+".npy")
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/rtms_syntho_salt_test/rtm_"+i_str+".npy")[:,:]

vmin_gt = 1.5
vmax_gt = 4#np.max(gt)

extent = (0,d*gt.shape[1],d*gt.shape[0],0)

plt.figure(figsize=(7,3));  
plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"gt_"+i_str+".png"), bbox_inches = "tight", dpi=300)


a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(7,3));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"rtm_"+i_str+".png"), bbox_inches = "tight", dpi=300)



plt.figure(figsize=(7,3));  
plt.imshow(gt0, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"gt0_"+i_str+".png"), bbox_inches = "tight", dpi=300)




dataset_name = "synth_salt"

#net_name = "00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691"
#net_name = "00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451"
#net_name = "00159-gpus2-batch10-synth_salt_badback-offsetsTrue240"
#net_name = "00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue631"
net_name = "00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse931"
path = "sampling/"+net_name+"/rtm_"+i_str+"/saved/"

files_rtm = dnnlib.util.list_dir(path)

first = np.load(path+"000000.npy")

num_post_samples = len(files_rtm)  # Assuming num_expected is defined
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1



from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

post_mean = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t = ssim(gt,post_mean, data_range=np.max(gt) - np.min(gt))


plt.figure(figsize=(7,3)); 
plt.imshow(post_mean, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_mean_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()



plt.figure(figsize=(7,3)); 
plt.imshow(images_np_stack[1,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_post"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()

# plt.figure(figsize=(7,3)); 
# plt.imshow(images_np_stack[0,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir, net_name+"_p"+str(num_post_samples)+i_str+"_post_compass.png"),bbox_inches = "tight",dpi=300); plt.close()

#cmap_error = cc.cm['CET_L3']
cmap_error = "magma"

post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
plt.imshow(2*post_std,  vmin=0, vmax=0.5,   cmap = cmap_error, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_std_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()
 

#import matplotlib.colors as mcolors

#colors = [(1.0, 0.0, 0.0, 1.0)]  # Red for above the threshold
#cmap_over_threshold = mcolors.ListedColormap(colors)

# Set the colormap to show the original cmap1 and red for values above the threshold


rmse_t = np.sqrt(mean_squared_error(gt, post_mean))
rmsstd = np.sqrt(np.mean(post_std**2))

post_error = np.abs(post_mean-gt)
plt.figure(figsize=(7,3));  #plt.title("Error RMSE:"+str(round(rmse_t,4)))
plt.imshow(post_error, vmin=0, vmax=0.5, cmap = cmap_error, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_error_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


cmap_error_gray = cc.cm['CET_L1']
cmap_error_gray.set_over('red')
threshold = 2
support = post_error / (post_std+1e-2)
perc_256 = np.mean((support) > threshold)*100

# >>> print(perc_256)
# 20.70159912109375

plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
plt.imshow(support,  vmin=0, vmax=threshold,   cmap = cmap_error_gray, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
plt.colorbar(fraction=0.0235, pad=0.01,extend="max")

plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_bouman_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()
  

#lower_percentile=2.5
#upper_percentile=97.5
lower_percentile=1
upper_percentile=99
lower_bound = np.percentile(images_np_stack[:,0,:,:], lower_percentile, axis=0)
upper_bound = np.percentile(images_np_stack[:,0,:,:], upper_percentile, axis=0)
# Create a mask where the ground truth is within the credible interval
coverage_mask = (gt >= lower_bound) & (gt <= upper_bound)
# Calculate the coverage as the percentage of pixels inside the credible interval
coverage = np.mean(coverage_mask) * 100  # percentage

# plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
# plt.imshow(coverage_mask,  extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
# #plt.colorbar(fraction=0.0235, pad=0.04)
# plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_coverage_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()
  
# for i in range(1,10):
#     plt.figure(figsize=(12,5));  
#     plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1)
#     plt.ylabel("Z [grid]"); plt.xlabel("X [grid]"); 
#     #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
#     plt.savefig(os.path.join(image_dir, net_name+"posterior_"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/posterior_*.png sampling/final_plots/seam/samples_movie.gif
# echo "Experiment done!"
range_depth = [d * i for i in range(0, 256)]
#trace_ind = 256 
trace_ind = 295 
plt.figure(figsize=(7,3)); #plt.title("Vertical trace at X grid point "+str(trace_ind))
for i in range(1,num_post_samples):
    plt.plot(range_depth,images_np_stack[i,0,:,trace_ind], linewidth=0.4, alpha=0.3, color="red")

plt.plot(range_depth,images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(range_depth,gt[:,trace_ind], linewidth=0.8,color="black", label="Ground truth ")
#plt.plot(range_depth,lower_bound[:,trace_ind], linewidth=0.8,color="red",linestyle="--", label="Lower bound ")
#plt.plot(range_depth,upper_bound[:,trace_ind], linewidth=0.8,color="red",linestyle="--", label="Upper bound ")
#plt.ylim(1.2,to5.5)
plt.ylabel("Velocity [Km/s]")
plt.xlabel("Depth [Km]")
plt.legend()
#plt.tight_layout()
plt.savefig(os.path.join(image_dir,str(num_post_samples)+i_str+net_name+"_traces_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()




uce, err_in_bin, avg_uncert_in_bin, prop_in_bin= uceloss(post_error, post_std, n_bins=20, outlier=0.0, range=None)


fig, ax  = plt.subplots(1, 1, figsize=(4, 4))
#ax.plot([0, 0], [1, 1], 'k--')
#plt.plot([0, 0], [1, 1], 'k--',color="black")
ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle="--",color="black",label="Perfect calibration")
plt.plot(avg_uncert_in_bin,err_in_bin,color="red",label="UCE="+str(round(uce[0],4)))
plt.xlim(0,1.5); plt.ylim(0,1.5);
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
# 00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse931
# >>> print(uce)
# [0.01394518]
# >>> print(coverage)
# 87.06741333007812
# >>> print(perc_256)
# 7.36083984375
# >>> print(rmse_t)
# 0.14501372169488183
# >>> print(ssim_t)
# 0.8490099627693226
# >>> print(rmsstd)
# 0.09373241479102835


# >>> print(net_name)
# 00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue631
# >>> print(uce)
# [0.0058881]
# >>> print(coverage)
# 89.73388671875
# >>> print(perc_256)
# 4.485321044921875
# >>> print(rmse_t)
# 0.10632470965215107
# >>> print(ssim_t)
# 0.887419331444549
# >>> print(rmsstd)
# 0.09379222316209151

# >>> print(net_name)
# 00159-gpus2-batch10-synth_salt_badback-offsetsTrue240
# >>> print(uce)
# [0.01012125]
# >>> print(coverage)
# 78.10134887695312
# >>> print(perc_256)
# 4.29229736328125
# >>> print(rmse_t)
# 0.1380776846294164
# >>> print(ssim_t)
# 0.8380522177264812
# >>> print(rmsstd)
# 0.12927269531057586

# >>> print(net_name)
# 00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451
# >>> print(uce)
# [0.00574499]
# >>> print(coverage)
# 76.10015869140625
# >>> print(perc_256)
# 4.43115234375
# >>> print(rmse_t)
# 0.11306092858669693
# >>> print(ssim_t)
# 0.8724134831636494
# >>> print(rmsstd)
# 0.10056742887848463

# >>> print(net_name)
# 00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691
# >>> print(uce)
# [0.00861486]
# >>> print(coverage)
# 78.61099243164062
# >>> print(perc_256)
# 6.014251708984375
# >>> print(rmse_t)
# 0.14099859472717402
# >>> print(ssim_t)
# 0.8485570724456214
# >>> print(rmsstd)
# 0.10449647669042994



##########
net_name = "newloss_cont-offsetsTrue751"
#net_name = "newloss-offsetsFalse450"
#path = "sampling/00141-gpus2-batch10-synth_ext-offsetsFalse720/rtm_"+i_str+"/saved/"
#path = "sampling/00152-gpus2-batch10-synth_ext_newloss-offsetsFalse450/rtm_"+i_str+"/saved/"

rtm_str_list = ['rtm_0913.npy', 'rtm_0914.npy', 'rtm_0894.npy', 'rtm_0928.npy', 'rtm_0982.npy', 'rtm_0921.npy', 'rtm_0937.npy', 'rtm_0930.npy', 'rtm_0861.npy', 'rtm_0854.npy', 'rtm_0977.npy', 'rtm_0979.npy', 'rtm_0927.npy', 'rtm_0878.npy', 'rtm_0929.npy', 'rtm_0895.npy', 'rtm_0967.npy', 'rtm_0831.npy', 'rtm_0903.npy', 'rtm_0904.npy', 'rtm_0855.npy', 'rtm_0860.npy', 'rtm_0943.npy', 'rtm_0869.npy', 'rtm_0823.npy', 'rtm_0900.npy', 'rtm_0972.npy', 'rtm_0880.npy', 'rtm_0991.npy', 'rtm_0996.npy', 'rtm_0998.npy', 'rtm_0923.npy', 'rtm_0800.npy', 'rtm_0980.npy', 'rtm_0964.npy', 'rtm_0896.npy', 'rtm_0898.npy', 'rtm_0916.npy', 'rtm_0810.npy', 'rtm_0946.npy', 'rtm_0990.npy', 'rtm_0881.npy', 'rtm_0973.npy', 'rtm_0825.npy', 'rtm_0917.npy', 'rtm_0834.npy', 'rtm_0965.npy', 'rtm_0981.npy', 'rtm_0873.npy', 'rtm_0988.npy', 'rtm_0922.npy']

#rtm_str_list = ['rtm_0913.npy', 'rtm_0914.npy', 'rtm_0894.npy' ]

uces = []
avg_uncert_in_bins= []
err_in_bins = []

for i_rtm_str in rtm_str_list:
    print(i_rtm_str[:-4])
    print(i_rtm_str[4:-4])
    gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_syntho_ext_test/gt_"+i_rtm_str[4:-4]+".npy")
    #path = "sampling/00152-gpus2-batch10-synth_ext_newloss-offsetsFalse450/"+i_rtm_str[:-4]+"/saved/"
    path = "sampling/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue751/"+i_rtm_str[:-4]+"/saved/"
    files_rtm = dnnlib.util.list_dir(path)
    first = np.load(path+"000000.npy")
    num_post_samples = len(files_rtm)  # Assuming num_expected is defined
    images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))
    batch_count = 0
    for file_i in files_rtm:
        file_str = path+file_i
        images_np_stack[batch_count,0,:,:] = np.load(file_str)
        batch_count +=1
    post_mean = np.mean(images_np_stack,axis=0)[0,:,:]
    post_std = np.std(images_np_stack,axis=0)[0,:,:]
    post_error = np.abs(post_mean-gt)
    uce, err_in_bin, avg_uncert_in_bin, prop_in_bin= uceloss(post_error, post_std, n_bins=20, outlier=0.0, range=[0,1])
    print(uce)
    uces.append(uce[0])
    avg_uncert_in_bins.append([avg_uncert_in_bin])
    err_in_bins.append([err_in_bin])
    #uce_total += uce
    #plt.plot(avg_uncert_in_bin,err_in_bin,color="red",alpha=0.5)

fig, ax  = plt.subplots(1, 1, figsize=(4, 4))
for i in range(len(avg_uncert_in_bins)-1):
    plt.plot(avg_uncert_in_bins[i][0],err_in_bins[i][0],color="red",linewidth=0.8,alpha=0.4)

plt.plot(avg_uncert_in_bins[-1][0],err_in_bins[-1][0],color="red",linewidth=0.8,alpha=0.4,label="Calibration w/ offsets UCE="+str(round(np.mean(uces),4)))

ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle="--",color="black",label="Perfect calibration")
plt.xlim(0,1.1); plt.ylim(0,1.1);
plt.ylabel("Error [Km/s]")
plt.xlabel("Uncertainty [Km/s]")
ax.set_aspect(1)
plt.legend()
plt.savefig(os.path.join(image_dir,net_name+str(num_post_samples)+"calibration_all.png"),bbox_inches = "tight",dpi=300); plt.close()


cond_norm = 15000
cond_loc = "/slimdata/rafaeldata/fwiuq_eod/rtms_syntho_ext_test/rtm_0811.npy"

cond = np.load(cond_loc) / cond_norm
# print(use_offsets)
# if not use_offsets:
#     print("use only zero offset")
#     cond = cond[12,:,:]
#     #cond = cond[np.newaxis,...]

# cond = torch.from_numpy(cond) 
# cond = cond.repeat(1,1,1,1).to((device))
# print(cond.shape)

image_dir = "sampling/final_plots/"

a = np.quantile(np.absolute(cond),0.98)
for i in range(25):
    plt.figure(); plt.title("Condition rtm")
    plt.imshow(cond[i,:,:], vmin=-a,vmax=a, cmap = "gray")
    plt.axis("off")
    cb = plt.colorbar(fraction=0.0235, pad=0.04); 
    plt.savefig(image_dir+"rtm_condition_"+str(i)+".png",bbox_inches = "tight",dpi=300)



