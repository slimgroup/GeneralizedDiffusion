#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 
import os
import matplotlib.pyplot as plt
import dnnlib
import numpy as np
import colorcet as cc

plt.rcParams["font.family"] = "serif"

#cmap_gt = cc.cm['CET_L03']
cmap_gt = cc.cm['rainbow4']
#cmap_gt = cc.cm['gouldian']

image_dir = "sampling/final_plots/compass/"

#i_str ="0981"
#i_str ="0873"
#i_str ="0988"
#i_str ="0922"
i_str ="0850"
d = 0.0125
gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/gts_compass_ext_test/gt_"+i_str+".npy")
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/rtms_compass_ext_test/rtm_"+i_str+".npy")[26,:,:]

vmin_gt = 1.5
vmax_gt = np.max(gt)

extent = (0,d*gt.shape[1],d*gt.shape[0],0)

a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(7,3));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"rtm_"+i_str+"_27.png"), bbox_inches = "tight", dpi=300)


plt.figure(figsize=(7,3));  
plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir+"gt_"+i_str+".png"), bbox_inches = "tight", dpi=300)




net_name = "00172-gpus2-batch10-compass-offsetsFalse150"
#net_name = "00167-gpus2-batch10-compass-offsetsTrue120"
#net_name = "00167-gpus2-batch10-compass-offsetsTrue180"
#net_name = "00167-gpus2-batch10-compass-offsetsTrue60"
#net_name = "00172-gpus2-batch10-compass-offsetsFalse90"
#net_name = "00172-gpus2-batch10-compass-offsetsFalse720"
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
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_mean_compass.png"),bbox_inches = "tight",dpi=300); plt.close()


# plt.figure(figsize=(7,3)); 
# plt.imshow(images_np_stack[0,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1, extent=extent)
# plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]"); 
# #cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
# plt.savefig(os.path.join(image_dir, net_name+"_p"+str(num_post_samples)+i_str+"_post_compass.png"),bbox_inches = "tight",dpi=300); plt.close()

#cmap_error = cc.cm['CET_L3']
cmap_error = "magma"

post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
plt.imshow(post_std,  vmin=0, vmax=0.5,   cmap = cmap_error, extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_std_compass.png"),bbox_inches = "tight",dpi=300); plt.close()
 

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
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_error_compass.png"),bbox_inches = "tight",dpi=300); plt.close()


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

plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_bouman_compass.png"),bbox_inches = "tight",dpi=300); plt.close()
  

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

plt.figure(figsize=(7,3));    #plt.title("Posterior deviation")
plt.imshow(coverage_mask,  extent=extent)
plt.ylabel("Depth [Km]"); plt.xlabel("X [Km]");  #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_coverage_compass.png"),bbox_inches = "tight",dpi=300); plt.close()
  
# for i in range(1,10):
#     plt.figure(figsize=(12,5));  
#     plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, aspect=1)
#     plt.ylabel("Z [grid]"); plt.xlabel("X [grid]"); 
#     #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
#     plt.savefig(os.path.join(image_dir, net_name+"posterior_"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/posterior_*.png sampling/final_plots/seam/samples_movie.gif
# echo "Experiment done!"
range_depth = [d * i for i in range(0, 256)]
trace_ind = 256 
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
plt.savefig(os.path.join(image_dir,str(num_post_samples)+i_str+net_name+"_traces_compass.png"),bbox_inches = "tight",dpi=300); plt.close()



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

uce, err_in_bin, avg_uncert_in_bin, prop_in_bin= uceloss(post_error, post_std, n_bins=20, outlier=0.0, range=None)


fig, ax  = plt.subplots(1, 1, figsize=(4, 4))
#ax.plot([0, 0], [1, 1], 'k--')
#plt.plot([0, 0], [1, 1], 'k--',color="black")
ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle="--",color="black",label="Perfect calibration")
plt.plot(avg_uncert_in_bin,err_in_bin,color="red",label="UCE="+str(round(uce[0],4)))
plt.xlim(0,0.8); plt.ylim(0,0.8);
plt.ylabel("Error [Km/s]")
plt.xlabel("Uncertainty [Km/s]")
ax.set_aspect(1)
plt.legend()
plt.savefig(os.path.join(image_dir,str(num_post_samples)+i_str+net_name+"_calibration_compass.png"),bbox_inches = "tight",dpi=300); plt.close()


print(net_name)
print(uce)
print(coverage)
print(perc_256)
print(rmse_t)
print(ssim_t)
print(rmsstd)



# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue120
# >>> print(uce)
# [0.00553323]
# >>> print(coverage)
# 90.62347412109375
# >>> print(perc_256)
# 6.084442138671875
# >>> print(rmse_t)
# 0.10417913083988695
# >>> print(ssim_t)
# 0.8402446668201079
# >>> 


# 00172-gpus2-batch10-compass-offsetsFalse150
# >>> print(uce)
# [0.01169183]
# >>> print(coverage)
# 82.11593627929688
# >>> print(perc_256)
# 10.553741455078125
# >>> print(rmse_t)
# 0.12988547838984035
# >>> print(ssim_t)
# 0.7881776526828402


#with more noise in diffusion
# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue120
# >>> print(uce)
# [0.00497367]
# >>> print(coverage)
# 85.20965576171875
# >>> print(perc_256)
# 6.882476806640625
# >>> print(rmse_t)
# 0.10544197945567217
# >>> print(ssim_t)
# 0.8396600758583952

# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue120
# >>> print(uce)
# [0.00851244]
# >>> print(coverage) 
# 79.23812866210938
# >>> print(perc_256)
# 8.92181396484375
# >>> print(rmse_t)
# 0.105539625145596
# >>> print(ssim_t)
# 0.8379774566535172

# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue180
# >>> print(uce)
# [0.01674954]
# >>> print(coverage)
# 69.4091796875
# >>> print(perc_256)
# 12.775421142578125
# >>> print(rmse_t)
# 0.09794022636174929
# >>> print(ssim_t)
# 0.8609377335535372

# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue60
# >>> print(uce)
# [0.00631859]
# >>> print(coverage)
# 79.66461181640625
# >>> print(perc_256)
# 7.048797607421875
# >>> print(rmse_t)
# 0.130438933920633
# >>> print(ssim_t)
# 0.7692147093425725

# >>> print(net_name)
# 00167-gpus2-batch10-compass-offsetsTrue300
# >>> print(coverage)
# 62.540435791015625
# >>> print(perc_256)
# 15.752410888671875
# >>> print(rmse_t)
# 0.09454177697623663
# >>> print(ssim_t)
# 0.8737806799210497

# >>> print(coverage)
# 59.523773193359375
# >>> print(perc_256)
# 20.70159912109375
# >>> print(rmse_t)
# 0.1272325140248651
# >>> print(ssim_t)
# 0.810670112033243

# >>> print(coverage)
# 77.79388427734375
# >>> print(perc_256)
# 7.610321044921875
# >>> print(rmse_t)
# 0.14187190934429986
# >>> print(ssim_t)
# 0.7413129179951139




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



