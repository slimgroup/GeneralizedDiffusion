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

image_dir = "sampling/final_plots/seam/"

vmin_gt = 1.5
vmax_gt = 4.75
cmap_gt = cc.cm['rainbow4']
cmap_gray= cc.cm['CET_L1']
cmap_error = "magma"
#for i_str in ["0008"]:
i_str = "0004"
gt  = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gts_seam_filter_test/gt_"+i_str+".npy")
gt0 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/gt0s_seam_filter_test/gt0_"+i_str+".npy")
rtm = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_filter_test/rtm_"+i_str+".npy")




n = (512,1744)
n_offsets = 50
rtm_ext = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_ext_test/rtm_"+i_str+".npy")[:,:,:]
#rtm_ext = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_ext_2_stack_test/rtm_"+i_str+".npy")[:,:,:]
#rtm_ext = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_seam_ext_512_2_nostack_test/rtm_"+i_str+".npy")[:,:,:]


from matplotlib.gridspec import GridSpec

# Assuming rtm, n, d, offset_start, offset_end, plot_path, savename, etc., are already defined

# Reshape the array
y = np.transpose(rtm_ext, axes=(1, 2, 0)).reshape(n[0], n[1], n_offsets, 1)

# Set matplotlib font and size configurations
plt.rc("figure", titlesize=40)
plt.rc("font", family="serif")
plt.rc("xtick", labelsize=40)
plt.rc("ytick", labelsize=40)
plt.rc("axes", labelsize=40)
plt.rc("axes", titlesize=40)

d = (0.02,0.02)
# X, Z position in km
xpos = 15.75#e3
#for xpos in range(15,20):
zpos = 6#e3 
xgrid = int(round(xpos / d[0]))
zgrid = int(round(zpos / d[1]))
# Create a figure and a 2x2 grid of subplots
fig = plt.figure(figsize=(25, 12))
gs = GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 4], figure=fig)
axs = np.empty((2, 2), dtype=object)
axs[0, 0] = fig.add_subplot(gs[0, 0])
axs[1, 0] = fig.add_subplot(gs[1, 0])
axs[0, 1] = fig.add_subplot(gs[0, 1])
axs[1, 1] = fig.add_subplot(gs[1, 1])
# Adjust spacing
fig.subplots_adjust(hspace=0.0, wspace=0.0)
# Calculate vmin and vmax
vmax1 = np.quantile(np.abs(y[zgrid,:, :, 0].ravel()), 0.99)
vmin1 = -vmax1
vmax2 = np.quantile(np.abs(y[:, :, n_offsets // 2, 0].ravel()), 0.95)
vmin2 = -vmax2
vmax3 = np.quantile(np.abs(y[:,xgrid, :, 0].ravel()), 0.999)
vmin3 = -vmax3
#offset_start = -500
#offset_end   = 500
offset_start = -2.00
offset_end   = 2.00
# Top left subplot
axs[0, 0].imshow(y[zgrid,:, :, 0].T, aspect="auto", cmap="gray", interpolation="none", 
                 vmin=vmin1, vmax=vmax1, extent=(0, (n[1] - 1) * d[0], offset_start, offset_end))
axs[0, 0].set_ylabel("Offset [km]")
axs[0, 0].set_xticklabels([])
axs[0, 0].hlines(y=0, xmin=0, xmax=(n[1] - 1) * d[0], colors="b", linewidth=3)
axs[0, 0].vlines(x=xpos, ymin=offset_start, ymax=offset_end, colors="b", linewidth=3)
# Bottom left subplot
axs[1, 0].imshow(y[:, :, n_offsets // 2, 0], aspect="auto", cmap="gray", interpolation="none", 
                 vmin=vmin2, vmax=vmax2, extent=(0,(n[1] - 1) * d[1] , (n[0] - 1) * d[0] , 0))
axs[1, 0].set_xlabel("X [Km]")
axs[1, 0].set_ylabel("Depth [Km]")
# axs[1, 0].set_xticks([0, 1, 2, 3, 4, 5])
# axs[1, 0].set_xticklabels(["0", "1", "2", "3", "4", "5"])
axs[1, 0].set_yticks([2, 4, 6,8])
axs[1, 0].set_yticklabels(["2", "4", "6", "8"])
axs[1, 0].hlines(y=zpos, xmin=0, xmax=(n[1] - 1) * d[1], colors="b", linewidth=3)
axs[1, 0].vlines(x=xpos, ymin=0, ymax=(n[0] - 1) * d[0], colors="b", linewidth=3)
# Top right subplot (invisible)
axs[0, 1].set_visible(False)
# Bottom right subplot
axs[1, 1].imshow(y[:, xgrid, :, 0], aspect="auto", cmap="gray", interpolation="none", 
                 vmin=vmin3, vmax=vmax3, extent=(offset_start, offset_end, (n[0] - 1) * d[1], 0))
axs[1, 1].set_xlabel("Offset [km]")
axs[1, 1].set_yticklabels([])
axs[1, 1].hlines(y=zpos, xmin=offset_start, xmax=offset_end, colors="b", linewidth=3)
axs[1, 1].vlines(x=0, ymin=0, ymax=(n[0] - 1) * d[1], colors="b", linewidth=3)
# Remove spines
for ax in axs.ravel():
    if ax:
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

#plt.tight_layout()

# Save the figure
fig_name = f"fig_cig_seam.png"
fig.savefig(f"{image_dir}/{fig_name}", bbox_inches="tight", pad_inches=0.4, dpi=300)
plt.close(fig)



d = 0.02
extent = (0,d*gt.shape[1],d*gt.shape[0],0)


plt.figure(figsize=(12,5));  
plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, extent=extent,aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt_seam.png"), bbox_inches = "tight", dpi=300)


plt.figure(figsize=(12,5));  
plt.imshow(gt, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt, extent=extent,aspect=1)
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt_seam_noaxis.png"), bbox_inches = "tight", dpi=300)




plt.figure(figsize=(12,5));  
plt.imshow(gt0, vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt0_seam.png"), bbox_inches = "tight", dpi=300)

a = np.quantile(np.absolute(rtm),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(rtm, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_0.png"), bbox_inches = "tight", dpi=300)


#laplacian of gt 
from scipy import ndimage
gt_lap = ndimage.laplace(gt)

a = np.quantile(np.absolute(gt_lap),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(gt_lap, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.axis("off")
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/gt_seam_lap.png"), bbox_inches = "tight", dpi=300)




rtm1 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v1_0005.npy")
a = np.quantile(np.absolute(rtm1),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm1, cmap="seismic", vmin=-a, vmax=a,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_1_seismic.png"), bbox_inches = "tight", dpi=300)




#rtmgt = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_gt_0005.npy")
rtmgt = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_vgt_ext_better_il_0005_p1.npy")
#rtmgt = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_vgt_ext_better_il_mathias__p1.npy")
a = np.quantile(np.absolute(rtmgt),0.95)



plt.figure(figsize=(12,5));  
plt.imshow(rtmgt, cmap=cmap_gray, vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_gt_noaxis_mathias_95.png"), bbox_inches = "tight", dpi=300)


plt.figure(figsize=(12,5));  
plt.imshow(rtmgt[512:,:], cmap=cmap_gray, vmin=-a, vmax=a,extent=extent, aspect=1)
plt.axis("off")
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_gt_noaxis_half.png"), bbox_inches = "tight", dpi=300)


rtm2_ext = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_ext_0005.npy")
a = np.quantile(np.absolute(rtm2_ext),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm2_ext, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_ext_v2_noaxis.png"), bbox_inches = "tight", dpi=300)



rtm2 = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_0005.npy")
a = np.quantile(np.absolute(rtm2),0.95)

plt.figure(figsize=(12,5));  
plt.imshow(rtm2, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_noaxis.png"), bbox_inches = "tight", dpi=300)



plt.figure(figsize=(12,5));  
plt.imshow(rtm2-rtmgt, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_error_gt.png"), bbox_inches = "tight", dpi=300)



plt.figure(figsize=(12,5));  
plt.imshow(rtm2_ext-rtmgt, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_error_gt.png"), bbox_inches = "tight", dpi=300)


np.linalg.norm(rtm2-rtmgt)
np.linalg.norm(rtms_post_v2_ext_freq-rtmgt)
# >>> np.linalg.norm(rtm2-rtmgt)
# 13386592000.0
 # 11118640000.0

#np.linalg.norm(rtm2_ext-rtmgt)
#11118640000.0

#with more posterior samples 64
# >>> np.linalg.norm(rtm2_ext-rtmgt)
# 11010245000.0
# >>> 

# >>> np.linalg.norm(rtms_post_v2_ext_freq-rtmgt)
# 6.891078188599325e-05


plt.figure(figsize=(12,5));  
plt.imshow(np.abs(rtm2-rtmgt), cmap=cmap_error, vmin=0, vmax=1e8,  extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_error_gt_magma.png"), bbox_inches = "tight", dpi=300)

rtms_post_v2 = np.zeros((16,1,gt.shape[0],gt.shape[1]))

for i in range(16):
    rtms_post_v2[i,:,:,:]= np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_0005_p"+str(i+1)+".npy")


rtms_post_v2_ext = np.zeros((32,1,gt.shape[0],gt.shape[1]))

for i in range(32):
    rtms_post_v2_ext[i,:,:,:]= np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_ext_0005_p"+str(i+1)+".npy")



first = np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_ext_better_il_0005_p1.npy")
rtms_post_v2_ext_freq = np.zeros((64,1,first.shape[0],first.shape[1]))

for i in range(16):
    rtms_post_v2_ext_freq[i,:,:,:]= np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_v2_ext_better_il_0005_p"+str(i+1)+".npy")

for i in range(16,64):
    rtms_post_v2_ext_freq[i,:,:,:]= np.load("/slimdata/rafaeldata/fwiuq_eod/seam_npz/rtms_paper/rtm_vgt_ext_better_il_0005_p"+str(i+1)+".npy")


post_std_rtm_ext_freq = np.std(rtms_post_v2_ext_freq,axis=0)[0,:,:]
post_mean_rtm_ext_freq = np.mean(rtms_post_v2_ext_freq,axis=0)[0,:,:]

post_std_rtm_ext = np.std(rtms_post_v2_ext,axis=0)[0,:,:]
post_std_rtm = np.std(rtms_post_v2,axis=0)[0,:,:]

plt.figure(figsize=(12,5));  
plt.imshow(3*post_std_rtm_ext_freq[:,:], cmap=cmap_error, vmin=0,vmax=9e-08, extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_std_64.png"), bbox_inches = "tight", dpi=300)


plt.figure(figsize=(12,5));  
plt.imshow(3*post_std_rtm_ext_freq[512:,:], cmap=cmap_error, vmin=0,vmax=9e-08, extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_std_half_64.png"), bbox_inches = "tight", dpi=300)

a = np.quantile(np.absolute(post_mean_rtm_ext_freq),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(post_mean_rtm_ext_freq[512:,:], cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_mean_half_64.png"), bbox_inches = "tight", dpi=300)


for i in range(1,10):
    a = np.quantile(np.absolute(rtms_post_v2_ext_freq[i,0,:,:]),0.95)
    plt.figure(figsize=(12,5));  
    plt.imshow(rtms_post_v2_ext_freq[i,0,:,:], cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
    #plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
    plt.axis("off")
    plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_p"+str(i)+".png"), bbox_inches = "tight", dpi=200)

for i in range(1,10):
    a = np.quantile(np.absolute(rtms_post_v2_ext_freq[i,0,:,:]),0.95)
    plt.figure(figsize=(12,5));  
    plt.imshow(rtms_post_v2_ext_freq[i,0,512:,:], cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
    #plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
    plt.axis("off")
    plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_half_p"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_p*.png sampling/final_plots/seam/rtm_movie.gif

# convert -delay 15 -loop 0 sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_half_p*.png sampling/final_plots/seam/rtm_movie_half.gif


plt.figure(figsize=(12,5));  
plt.imshow(np.abs(post_mean_rtm_ext_freq[512:,:]-rtmgt[512:,:]), cmap=cmap_error, vmin=0,vmax=9e-08, extent=extent, aspect=1)
plt.axis("off")
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_freq_rtms_error_half_64.png"), bbox_inches = "tight", dpi=300)



plt.figure(figsize=(12,5));  
plt.imshow(3*post_std_rtm_ext, cmap=cmap_error, vmin=0, vmax=1e8, extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_std_gt_magma.png"), bbox_inches = "tight", dpi=300)



plt.figure(figsize=(12,5));  
plt.imshow(3*post_std_rtm, cmap=cmap_error, vmin=0, vmax=1e8, extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_std_gt_magma.png"), bbox_inches = "tight", dpi=300)

#mean of rtms 
post_mean_rtm = np.mean(rtms_post_v2,axis=0)[0,:,:]
post_mean_rtm_ext = np.mean(rtms_post_v2_ext,axis=0)[0,:,:]


from scipy.signal import hilbert
import numpy as np

def normalize_std(mu, sigma):
    analytic_mu = hilbert(mu, axis=1)
    return sigma*np.abs(analytic_mu)/(np.abs(analytic_mu)**2 + 5000000000000), analytic_mu


post_std_rtm_ext_norm, analytic_mu = normalize_std(post_mean_rtm_ext, post_std_rtm_ext)


a = np.quantile(np.absolute(post_std_rtm_ext_norm),0.98)
plt.figure(figsize=(12,5));  
plt.imshow(post_std_rtm_ext_norm, cmap=cmap_error, vmax=a,vmin=0, extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_std_norm.png"), bbox_inches = "tight", dpi=300)



np.linalg.norm(post_mean_rtm-rtmgt)
np.linalg.norm(post_mean_rtm_ext-rtmgt)
# >>> np.linalg.norm(rtm2-rtmgt)
# 13386592000.0
 #10885423727.570002
 # 9549810682.278156 #np.linalg.norm(post_mean_rtm_ext-rtmgt)

a = np.quantile(np.absolute(post_mean_rtm),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(post_mean_rtm, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_rtms_mean.png"), bbox_inches = "tight", dpi=300)


a = np.quantile(np.absolute(post_mean_rtm_ext),0.95)
plt.figure(figsize=(12,5));  
plt.imshow(post_mean_rtm_ext, cmap="gray", vmin=-a, vmax=a,extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_rtms_mean.png"), bbox_inches = "tight", dpi=300)






cmap_error = "magma"
plt.figure(figsize=(12,5));  
plt.imshow(np.abs(post_mean_rtm_ext-rtmgt), cmap=cmap_error, vmin=0, vmax=1e8,  extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_ext_error_gt_magma_wmean.png"), bbox_inches = "tight", dpi=300)



cmap_error = "magma"
plt.figure(figsize=(12,5));  
plt.imshow(np.abs(post_mean_rtm-rtmgt), cmap=cmap_error, vmin=0, vmax=1e8,  extent=extent, aspect=1)
#plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
plt.axis("off")
#cb = plt.colorbar(fraction=0.0242, pad=0.01); #cb.set_label('[Km/s]')
plt.savefig(os.path.join("sampling/final_plots/seam/rtm_seam_v2_error_gt_magma_wmean.png"), bbox_inches = "tight", dpi=300)


range_depth = range(512)
trace_ind = 256 
plt.figure(figsize=(7,3)); plt.title("Vertical trace at X grid point "+str(trace_ind))
# for i in range(1,num_post_samples):
#  
#plt.plot(range_depth,post_mean_rtm[:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="Posterior samples")
plt.plot(range_depth,rtm2[:,trace_ind], linewidth=0.8,color="red", alpha=0.3, label="RTM in posterior mean")

#plt.plot(images_np_stack[15,0,:,trace_ind], linewidth=0.8,color="black", label="Ground truth")
plt.plot(range_depth,rtmgt[:,trace_ind], linewidth=0.8,color="black", label="Ground truth ")
plt.plot(range_depth,np.linalg.norm(rtmgt[:,trace_ind])*gt[:,trace_ind], linewidth=0.8,color="green", label="gt velocity")
plt.plot(range_depth,np.linalg.norm(rtmgt[:,trace_ind])*post_mean_2[:,trace_ind], linewidth=0.8,color="blue", label="infered mean velocity")

#plt.ylim(1.2,to5.5)

plt.ylabel("Velocity [Km/s]")
plt.xlabel("Depth [grid point]")
plt.legend()
plt.savefig(os.path.join("sampling/final_plots/seam/_trace_vert_rtm.png"),bbox_inches = "tight",dpi=300); plt.close()




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



#path = "sampling/120/rtm_0001/saved/"

#path = "sampling/00179-gpus2-batch10-seam_filter-offsetsFalse210/rtm_"+i_str+"/saved/"
path = "sampling/00196-gpus2-batch4-seam_ext_512-offsetsTrue150/back/rtm_"+i_str+"/saved/"
files_rtm = dnnlib.util.list_dir(path)


first = np.load(path+"000000.npy")
num_post_samples = len(files_rtm)
images_np_stack = np.zeros((num_post_samples,1,first.shape[0],first.shape[1]))

batch_count = 0
for file_i in files_rtm:
    file_str = path+file_i
    images_np_stack[batch_count,0,:,:] = np.load(file_str)
    batch_count +=1


post_mean_1 = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t = ssim(gt,post_mean_1, data_range=np.max(gt) - np.min(gt))
# >>> ssim_t
# 0.7011606845663645



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
plt.imshow(post_std,  vmin=0, vmax=0.5,   cmap = cmap_error,extent=extent, aspect=1)
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
print(perc_256)

# >>> perc_256
# 27.57110595703125
# >>> perc_256
# 25.742949039564223

# >>> perc_256
# 3.441486883600917

plt.figure(figsize=(12,5));    #plt.title("Posterior deviation")
plt.imshow(support,  vmin=0, vmax=threshold,   cmap = cmap_error_gray,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]");   #plt.axis("off"); 
#plt.colorbar(fraction=0.0235, pad=0.04)
plt.savefig(os.path.join(image_dir, "_p"+str(num_post_samples)+i_str+"_seam_bouman_1.png"),bbox_inches = "tight",dpi=300); plt.close()
  


for i in range(1,10):
    plt.figure(figsize=(12,5));  
    plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt,extent=extent, aspect=1)
    plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
    #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
    plt.savefig(os.path.join(image_dir, "aspire_1_posterior_"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 sampling/final_plots/seam/posterior_*.png sampling/final_plots/seam/samples_movie.gif
# echo "Experiment done!"


lower_percentile=1
upper_percentile=99
lower_bound = np.percentile(images_np_stack[:,0,:,:], lower_percentile, axis=0)
upper_bound = np.percentile(images_np_stack[:,0,:,:], upper_percentile, axis=0)
# Create a mask where the ground truth is within the credible interval
coverage_mask = (gt >= lower_bound) & (gt <= upper_bound)
# Calculate the coverage as the percentage of pixels inside the credible interval
coverage = np.mean(coverage_mask) * 100 

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
#net_name_path = "00203-gpus2-batch4-seam_ext_512_real_2_stack_noback-offsetsTrue180"
#net_name_path = "00203-gpus2-batch4-seam_ext_512_real_2_stack_noback-offsetsTrue280"

#net_name_path = "00201-gpus2-batch4-seam_ext_512_real_2_stack-offsetsTrue301"
net_name_path = "00203-gpus2-batch4-seam_ext_512_real_2_stack_noback-offsetsTrue180"
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

# for i in range(1,10):
#     plt.figure(figsize=(12,5));  
#     plt.imshow(images_np_stack[i,0,:,:], vmin=vmin_gt, vmax=vmax_gt, cmap=cmap_gt,extent=extent, aspect=1)
#     plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#     #cb = plt.colorbar(fraction=0.0242, pad=0.01); cb.set_label('[Km/s]')
#     plt.savefig(os.path.join(image_dir, "aspire_2_posterior_"+str(i)+".png"), bbox_inches = "tight", dpi=200)

# convert -delay 15 -loop 0 /slimdata/rafaeldata/GeneralizedDiffusion/sampling/final_plots/seam/aspire_2_posterior_*.png /slimdata/rafaeldata/GeneralizedDiffusion/sampling/final_plots/seam/aspire_2_posterior.gif

# convert -delay 15 -loop 0 /slimdata/rafaeldata/GeneralizedDiffusion/sampling/final_plots/seam/aspire_1_posterior_*.png /slimdata/rafaeldata/GeneralizedDiffusion/sampling/final_plots/seam/aspire_1_posterior.gif


post_mean_2 = np.mean(images_np_stack,axis=0)[0,:,:]
ssim_t = ssim(gt,post_mean_2, data_range=np.max(gt) - np.min(gt))



#plot some posterior statistics
plt.figure(figsize=(12,5));  # plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
plt.imshow(post_mean_2,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,extent=extent, aspect=1)
plt.ylabel("Z [Km]"); plt.xlabel("X [Km]"); 
#cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_mean_"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


plt.figure(figsize=(12,5));  # plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
plt.imshow(post_mean_2,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt,extent=extent, aspect=1)
plt.axis("off")
plt.savefig(os.path.join(image_dir, str(num_post_samples)+i_str+net_name+"_mean_noaxis"+dataset_name+".png"),bbox_inches = "tight",dpi=300); plt.close()


#import colorcet as cc;cmap_error = cc.cm['CET_L3']
cmap_error = "magma"
post_std = np.std(images_np_stack,axis=0)[0,:,:]
plt.figure(figsize=(12,5));   #plt.title("Posterior deviation")
plt.imshow(post_std,  vmin=0, vmax=0.5,   cmap = cmap_error,extent=extent, aspect=1)
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
support = post_error / (post_std+1e-1)
perc_256 = np.mean((support) > threshold)*100
print(perc_256)
# >>> perc_256
# 2.0339825831422016

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
# 280True
# >>> print(uce)
# [0.04351995]
# >>> print(coverage)
# 73.60795047305045
# >>> print(perc_256)
# 4.407634210149083
# >>> print(rmse_t)
# 0.2120619707810562
# >>> print(ssim_t)
# 0.7039108014130266
# >>> print(rmsstd)
# 0.12860713016058048


# >>> print(net_name)
# 180True
# >>> print(uce)
# [0.03421645]
# >>> print(coverage)
# 76.90172107941514
# >>> print(perc_256)
# 17.099922950114678
# >>> print(rmse_t)
# 0.2070971217628311
# >>> print(ssim_t)
# 0.712839177041966
# >>> print(rmsstd)
# 0.13913872963614715


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
