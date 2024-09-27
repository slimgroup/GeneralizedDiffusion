import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

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


