#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import numpy as np

ssims_curr = np.load('sampling/metrics/00154-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse661000015_ssims.npy')
rmses_curr = np.load('sampling/metrics/00154-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse661000015_rmses.npy')

np.mean(ssims_curr)
np.mean(rmses_curr)

ssims_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_ssims.npy')
ssims_new = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue751000015_ssims.npy')

rmses_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_rmses.npy')
rmses_new = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue751000015_rmses.npy')

np.mean(ssims_prev)
np.mean(ssims_new)

np.mean(rmses_prev)
np.mean(rmses_new)



#some improvement compared to epoch 450 but still not as good as with offsets. 
# >>> ssims_curr = np.load('sampling/metrics/00154-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse661000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00154-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse661000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8786010402050205
# >>> np.mean(rmses_curr)
# 0.15264828186117352

#new loss improves results on not offsets as well
# >>> ssims_curr = np.load('sampling/metrics/00152-gpus2-batch10-synth_ext_newloss-offsetsFalse450000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00152-gpus2-batch10-synth_ext_newloss-offsetsFalse450000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8613723890742568
# >>> np.mean(rmses_curr)
# 0.16367779473174188

#salt looks creat
# >>> ssims_curr = np.load('sampling/metrics/00151-gpus2-batch10-synth_salt_newloss-offsetsFalse450000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00151-gpus2-batch10-synth_salt_newloss-offsetsFalse450000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.9232199859365331
# >>> np.mean(rmses_curr)
# 0.08013433085995578

# offsets is still improving
# >>> ssims_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_ssims.npy')
# >>> ssims_new = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue751000015_ssims.npy')
# >>> 
# >>> rmses_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_rmses.npy')
# >>> rmses_new = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue751000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_prev)
# 0.8518655711324664
# >>> np.mean(ssims_new)
# 0.8936891608686346
# >>> 
# >>> np.mean(rmses_prev)
# 0.16958776158308744
# >>> np.mean(rmses_new)
# 0.14262473809234746

#the new loss is MUCH better
# >>> ssims_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_ssims.npy')
# >>> ssims_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue680000015_ssims.npy')
# >>> 
# >>> rmses_prev = np.load('sampling/metrics/00150-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue301000015_rmses.npy')
# >>> rmses_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue680000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_prev)
# 0.8518655711324664
# >>> np.mean(ssims_new)
# 0.8104840648504991
# >>> 
# >>> np.mean(rmses_prev)
# 0.16958776158308744
# >>> np.mean(rmses_new)
# 0.20357397616212222

# currently offsets still dont help but the gap is smaller
# >>> ssims_prev = np.load('sampling/metrics/00141-gpus2-batch10-synth_ext-offsetsFalse720000015_ssims.npy')
# >>> ssims_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue680000015_ssims.npy')
# >>> 
# >>> rmses_prev = np.load('sampling/metrics/00141-gpus2-batch10-synth_ext-offsetsFalse720000015_rmses.npy')
# >>> rmses_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue680000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_prev)
# 0.8275650451683734
# >>> np.mean(ssims_new)
# 0.8104840648504991
# >>> 
# >>> np.mean(rmses_prev)
# 0.18756248269965525
# >>> np.mean(rmses_new)
# 0.20357397616212222

# currently offsets dont help 
# >>> ssims_prev = np.load('sampling/metrics/00141-gpus2-batch10-synth_ext-offsetsFalse560000015_ssims.npy')
# >>> ssims_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue520000015_ssims.npy')
# >>> 
# >>> rmses_prev = np.load('sampling/metrics/00141-gpus2-batch10-synth_ext-offsetsFalse560000015_rmses.npy')
# >>> rmses_new = np.load('sampling/metrics/00142-gpus2-batch10-synth_ext-offsetsTrue520000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_prev)
# 0.8058967206806387
# >>> np.mean(ssims_new)
# 0.7772640128807483
# >>> 
# >>> np.mean(rmses_prev)
# 0.20087899838624126
# >>> np.mean(rmses_new)
# 0.32431317538828097
