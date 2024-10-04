#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import numpy as np

ssims_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_ssims.npy')
rmses_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_rmses.npy')

np.mean(ssims_curr)
np.mean(rmses_curr)

ssims_prev = np.load('sampling/metrics/00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451000015_ssims.npy')
ssims_new = np.load('sampling/metrics/00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691000015_ssims.npy')

rmses_prev = np.load('sampling/metrics/00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451000015_rmses.npy')
rmses_new = np.load('sampling/metrics/00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691000015_rmses.npy')

np.mean(ssims_prev)
np.mean(ssims_new)

np.mean(rmses_prev)
np.mean(rmses_new)


#compass with offsets is really good better than wiser with normalizing flows. 
# >>> ssims_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8769190112744195
# >>> np.mean(rmses_curr)
# 0.09216434046692745

# similar. with offsets slitly better
# >>> ssims_prev = np.load('sampling/metrics/00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451000015_ssims.npy')
# >>> ssims_new = np.load('sampling/metrics/00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691000015_ssims.npy')
# >>> 
# >>> rmses_prev = np.load('sampling/metrics/00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue451000015_rmses.npy')
# >>> rmses_new = np.load('sampling/metrics/00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse691000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_prev)
# 0.9102194437723119
# >>> np.mean(ssims_new)
# 0.9071890570590393
# >>> 
# >>> np.mean(rmses_prev)
# 0.0848338078908936
# >>> np.mean(rmses_new)
# 0.08695283849692045

#interesting they are about the same!
# >>> ssims_curr = np.load('sampling/metrics/00159-gpus2-batch10-synth_salt_badback-offsetsTrue240000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00159-gpus2-batch10-synth_salt_badback-offsetsTrue240000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# np.float64(0.8802014089120533)
# >>> np.mean(rmses_curr)
# np.float64(0.10789333557850776)

# >>> ssims_curr = np.load('sampling/metrics/00158-gpus2-batch10-synth_salt_badback-offsetsFalse360000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00158-gpus2-batch10-synth_salt_badback-offsetsFalse360000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# np.float64(0.8867800166191827)
# >>> np.mean(rmses_curr)
# np.float64(0.10628821865042873)

#some improvement. still better than without offsets. Now need to run with out offests longer. 
# >>> ssims_curr = np.load('sampling/metrics/00156-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue1381000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00156-gpus2-batch10-synth_ext_newloss_cont-offsetsTrue1381000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.903768908053701
# >>> np.mean(rmses_curr)
# 0.1365139224983318

#a lot of improvement. almost as good as with offsets. Now need to run with offests longer. 
# >>> ssims_curr = np.load('sampling/metrics/00155-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse1081000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00155-gpus2-batch10-synth_ext_newloss_cont-offsetsFalse1081000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8903898150298829
# >>> np.mean(rmses_curr)
# 0.14587060088604323

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
