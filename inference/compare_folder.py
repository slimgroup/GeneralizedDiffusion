#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import numpy as np

ssims_curr = np.load('sampling/metrics/00193-gpus2-batch4-seam_filter_512-offsetsFalse390000015_ssims.npy')
rmses_curr = np.load('sampling/metrics/00193-gpus2-batch4-seam_filter_512-offsetsFalse390000015_rmses.npy')

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


#aspire 1 512 is good
# >>> ssims_curr = np.load('sampling/metrics/00193-gpus2-batch4-seam_filter_512-offsetsFalse390000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00193-gpus2-batch4-seam_filter_512-offsetsFalse390000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6929933747426862
# >>> np.mean(rmses_curr)
# 0.2645962042022104

#aspire 1 extra it definitely overfits quickly not overall better than without extra
# >>> ssims_curr = np.load('sampling/metrics/00190-gpus2-batch10-seam_filter_extra-offsetsFalse210000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00190-gpus2-batch10-seam_filter_extra-offsetsFalse210000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6816747406655749
# >>> np.mean(rmses_curr)
# 0.28952915172225463

#aspire 1 extra
# >>> ssims_curr = np.load('sampling/metrics/00190-gpus2-batch10-seam_filter_extra-offsetsFalse630000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00190-gpus2-batch10-seam_filter_extra-offsetsFalse630000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6683730134496737
# >>> np.mean(rmses_curr)
# 0.310053095606509

#Aspire 1
# >>> ssims_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse210000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse210000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6803279387248475
# >>> np.mean(rmses_curr)
# 0.2800342991214415


#stacked is better with back. 
# >>> ssims_curr = np.load('sampling/metrics/00189-gpus2-batch10-seam_filter_2_stack_noback-offsetsTrue630000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00189-gpus2-batch10-seam_filter_2_stack_noback-offsetsTrue630000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6823422909349793
# >>> np.mean(rmses_curr)
# 0.2992105065781036

#aspire 3 seems a bit overfitted
# >>> ssims_curr = np.load('sampling/metrics/00185-gpus2-batch10-seam_filter_3-offsetsFalse480000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00185-gpus2-batch10-seam_filter_3-offsetsFalse480000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6770184447984672
# >>> np.mean(rmses_curr)
# 0.3022267517020551

#aspire 3 is currently better than aspire 2 but not better with aspire 2 stacked
# >>> import numpy as np
# >>> ssims_curr = np.load('sampling/metrics/00185-gpus2-batch10-seam_filter_3-offsetsFalse870000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00185-gpus2-batch10-seam_filter_3-offsetsFalse870000015_rmses.npy')
# >>> np.mean(ssims_curr)
# 0.6754199497129805
# >>> np.mean(rmses_curr)
# 0.3043104514972582

#A tiny bit of improvement from 570 epoch. probably not worth it. 
# >>> ssims_curr = np.load('sampling/metrics/00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue270000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue270000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6874346102096549
# >>> np.mean(rmses_curr)
# 0.2662565654975931

#multiple rtms helps with rmse in fact it is only better than aspire 1 if we use the extra rtms 
# >>> ssims_curr = np.load('sampling/metrics/00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue570000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00187-gpus2-batch10-seam_filter_2_stack-offsetsTrue570000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6852602468983702
# >>> np.mean(rmses_curr)
# 0.2613443545165191

# a bit better with more time. 
# >>> ssims_curr = np.load('sampling/metrics/00182-gpus2-batch10-seam_filter_2-offsetsFalse540000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00182-gpus2-batch10-seam_filter_2-offsetsFalse540000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6840824908990304
# >>> np.mean(rmses_curr)
# 0.31032668062345714

# >>> ssims_curr = np.load('sampling/metrics/00182-gpus2-batch10-seam_filter_2-offsetsFalse300000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00182-gpus2-batch10-seam_filter_2-offsetsFalse300000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6786569254598425
# >>> np.mean(rmses_curr)
# 0.3141641735152791



#did not gain a lot from the new training
# >>> ssims_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse450000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse450000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6737242350201043
# >>> np.mean(rmses_curr)
# 0.291648884192577
# >>> ssims_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse690000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00179-gpus2-batch10-seam_filter-offsetsFalse690000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6717851143879638
# >>> np.mean(rmses_curr)
# 0.29485213269727806

#compass with offsets is really good better than wiser with normalizing flows. 
#beats without offsets with lesss epochs. 
# >>> ssims_curr = np.load('sampling/metrics/00172-gpus2-batch10-compass-offsetsFalse720000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00172-gpus2-batch10-compass-offsetsFalse720000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8366788027015463
# >>> np.mean(rmses_curr)
# 0.11424790154406482

# >>> ssims_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue300000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.8769190112744195
# >>> np.mean(rmses_curr)
# 0.09216434046692745

#already overfit
# >>> ssims_curr = np.load('sampling/metrics/00173-gpus2-batch10-seam_new-offsetsFalse391000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00173-gpus2-batch10-seam_new-offsetsFalse391000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6371933183565034
# >>> np.mean(rmses_curr)
# 0.3564152611037986

# >>> ssims_curr = np.load('sampling/metrics/00173-gpus2-batch10-seam_new-offsetsFalse931000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00173-gpus2-batch10-seam_new-offsetsFalse931000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6295618823417942
# >>> np.mean(rmses_curr)
# 0.3894588063647079



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
