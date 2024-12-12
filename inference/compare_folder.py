#module load Miniconda/3;module load ompi-cpu; salloc -A rafael -t01:80:00 --partition=cpu --mem-per-cpu=20G 

import numpy as np


# Load the arrays
loaded_arrays = np.load('sampling/metrics/00213-gpus1-batch4-synth_dobs-offsetsTrue1081_metrics.npz')

# Access individual arrays
ssims = loaded_arrays['ssims']
rmses = loaded_arrays['rmses']
covs = loaded_arrays['covs']
uces = loaded_arrays['uces']
zscores = loaded_arrays['zscores']

np.mean(ssims)
np.mean(rmses)
np.mean(covs)
np.mean(uces)
np.mean(zscores)

np.std(ssims)
np.std(rmses)
np.std(covs)
np.std(uces)
np.std(zscores)


import numpy as np

ssims_curr = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue270000015_ssims.npy')
rmses_curr = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue270000015_rmses.npy')

np.mean(ssims_curr)
np.mean(rmses_curr)

# >>> loaded_arrays = np.load('sampling/metrics/00218-gpus2-batch4-seam_ext_512_3_stack_noback-offsetsTrue120_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7169016679954983
# >>> np.mean(rmses)
# 0.23122311590148342
# >>> np.mean(covs)
# 55.71608237170298
# >>> np.mean(uces)
# 0.0483035998833181
# >>> np.mean(zscores)
# 4.769225514263188
# >>> 
# >>> np.std(ssims)
# 0.028271175999481018
# >>> np.std(rmses)
# 0.022106009850062096
# >>> np.std(covs)
# 1.5399961236154815
# >>> np.std(uces)
# 0.011837147370020083
# >>> np.std(zscores)
# 1.8214181753460594

#aspire 3 maybe getting worse
# >>> loaded_arrays = np.load('sampling/metrics/00204-gpus2-batch4-seam_ext_512_real_3_stack-offsetsTrue200_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7227434032992501
# >>> np.mean(rmses)
# 0.21906454702538442
# >>> np.mean(covs)
# 37.36916638295584
# >>> np.mean(uces)
# 0.06881969779673447
# >>> np.mean(zscores)
# 6.832535769961296
# >>> 
# >>> np.std(ssims)
# 0.02690298230725761
# >>> np.std(rmses)
# 0.01966425514978344
# >>> np.std(covs)
# 3.193214788271314
# >>> np.std(uces)
# 0.00823177193375164
# >>> np.std(zscores)
# 1.274483736791525

# aspire 3 stilln ot better than aspire 2
# >>> loaded_arrays = np.load('sampling/metrics/00204-gpus2-batch4-seam_ext_512_real_3_stack-offsetsTrue120_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7245998289427913
# >>> np.mean(rmses)
# 0.2150370063819609
# >>> np.mean(covs)
# 45.32402108568664
# >>> np.mean(uces)
# 0.05870244783414155
# >>> np.mean(zscores)
# 5.9279555574469605
# >>> 
# >>> np.std(ssims)
# 0.024498743490204842
# >>> np.std(rmses)
# 0.019260152768833628
# >>> np.std(covs)
# 2.1534256127651696
# >>> np.std(uces)
# 0.008581719795724167
# >>> np.std(zscores)
# 1.1931537093997424



#getting better
# >>> np.mean(ssims)
# 0.5307845384098275
# >>> np.mean(rmses)
# 0.38523646889772833
# >>> np.mean(covs)
# 78.98721133961396
# >>> np.mean(uces)
# 0.049382656507529786
# >>> np.mean(zscores)
# 3.2493890500536153
# >>> 
# >>> np.std(ssims)
# 0.08200322897402598
# >>> np.std(rmses)
# 0.10347168921418909
# >>> np.std(covs)
# 8.33421910456727
# >>> np.std(uces)
# 0.03040071055139078
# >>> np.std(zscores)
# 2.9804592286968363


#everything got better except covs and zscores
# >>> loaded_arrays = np.load('sampling/metrics/00213-gpus1-batch4-synth_dobs-offsetsTrue1321_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.5698818723963646
# >>> np.mean(rmses)
# 0.348873324873113
# >>> np.mean(covs)
# 78.27143949620864
# >>> np.mean(uces)
# 0.04424946581552323
# >>> np.mean(zscores)
# 3.0672260359221815
# >>> 
# >>> np.std(ssims)
# 0.09576020365718348
# >>> np.std(rmses)
# 0.10195795617096068
# >>> np.std(covs)
# 8.780070503316468
# >>> np.std(uces)
# 0.032244206511000996
# >>> np.std(zscores)
# 3.1088121093429892


#about the same but worse than if longer
# >>> loaded_arrays = np.load('sampling/metrics/00213-gpus1-batch4-synth_dobs-offsetsTrue1081_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.5387347442822475
# >>> np.mean(rmses)
# 0.3738034236349795
# >>> np.mean(covs)
# 79.05493343577666
# >>> np.mean(uces)
# 0.04645151949543577
# >>> np.mean(zscores)
# 3.087017582912071
# >>> 
# >>> np.std(ssims)
# 0.08463746131193432
# >>> np.std(rmses)
# 0.10486594461350601
# >>> np.std(covs)
# 8.681640154787397
# >>> np.std(uces)
# 0.03086485046625691
# >>> np.std(zscores)
# 2.980337109090369

#getting better
# >>> loaded_arrays = np.load('sampling/metrics/00210-gpus1-batch4-synth_dobs-offsetsTrue620_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.484961987952801
# >>> np.mean(rmses)
# 0.40259596682450344
# >>> np.mean(covs)
# 82.95522951612286
# >>> np.mean(uces)
# 0.057856764996485295
# >>> np.mean(zscores)
# 2.6873270670572915
# >>> 
# >>> np.std(ssims)
# 0.07786129932837801
# >>> np.std(rmses)
# 0.1192083335641563
# >>> np.std(covs)
# 7.677647077916854
# >>> np.std(uces)
# 0.031230337231043376
# >>> np.std(zscores)
# 2.6350087614250643

#not a lot of improvement
# >>> loaded_arrays = np.load('sampling/metrics/00210-gpus1-batch4-synth_dobs-offsetsTrue220_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.4080936209952558
# >>> np.mean(rmses)
# 0.46725280478423925
# >>> np.mean(covs)
# 81.92481994628906
# >>> np.mean(uces)
# 0.08008446782216434
# >>> np.mean(zscores)
# 3.7309885025024414
# >>> 
# >>> np.std(ssims)
# 0.07785501390730207
# >>> np.std(rmses)
# 0.1300395582131322
# >>> np.std(covs)
# 8.116402241559541
# >>> np.std(uces)
# 0.05078178253338384
# >>> np.std(zscores)
# 3.984563287401763

# >>> loaded_arrays = np.load('sampling/metrics/00210-gpus1-batch4-synth_dobs-offsetsTrue300_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.42567974901061645
# >>> np.mean(rmses)
# 0.4856874556536691
# >>> np.mean(covs)
# 82.63774198644302
# >>> np.mean(uces)
# 0.08896429621334608
# >>> np.mean(zscores)
# 3.4830579570695464
# >>> 
# >>> np.std(ssims)
# 0.07202251676002124
# >>> np.std(rmses)
# 0.20640115610740067
# >>> np.std(covs)
# 10.735764906492564
# >>> np.std(uces)
# 0.09522745025434916
# >>> np.std(zscores)
# 5.341030085510689

#noback, nice in some metrics better in some not but already overfitted
# >>> loaded_arrays = np.load('sampling/metrics/00203-gpus2-batch4-seam_ext_512_real_2_stack_noback-offsetsTrue280_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7167693197677482
# >>> np.mean(rmses)
# 0.23058964272684768
# >>> np.mean(covs)
# 73.42051934758457
# >>> np.mean(uces)
# 0.0460003641964621
# >>> np.mean(zscores)
# 4.87106743208859
# >>> 
# >>> np.std(ssims)
# 0.024550087383568266
# >>> np.std(rmses)
# 0.017991341002105806
# >>> np.std(covs)
# 4.236705891251555
# >>> np.std(uces)
# 0.009264998813488532
# >>> np.std(zscores)
# 1.337994795390478

# >>> loaded_arrays = np.load('sampling/metrics/00203-gpus2-batch4-seam_ext_512_real_2_stack_noback-offsetsTrue180_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.720455120041112
# >>> np.mean(rmses)
# 0.2337460752918894
# >>> np.mean(covs)
# 75.60052434238818
# >>> np.mean(uces)
# 0.03814248556142938
# >>> np.mean(zscores)
# 3.7958267631880735
# >>> 
# >>> np.std(ssims)
# 0.022192326233522953
# >>> np.std(rmses)
# 0.03206662392326925
# >>> np.std(covs)
# 4.610545248794661
# >>> np.std(uces)
# 0.006947039572712862
# >>> np.std(zscores)
# 1.0632011309875793



#still not beetter
# >>> # Load the arrays
# >>> loaded_arrays = np.load('sampling/metrics/00204-gpus2-batch4-seam_ext_512_real_3_stack-offsetsTrue80_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7203704059282431
# >>> np.mean(rmses)
# 0.210050331892989
# >>> np.mean(covs)
# 49.28212097087975
# >>> np.mean(uces)
# 0.05665823506022693
# >>> np.mean(zscores)
# 5.759670912516382
# >>> 
# >>> np.std(ssims)
# 0.0227354552800916
# >>> np.std(rmses)
# 0.013898727680911051
# >>> np.std(covs)
# 1.8963951180916487
# >>> np.std(uces)
# 0.009992718384057379
# >>> np.std(zscores)
# 1.2770640970374265

# >>> loaded_arrays = np.load('sampling/metrics/00200-gpus2-batch4-seam_ext_512_real_2_stack-offsetsTrue151_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7295459824755643
# >>> np.mean(rmses)
# 0.2077835012165946
# >>> np.mean(covs)
# 56.06289084898222
# >>> np.mean(uces)
# 0.04101081638313578
# >>> np.mean(zscores)
# 19.938225702408257
# >>> 
# >>> np.std(ssims)
# 0.024408173395784678
# >>> np.std(rmses)
# 0.01687550487681778
# >>> np.std(covs)
# 3.9955105797173434
# >>> np.std(uces)
# 0.006693113609907998
# >>> np.std(zscores)
# 4.295685199698915

#some better some worse. 
# >>> loaded_arrays = np.load('sampling/metrics/00201-gpus2-batch4-seam_ext_512_real_2_stack-offsetsTrue301_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.724160175837119
# >>> np.mean(rmses)
# 0.21803732024997668
# >>> np.mean(covs)
# 70.03914369355648
# >>> np.mean(uces)
# 0.0474121844318301
# >>> np.mean(zscores)
# 23.013249668506305
# >>> 
# >>> np.std(ssims)
# 0.025261689875790784
# >>> np.std(rmses)
# 0.019381096339456665
# >>> np.std(covs)
# 4.097674221668408
# >>> np.std(uces)
# 0.007424037085543199
# >>> np.std(zscores)
# 3.8503394813348484

# >>> loaded_arrays = np.load('sampling/metrics/00200-gpus2-batch4-seam_ext_512_real_2_stack-offsetsTrue211_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7275727373705907
# >>> np.mean(rmses)
# 0.21090943763969883
# >>> np.mean(covs)
# 67.3618561630949
# >>> np.mean(uces)
# 0.04185350461883469
# >>> np.mean(zscores)
# 3.9641774028813073
# >>> 
# >>> np.std(ssims)
# 0.02415009612256487
# >>> np.std(rmses)
# 0.01754418804486786
# >>> np.std(covs)
# 4.010773118457384
# >>> np.std(uces)
# 0.007568903725830948
# >>> np.std(zscores)
# 0.8173202586109733

#looking good
# >>> loaded_arrays = np.load('sampling/metrics/00200-gpus2-batch4-seam_ext_512_real_2_stack-offsetsTrue211_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7275634117565699
# >>> np.mean(rmses)
# 0.21087600125756942
# >>> np.mean(covs)
# 67.25204887740108
# >>> np.mean(uces)
# 0.04207244608432269
# >>> np.mean(zscores)
# 21.166656214162845
# >>> 
# >>> np.std(ssims)
# 0.024045723996114855
# >>> np.std(rmses)
# 0.017512292099698394
# >>> np.std(covs)
# 4.091772267052213
# >>> np.std(uces)
# 0.0073202958376669906
# >>> np.std(zscores)
# 4.036644185988019

#overfitting 
# >>> loaded_arrays = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue480_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7055780613169548
# >>> np.mean(rmses)
# 0.2599192559240713
# >>> np.mean(covs)
# 65.78502130070957
# >>> np.mean(uces)
# 0.07015967415939775
# >>> np.mean(zscores)
# 26.278224560098913
# >>> 
# >>> np.std(ssims)
# 0.030865633115248273
# >>> np.std(rmses)
# 0.015403956885010946
# >>> np.std(covs)
# 7.021543467028213
# >>> np.std(uces)
# 0.015835032303863263
# >>> np.std(zscores)
# 5.339200375668218

#with 32 samples
# >>> loaded_arrays = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue150_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7106154824486233
# >>> np.mean(rmses)
# 0.2561000466199098
# >>> np.mean(covs)
# 60.649220877831134
# >>> np.mean(uces)
# 0.046394261623783135
# >>> np.mean(zscores)
# 3.4453505769782113
# >>> 
# >>> np.std(ssims)
# 0.025797653758581715
# >>> np.std(rmses)
# 0.013763212081766465
# >>> np.std(covs)
# 10.328265786436656
# >>> np.std(uces)
# 0.015410428243075423
# >>> np.std(zscores)
# 0.7559691381723516

# with 64 posterior samples (better of course)
# >>> # Load the arrays
# >>> loaded_arrays = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue150_metrics.npz')

# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.7132293127960135
# >>> np.mean(rmses)
# 0.2552170294791712
# >>> np.mean(covs)
# 66.60406830113962
# >>> np.mean(uces)
# 0.04374536692761795
# >>> np.mean(zscores)
# 18.042734128619554
# >>> 
# >>> np.std(ssims)
# 0.02604409884285036
# >>> np.std(rmses)
# 0.013866959374065407
# >>> np.std(covs)
# 10.47409784393786
# >>> np.std(uces)
# 0.015462894308613065
# >>> np.std(zscores)
# 5.006301404949596

# with 16 posterior samples
# >>> loaded_arrays = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue150_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.705960032186467
# >>> np.mean(rmses)
# 0.25814610136600025
# >>> np.mean(covs)
# 52.92839960220757
# >>> np.mean(uces)
# 0.04989581636121334
# >>> np.mean(zscores)
# 20.778928984195815
# >>> 
# >>> np.std(ssims)
# 0.025434530270569855
# >>> np.std(rmses)
# 0.013425967384398874
# >>> np.std(covs)
# 9.771400488703607
# >>> np.std(uces)
# 0.014787930811142316
# >>> np.std(zscores)
# 6.027910214985772

# >>> loaded_arrays = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue300_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.704943518748251
# >>> np.mean(rmses)
# 0.25754659218348536
# >>> np.mean(covs)
# 53.50057619427322
# >>> np.mean(uces)
# 0.06475994283854503
# >>> np.mean(zscores)
# 25.522172560385606
# >>> 
# >>> np.std(ssims)
# 0.03067459032775649
# >>> np.std(rmses)
# 0.016155788647581555
# >>> np.std(covs)
# 10.79709972085251
# >>> np.std(uces)
# 0.01670871129024418
# >>> np.std(zscores)
# 5.748125431997944

# >>> loaded_arrays = np.load('sampling/metrics/00162-gpus2-batch10-synth_salt_badback_cont-offsetsTrue631_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.9188846450158819
# >>> np.mean(rmses)
# 0.0795380203624175
# >>> np.mean(covs)
# 72.522720636106
# >>> np.mean(uces)
# 0.009228814999202957
# >>> np.mean(zscores)
# 5.416675642424939
# >>> 
# >>> np.std(ssims)
# 0.05825978780079537
# >>> np.std(rmses)
# 0.05556564070987338
# >>> np.std(covs)
# 8.60000536041981
# >>> np.std(uces)
# 0.009429469369150487
# >>> np.std(zscores)
# 3.7857284333661188

# >>> loaded_arrays = np.load('sampling/metrics/00163-gpus2-batch10-synth_salt_badback_cont-offsetsFalse931_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.9109068750284395
# >>> np.mean(rmses)
# 0.085071461105811
# >>> np.mean(covs)
# 74.68836167279412
# >>> np.mean(uces)
# 0.012295457162930514
# >>> np.mean(zscores)
# 7.844842648973652
# >>> 
# >>> np.std(ssims)
# 0.06486796528994532
# >>> np.std(rmses)
# 0.06104994728736084
# >>> np.std(covs)
# 9.649387016820071
# >>> np.std(uces)
# 0.013810440727939903
# >>> np.std(zscores)
# 6.4216993304931


# >>> loaded_arrays = np.load('sampling/metrics/00172-gpus2-batch10-compass-offsetsFalse150_metrics.npz')
# >>> 
# >>> # Access individual arrays
# >>> ssims = loaded_arrays['ssims']
# >>> rmses = loaded_arrays['rmses']
# >>> covs = loaded_arrays['covs']
# >>> uces = loaded_arrays['uces']
# >>> zscores = loaded_arrays['zscores']
# >>> 
# >>> np.mean(ssims)
# 0.8112031379236931
# >>> np.mean(rmses)
# 0.12307138198545084
# >>> np.mean(covs)
# 73.78565470377605
# >>> np.mean(uces)
# 0.01314598669507817
# >>> np.mean(zscores)
# 10.716078016493055
# >>> 
# >>> np.std(ssims)
# 0.036243402401979
# >>> np.std(rmses)
# 0.018824216518388705
# >>> np.std(covs)
# 3.674732528279025
# >>> np.std(uces)
# 0.006768468238817529
# >>> np.std(zscores)
# 2.587325933991739

# >>> loaded_arrays = np.load('sampling/metrics/00167-gpus2-batch10-compass-offsetsTrue120_metrics.npy.npz')
# >>> 
# >>> 
# >>> np.mean(ssims)
# 0.846520537491669
# >>> np.mean(rmses)
# 0.10522567349499752
# >>> np.mean(covs)
# 74.83503553602431
# >>> np.mean(uces)
# 0.011068081303934729
# >>> np.mean(zscores)
# 9.924062093098959
# >>> 
# >>> np.std(ssims)
# 0.034595731974445286
# >>> np.std(rmses)
# 0.015161473883942124
# >>> np.std(covs)
# 2.768648952989299
# >>> np.std(uces)
# 0.005916031422936865
# >>> np.std(zscores)
# 2.6244288110246723a

# >>> ssims_curr = np.load('sampling/metrics/00194-gpus2-batch10-seam_ext-offsetsTrue300000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00194-gpus2-batch10-seam_ext-offsetsTrue300000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6908418704794292
# >>> np.mean(rmses_curr)
# 0.27362565730382493


#offsets and 512 is insanely good

# >>> ssims_curr = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue270000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00196-gpus2-batch4-seam_ext_512-offsetsTrue270000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.7056490746015949
# >>> np.mean(rmses_curr)
# 0.2575958135588526

#aspire 1 with offsets a bit better than 
# >>> ssims_curr = np.load('sampling/metrics/00194-gpus2-batch10-seam_ext-offsetsTrue180000015_ssims.npy')
# >>> rmses_curr = np.load('sampling/metrics/00194-gpus2-batch10-seam_ext-offsetsTrue180000015_rmses.npy')
# >>> 
# >>> np.mean(ssims_curr)
# 0.6903475472697458
# >>> np.mean(rmses_curr)
# 0.27103811639948083

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
