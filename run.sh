#!/bin/bash

#mask病灶顺序：视杯、视盘、黄斑、血管、脉络膜新生血管、出血、渗出、棉花絮斑

#所有数据生成numpy后再训练
python fundus_keras train data/all_dataset20201119.csv /nas2/projects/FundusLesionSegment/fundus_clf models/fundus_mask_clf_v3.0.h5 --datatype='npy' --log_path='models/fundus_mask_clf_v3.0.log'

#keras批量预测所有的mask
python fundus_keras evaluate data/fundus_mask_clf_v2.0.1.csv /nas2/projects/FundusLesionSegment/fundus_clf models/fundus_mask_clf_v2.2.h5 data/fundus_mask_clf_v2.2.csv --datatype='npy'

#keras训练原图
python fundus_keras train data/fundus_clf_v2.1.1.csv /nas2/rawdata/fundus/data/ models/fundus_clf_v2.2.h5 --input_channels=3 --log_path='models/fundus_clf_v2.2.log'

#keras批量预测所有图片
python fundus_keras evaluate data/fundus_clf_v2.0.1.csv /nas2/rawdata/fundus/data/ models/fundus_clf_v2.2.h5 data/fundus_clf_v2.2.csv --input_channels=3

#单张图片测试
python fundus_keras inference 2.jpg models/fundus_clf_v2.0.h5


#边训练边生成numpy
python fundus_keras train /nas2/projects/FundusLesionSegment2/data/cv1/all_dataset20201102.csv /nas2/projects/FundusLesionSegment2/data fundus_mask_clf.h5

#----------------------------调试训练---------------------------------

#keras训练原图
python fundus_keras train all_dataset20201119.csv /nas2/rawdata/fundus/data/ test.h5 --input_channels=3
#keras训练mask
nohup python fundus_keras train all_dataset20201119.csv ./ mask_test.h5 --datatype='npy' > mask_test.log 2>&1 &

#keras批量预测
python fundus_keras evaluate data/test.csv /nas2/rawdata/fundus/data/ models/fundus_clf_v2.0.h5 data/test_result.csv --input_channels=3