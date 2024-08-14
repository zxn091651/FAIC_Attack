age-estimation-pytorch-master/age-estimation-pytorch-master/appa-real-release
python train.py --data_dir [age-estimation-pytorch-master/age-estimation-pytorch-master/appa-real-release] --tensorboard tf_log
python train.py --data_dir [age-estimation-pytorch-master/age-estimation-pytorch-master/appa-real-release] --tensorboard tf_log MODEL.ARCH se_resnet50 TRAIN.OPT sgd TRAIN.LR 0.1
age-estimation-pytorch-master/age-estimation-pytorch-master/checkpoint/epoch047_0.02328_3.8090.pth
python test.py --data_dir appa-real-release --resume checkpoint/epoch047_0.02328_3.8090.pth