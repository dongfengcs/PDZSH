# PDZSH
code for  原型对齐和域感知的零样本哈希/Prototype-aligned and Domain-aware Zero-Shot Hashing
please contact us at fengdongcs@outlook.com for questions

1.Thanks to Yongqin Xian, the APY, AWA2, ImageNet datasets can be downloaded from
"https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly"
also thanks to Wei-Lun Chao, the w2v of ImageNet can be downloaded from 
https://github.com/pujols/zero-shot-learning
=========

2.run 'demo.m' for evaluating our PDZSH
=========

1.1 split_data.m : construct training set, test set and retrieval set
1.2 construct_ImageNet_subet.m : construct ImageNet subset for evaluation

2.1 evaluate_PDZSH.m : run and evaluate our PDZSH
2.2 train_PDZSH_Y.m : generate class prototype
2.3 train_PDZSH_B.m : learn and generate hashing codes