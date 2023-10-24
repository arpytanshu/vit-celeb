# Variation AutoEncoders + Celeb-A dataset


/shared/datasets/Celeb-A/captions.json



list_eval_partition.txt
list_attr_celeba.txt
img_align_celeba/*.jpg

### Stats of whole dataset
'mean': [0.5061, 0.4254, 0.3828]
'var': [0.0964, 0.0842, 0.0839]
'std': [0.3105, 0.2903, 0.2896]

### Stats of train split
'mean': [0.5068, 0.4262, 0.3835]
'var': [0.0963, 0.0843, 0.0839]
'std': [0.3103, 0.2903, 0.2896]


@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}



sanity run

e:2 iter:400 avg train loss: 0.369140625
P=0.787 R=0.718 F1=0.742 ACC=0.837, LOSS: 0.36328125
e:2 iter:410 avg train loss: 0.36328125

e:3 iter:400 avg train loss: 0.34375000
test: P=0.786 R=0.726 F1=0.748 ACC=0.839, LOSS: 0.359375
e:3 iter:410 avg train loss: 0.349609375