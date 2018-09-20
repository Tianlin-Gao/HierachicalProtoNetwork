
Code for the NIPS 2017 paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf).

If you use this code, please cite our paper:

```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
 ```

----------------------------------
./data/mini-imagenet 与项目目录同级

### 说明

执行python -m visdom.server& \r
访问http://localhost:8097，查看训练情况

### *.sh命名方式
(m)exp[train_way]_[test_way]_[n_shot].sh \r
m代表mini-imagenet实验 \r

### adapted-random-choose-hist hard_code的地方
- 一步adapted直接写在了engine.py 27行 开始
- 以及 utils/model.py 28行 开始