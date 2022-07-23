# Time2GraphRework
Improvement of shapelet extraction for Time2Graph

# AQOURSNet: Time2Graph Rework

[Ziyuan Chen](mailto:ziyuan.20@intl.zju.edu.cn), [Zhirong Chen](mailto:zhirong.20@intl.zju.edu.cn) | July 2022 <br>
Summer Research @ [Yang Yang](https://person.zju.edu.cn/yangy) [Lab](http://yangy.org/), Zhejiang University

This work is protected under the [MIT License](https://opensource.org/licenses/MIT). <br> **Copyright (c) 2022 Ziyuan Chen & Zhirong Chen** unless otherwise noted. 

<br>
<font size=5>

<font color=#00A1E9>**A**</font>utoencoded <br>
<font color=#00A1E9>**Q**</font>uantification <br>
<font color=#00A1E9>**O**</font>f <br>
<font color=#00A1E9>**U**</font>nsupervised <br>
<font color=#00A1E9>**R**</font>epresentative <br>
<font color=#00A1E9>**S**</font>hapelets <br>

</font>
<br>

## Running the Program
    $ python main.py
- ***WARNING:** the condensed `ts2vec` module has **not** been thoroughly tested. Clone [yuezhihan/ts2vec](https://github.com/yuezhihan/ts2vec) under the same folder in case it fails.*

## Model Pipeline
0. Data preparation
1. Time series ---extract---> Shapelets
2. Time series & shapelets ---embed---> Series embedding
3. Series embedding ---construct---> Graph
4. Graph ---embed---> Graph embedding
5. Graph embedding ---predict---> Predicted classes

## Folder Structure
- `ref_papers` - papers associated with *shapelets* providing essential background knowledge
    - `traditional` - evolution path since [2009](https://doi.org/10.1145/1557019.1557122): [Logical](https://doi.org/10.1145/2020408.2020587), [ST](https://doi.org/10.1145/2339530.2339579), [Unsupervised](https://doi.org/10.1109/ICDM.2012.26), [FS](https://doi.org/10.1137/1.9781611972832.74), [LS](https://doi.org/10.1145/2623330.2623613), [Random](https://doi.org/10.48550/arXiv.1503.05018), [Forest](https://doi.org/10.1109/BigData.2014.7004344), [Random forest](https://doi.org/10.1007/s10618-016-0473-y), etc. 
    - `neural_network` - [DNNs](https://doi.org/10.1109/IJCNN.2017.7966039) enter the game! [Dynamic](https://doi.org/10.48550/arXiv.1906.00917), [Adversarial](https://doi.org/10.48550/arXiv.1906.00917), [Adv. Dynamic](https://doi.org/10.1609/AAAI.v34i04.5948), [ShapeNet](https://ojs.aaai.org/index.php/AAAI/article/view/17018), [BSPCover](https://doi.org/10.1109/ICDE51399.2021.00254), etc. ([Review](https://doi.org/10.1007/s10618-019-00619-1))
- `cached_programs` - historical versions and experiments of KMeans, SVM, MLP, ResNet, Time2Vec, hierarchies, etc. 
    - ***WARNING:** Codes in the cache may not run properly since they are not optimized for environment compatibility.*
- `affiliated_licenses` - LICENSEs for code segments from [yuezhihan](https://github.com/yuezhihan/ts2vec/blob/main/LICENSE), [subhadarship](https://github.com/subhadarship/kmeans_pytorch/blob/master/LICENSE), [DynamicTimeWarping](https://github.com/DynamicTimeWarping/dtw-python/blob/master/LICENSE), and [pyg-team](https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE)@GitHub. 
- `ucr_dataset` - a neatly formatted version of [the UCR Dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip) in `.npz` format
    - The numpy arrays contained in each file have keys `train_data`, `train_label`, `test_data`, `test_label`
    - `*_data` has shape `(num_samples, num_features)`, `*_label` has shape `(num_samples,)`

## Call Hierarchy
- `main.py`
    - `utils.py` (#0: `ucr_dataset`, #3 wrapper: `GraphDataset`)
    - `construct_graph.py` (#1 wrapper: `extract_shapelets`, #2: `embed_series`, #3: `adjacency_matrix`)
        - `ts2vec.py` (#1 optional enhancement: `TS2Vec`)
        - `kmeans.py` (#1: `kmeans`)
        - `dtw`
    - `network.py` (#4: `GAT`, #5: `MultilayerPerceptron` & `FCResidualNetwork`)
        - `xgboost`

## Credits
- Original Time2Graph model by [Cheng et al., 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5769) & [Cheng et al., 2021](https://ieeexplore.ieee.org/document/9477138), code inspired by [petecheng@GitHub](https://github.com/petecheng/Time2GraphPlus)
- Time2Vec algorithm (essence of AQOURS) by [Kazemi et al., 2019](https://arxiv.org/abs/1907.05321), code adapted from [yuezhihan@GitHub](https://github.com/yuezhihan/ts2vec)
    - **Copyright (c) 2022 Zhihan Yue**
- KMeans acceleration on PyTorch by [subhadarship@GitHub](https://github.com/subhadarship/kmeans_pytorch), with adaptations
    - **Copyright (c) 2020 subhadarshi**
- DTW algorithm implemented by [DynamicTimeWarping@GitHub](https://github.com/DynamicTimeWarping/dtw-python)
    - **Copyright (c) 2019 Toni Giorgino**
- GAT structure by [Veličković et al., 2017](https://arxiv.org/abs/1710.10903), code inspired by [Stanford CS224w](https://colab.research.google.com/drive/1X4uOWv_xkefDu_h-pbJg-fEkMfR7NGz9?usp=sharing), [pyg-team@GitHub](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py), [DGraphXinye@GitHub](https://github.com/DGraphXinye/DGraphFin_baseline/blob/master/models/gat.py)
    - **Copyright (c) 2021 Matthias Fey, Jiaxuan You** (pyg-team)
- MLP & ResNet (prediction tail) structures inspired by [Wang et al., 2017](https://ieeexplore.ieee.org/document/7966039)
- Data from [the 
UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), as documented in [Dau et al., 2018](https://arxiv.org/abs/1810.07758)
- Python modules: [`torch`](https://pytorch.org/docs/stable/index.html), [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/), [`kmeans`](https://pypi.org/project/kmeans-pytorch/), [`dtw`](https://dynamictimewarping.github.io/python/), [`xgboost`](https://xgboost.readthedocs.io/en/stable/python/index.html), [`sklearn`](https://scikit-learn.org/stable/user_guide.html)
