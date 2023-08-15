# CNet
This is the code of our paper: **Correlation-Aware Graph Data Augmentation with Implicit
and Explicit Neighbors**. 

>Due to the limit of file uploads on GitHub, we use Cora data for the demo here. The rest of the experimental data can be downloaded from [Here](https://paperswithcode.com/datasets).

## Requirements
* Python 3.7.13
* Pytorch 1.10.0


## Usage Example
* CNet's Graph Data Augmentation (GDA) trial on Cora:
```jupyter notebook GDA/CNet(HF-ML).ipynb```
* t-SNE trial on Cora:
```jupyter notebook t-SNE/t-SNE-Cora.ipynb```
* Setting CNet's GNN experimental parameters:
```vi Config.ini ```
* Running CNet's GNN one trial on Cora:
```python CNet-GNN.py ```


## Running Environment 

The experimental results reported in paper are conducted on a single NVIDIA GeForce RTX 2070 with CUDA 11.4, which might be slightly inconsistent with the results induced by other platforms.


## t-SNE Results (Cora, Citeseer, Pubmed)


![](https://i.imgur.com/Zd9ryuk.png)
![](https://i.imgur.com/QtLFTq8.png)
![](https://i.imgur.com/bGtLD4D.png)



