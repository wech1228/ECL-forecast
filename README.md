# 基于用电量的时间序列预测算法

## 项目背景

本次项目的目标是基于用户的历史用电量数据预测未来一段时间的用电量，本次任务用到的模型包括ARIMA、LSTM、Informer。通过对比各模型的性能差异，更深入地学习各模型的优缺点。



## 数据集

数据集是来自UCI的时间序列数据，此数据集记录了370个用户的每15分钟用电量，时间从2011年到2014年，数据集一共包含140256行，370列，并且无缺失值。数据情况如下：

![image-20231110091858318](/Users/wenchen/Library/Application Support/typora-user-images/image-20231110091858318.png)

下载链接：[DOWNLOAD (uci.edu)](https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip)



## 项目所需python包

1. ARIMA

- Python 3.9.13
- matplotlib == 3.5.1
- numpy == 1.22.3
- pandas == 1.4.2
- sklearn == 1.0.2

* statsmodels == 0.14.0
* pmdarima == 2.0.3

2. LSTM

- Python 3.9.13
- matplotlib == 3.5.1
- numpy == 1.22.3
- pandas == 1.4.2
- sklearn == 1.0.2

* tensorflow == 2.9.2
* keras == 2.9.0

3. Informer

- Python 3.9.13
- matplotlib == 3.5.1
- numpy == 1.22.3
- pandas == 1.4.2
- sklearn == 1.0.2
- torch == 1.11.0



## 项目运行

1. ARIMA

本文件夹包含文件及使用方式如下：

* `ARIMA.ipynb`:加入滚动预测（预测结果加入训练集再进行下一轮预测）的ARIMA模型安装所有依赖后即可运行。运行结果保存至`result-arima`文件下。
* `result-arima`:ARIMA不同预测步长的预测结果与真实结果对比。
* `ARIMA-2.ipynb`:训练历史数据，一次预测多步时序，没有加入滚动预测。运行结果保存至`result-arima2`文件下。
* `result-arima2`:ARIMA-2不同预测步长的预测结果与真实结果对比。

ARIMA有关参数的详细说明如下：

| Parameter name | Description of parameter |
| :-- | --- |
| pred_len | prediction sequence length |
| horizen  | The number of steps to predict |

ARIMA-2有关参数的详细说明如下:

| Parameter name | Description of parameter |
| :-- | --- |
| pred_len | prediction sequence length |

备注：此文件输入数据集为原始数据集，将文件放入ARIMA文件夹下即可运行相关python文件。



2. LSTM(目前为Tensorflow版本，pytorch版后续进行上传)

本文件夹包含文件如下：

`LSTM.ipynb`:原始数据集下载后，即可运行此文件，模型结果、训练曲线、预测结果会保存至`result`、`model`文件夹，文件名为自定义参数缩写。

`results`:不同参数下训练曲线及预测结果

`model`:不同参数下保存的模型

LSTM有关参数的详细说明如下：

| Parameter name | Description of parameter                                     |
| :------------- | ------------------------------------------------------------ |
| data | data (defaults to `ECL`|
| root_path      | The root path of the data file (defaults to `./`)   |
| data_path      | The data file name (defaults to `LD2011_2014.txt`)                 |
| lookback_len       | The input sequence length of timesteps (defaults to `24`) |
| pred_len         | The prediction sequence length (defaults to `48`)            |
| train_epochs           | Train epochs(defaults to `80`) |
| batch_size    | Batch size of train input data (defaults to `256`) |
|	 patience	|	Early stopping patience	(defaults to `20`) |
|	lradj | Adjust learning rate	(defaults to `0.5`)	|



3. Informer

`main_informer.py`:运行此主程序即可运行本模型，`checkpoints`文件夹中包含训练完成的模型，后缀名为`.pth`，该模型文件包含完整的模型架构与各层权重，可以通过`torch.load`函数加载模型。

`results`:文件夹中包含`metrics.npy`、`pred.npy`、`true.npy`三个文件，`pred.npy`表示模型预测值，`true.npy`表示序列真实值。

`data`:此文件下包含预处理后的数据集，包含2012至2013年每小时的用电量数据。

Informer有关参数的详细说明如下：

| Parameter name   | Description of parameter                                     |
| :--------------- | ------------------------------------------------------------ |
| model            | The model of experiment. This can be set to `informer`, `informerstack`, `informerlight(TBD)` |
| data             | The dataset name                                             |
| root_path        | The root path of the data file (defaults to `./data/ETT/`)   |
| data_path        | The data file name (defaults to `ETTh1.csv`)                 |
| features         | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate) |
| target           | Target feature in S or MS task (defaults to `OT`)            |
| freq             | Freq for time features encoding (defaults to `h`). This can be set to `s`,`t`,`h`,`d`,`b`,`w`,`m` (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h |
| checkpoints      | Location of model checkpoints (defaults to `./checkpoints/`) |
| seq_len          | Input sequence length of Informer encoder (defaults to 96)   |
| label_len        | Start token length of Informer decoder (defaults to 48)      |
| pred_len         | Prediction sequence length (defaults to 24)                  |
| enc_in           | Encoder input size (defaults to 7)                           |
| dec_in           | Decoder input size (defaults to 7)                           |
| c_out            | Output size (defaults to 7)                                  |
| d_model          | Dimension of model (defaults to 512)                         |
| n_heads          | Num of heads (defaults to 8)                                 |
| e_layers         | Num of encoder layers (defaults to 2)                        |
| d_layers         | Num of decoder layers (defaults to 1)                        |
| s_layers         | Num of stack encoder layers (defaults to `3,2,1`)            |
| d_ff             | Dimension of fcn (defaults to 2048)                          |
| factor           | Probsparse attn factor (defaults to 5)                       |
| padding          | Padding type(defaults to 0).                                 |
| distil           | Whether to use distilling in encoder, using this argument means not using distilling (defaults to `True`) |
| dropout          | The probability of dropout (defaults to 0.05)                |
| attn             | Attention used in encoder (defaults to `prob`). This can be set to `prob` (informer), `full` (transformer) |
| embed            | Time features encoding (defaults to `timeF`). This can be set to `timeF`, `fixed`, `learned` |
| activation       | Activation function (defaults to `gelu`)                     |
| output_attention | Whether to output attention in encoder, using this argument means outputing attention (defaults to `False`) |
| do_predict       | Whether to predict unseen future data, using this argument means making predictions (defaults to `False`) |
| mix              | Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to `True`) |
| cols             | Certain cols from the data files as the input features       |
| num_workers      | The num_works of Data loader (defaults to 0)                 |
| itr              | Experiments times (defaults to 2)                            |
| train_epochs     | Train epochs (defaults to 6)                                 |
| batch_size       | The batch size of training input data (defaults to 32)       |
| patience         | Early stopping patience (defaults to 3)                      |
| learning_rate    | Optimizer learning rate (defaults to 0.0001)                 |
| des              | Experiment description (defaults to `test`)                  |
| loss             | Loss function (defaults to `mse`)                            |
| lradj            | Ways to adjust the learning rate (defaults to `type1`)       |
| use_amp          | Whether to use automatic mixed precision training, using this argument means using amp (defaults to `False`) |
| inverse          | Whether to inverse output data, using this argument means inversing output data (defaults to `False`) |
| use_gpu          | Whether to use gpu (defaults to `True`)                      |
| gpu              | The gpu no, used for training and inference (defaults to 0)  |
| use_multi_gpu    | Whether to use multiple gpus, using this argument means using mulitple gpus (defaults to `False`) |
| devices          | Device ids of multile gpus (defaults to `0,1,2,3`)           |

### 

## 结果

所有MSE,MAE对比结果为在对数据标准化后进行的。由于算力有限，预测序列较长的任务目前还未完成。
<img width="1242" alt="image" src="https://github.com/wech1228/ECL-forecast/assets/89915228/b5c9ab0d-d8f8-4d15-a054-8cfe7a090398">


