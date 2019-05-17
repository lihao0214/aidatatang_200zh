# 基于Kaldi的aidatatang_200zh的训练之葵花宝典

## 说明

- 测试环境：Ubuntu 16.04
- 说明：此示例为基于数据堂开源中文普通话语料集aidatatang_200zh发布的语音识别基准实验
- 文档更新时间：20190510
- 语音识别准确率：94.41%

## Kaldi简介

Kaldi是一款使用C++编写的免费的、开源的语音识别工具箱，具有Apache v2.0的开源协议。Kaldi工具箱的特点在于其实现了现代灵活的代码，从而易于人们修改和扩展。它在内部集成了有限状态转换器（Finite State Transducers ，FSTs），并提供了广泛的线性代数支持和完整的语音识别示例脚本，这为语音识别技术的开发和学习创造了良好的条件。

Kaldi的简要介绍请访问[这里](http://kaldi-asr.org/)

Kaldi的官方文档请访问[这里](http://kaldi-asr.org/doc/)

Kaldi的代码及版本控制请访问[这里]( https://github.com/kaldi-asr/kaldi)

## Kaldi安装

**安装步骤**

正式安装之前需要下载相关依赖包：

```shell
sudo apt-get install autoconf automaker gcc g++ libtool subversion gawk
sudo apt-get install libatlas-dev libatlas-base-dev gfortran zlib1g-dev
```

下载最新的Kaldi工具包，进入Kaldi安装路径，根据目录下INSTALL文件的说明安装Kaldi工具箱。

```shell
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi
```

首先，进入tools目录，按照tools/INSTALL的说明执行相应步骤。

在此之前，需要进行依赖项检查工作：

```shell
cd tools
./extras/check_dependencies.sh
```

如果检查结果没有问题，则进入编译环节。首先输入以下命令查看所使用电脑的CPU核数：

```shell
nproc
```

假设输出为4，则编译时可设置参数为4，加快编译速度：

```shell
make -j 4
```

注意，目前的Kaldi版本中已经不再默认安装IRSTLM，如有需求，可自行安装，如下所示：

```shell
./extras/install_irstlm.sh 
```

其次，进入src目录，按照src/INSTALL的说明在命令行分别输入：

```shell
 ./configure --shared
  make depend -j 4
  make -j 4
```

此阶段是编译阶段，将下载的包编译为可执行文件，耗时较长，请耐心等待。

上述说明仅适用于类UNIX系统，对于Windows系统的环境编译请参考windows/INSTALL文件。

**Kaldi中各文件解释**

- /egs：基于不同语料的实验示例脚本
- /tools：存放安装Kaldi所用的依赖库
- /src：存放源代码和可执行文件

**Kaldi安装验证**

验证Kaldi工具箱是否安装成功，可运行egs/yesno/示例，运行如下所示命令：

```shell
cd egs/yesno/s5
./run.sh
```

若顺利得到识别结果，恭喜你，Kaldi工具包已经安装成功，可以正式开展基于Kaldi工具包的语音识别研究了！

## aidatatang_200zh脚本训练

### 总体概览

进入egs/aidatatang_200zh/目录，可看到此处含有README文件和s5文件夹。

```shell
cd egs/aidatatang_200zh/
```

README文档是对aidatatang_200语料集的内容和下载方式的概述，s5目录包含了与基于该语料集进行语音识别实验相关的所有配置文件和脚本。

进入s5目录，可看到多个文件夹及文件，它们的简要介绍如下所示：

| 文件名    | 说明          | 备注                       |
| --------- | ------------- | -------------------------- |
| **conf**  | 配置目录      | 特征配置文件               |
| **local** | 脚本目录      | 特定工程所需脚本           |
| **steps** | 脚本目录      | Kaldi提供的数据处理工具    |
| **utils** | 脚本目录      | Kaldi提供的模型工具        |
| cmd.sh    | 硬件配置      | 单机集群配置               |
| path.sh   | 环境变量配置  | 导入环境变量               |
| run.sh    | 总脚本        | 总工程运行脚本             |
| RESULTS   | 实验结果(CER) | 罗列所有训练模型的识别结果 |

即：s5/conf里面是模型训练用到的配置文件。

​	s5/{local, steps, utils}里面则是run.sh所要用到的脚本文件。



在正式开始运行脚本之前，首先确认你想要在单机/服务器还是集群跑实验，如果是单机/服务器，则须修改cmd.sh的配置，如下所示：

```shell
cd s5
vim cmd.sh
### 将 queue.pl 改成 run.pl
```

接着，打开总工程脚本，修改数据集保存路径，即：

```shell
vim run.sh
### 将 data=/export/corpora/aidatatang 替换成你想保存数据集的路径
```

最后，运行此脚本，即可得到如RESULTS所示的基于aidatatang_200zh语料集的语音识别效果：

```shell
./run.sh
```

注：建议逐行执行run.sh脚本，以便清楚了解每个阶段的过程。

### 训练详解

理论上来说，只要相关环境配置正确，运行run.sh脚本即可完成整个训练步骤。然而，为了便于更好地理解语音识别的训练过程，有必要明白训练过程中的每个细节和步骤。在该示例脚本中，共包括数据下载、数据准备、GMM-HMM模型训练以及DNN-HMM模型训练四个阶段。下面将逐行详细介绍每个阶段所执行的任务。

**环境配置**

首先，在开展语音识别系统训练之前，需要激活必要的环境变量，分别是path.sh中声明的Kaldi运行环境变量和cmd.sh中声明的单机集群配置，如下所示：

```shell
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
```

**下载与解压数据集**

其次，在语音识别系统的开始阶段是数据的预处理步骤，该步骤将从指定地址下载数据堂开源中文语料集aidatatang_200zh数据集，并存放在指定目录下。

```shell
# corpus directory and download URL
data=/export/corpora/aidatatang/
data_url=www.openslr.org/resources/62

# You can obtain the database by uncommting the follwing lines
[ -d $data ] || mkdir -p $data || exit 1;
local/download_and_untar.sh $data $data_url data_aidatatang || exit 1;
```

**数据准备**

当成功下载数据集之后，则开始数据准备阶段。

第一步，为运行Kaldi程序准备相关映射文件，包括：

text：包含了每条语音对应的转录文本，格式为 ＜utterance-id＞＜transcription＞.

wav.scp：记录语音位置的索引文件，格式为＜utterance-id＞＜filename＞.

utt2spk：指明语音与说话人的对应关系，格式为＜utterance-id＞＜speaker-id＞.

spk2utt：指明说话人与语音的对应关系，可由utt2spk生成，格式为 ＜speaker-id＞＜utterance-id＞.

```shell
# Data Preparation: generate text, wav.scp, utt2spk, spk2utt
local/aidatatang_data_prep.sh $data/corpus $data/transcript || exit 1;
```

该命令执行后将会在当前目录下产生新的文件夹，可通过查看data/{local}{}/{train,dev,test}目录下的文件检查相关映射文件。

**准备词典**

第二步，产生包含语料集中所有转录文本的词典。

这里通过下载和合并CMU英文词典和CEDIT中-英词典生成中英文大词典。

```shell
# Lexicon Preparation: build a large lexicon that invovles words in both the training and decoding
local/aidatatang_prepare_dict.sh || exit 1;
```

生成词典的过程文件保存在data/local/dict文件夹下，最终生成的词典存放在data/local/dict/lexicon.txt中。

**准备语言模型**

第三步，根据语料集中的转录文本为语音识别训练准备语言模型。该过程分为以下两个阶段：

- 准备三音子模型的决策树问题集合以及编译L.fst，其中L.fst用于将音素序列映射成词语序列。

```shell
# Prepare Language Stuff
# Phone Sets, questions, L compilation
utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
```

- 训练三元文法语言模型，生成G.fst，并拼接L.fst与G.fst生成LG.fst。其中LG.fst可根据三元文法模型找出给定音素序列最有可能对应的词语序列。

```shell
# LM training
local/aidatatang_train_lms2.sh || exit 1;
# G compilation, check LG composition
local/aidatatang_format_data.sh
```

该阶段产生的过程文件保存在data/lang和data/lang_test目录下，可进一步检查相关模型。

至此，数据准备阶段的工作已经结束，下面将继续开展基于aidatatang_200zh语料集的GMM-HMM模型训练。

**提取特征**

在训练GMM-HMM模型之前，首先需要对原始的语音数据提取特征。这里我们提取的是MFCC加pitch特征。

```shell
# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train dev test; do
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done
```

生成的特征文件保存在mfcc目录下，计算过程的日志文件则保存在exp/make_mfcc目录下。

进入mfcc目录，可看到多个包含特征的.ark文件以及包含特征索引的.scp文件。程序默认生成的是二进制特征文件，可采用如下指令将其转成普通文本格式：

```shell
copy-feats ark:raw_mfcc_pitch_dev.1.ark ark,t:raw_mfcc_pitch_dev.1.txt
```

**GMM-HMM模型训练**

该阶段分为训练单音子隐马尔科夫模型（train_mono.sh）、训练与上下文相关的三音子模型（train_deltas.sh）、训练进行线性判别分析和最大似然线性变换的三音子模型（train_lda_mllt.sh）、训练进行发音人自适应（train_sat.sh）与基于特征空间最大似然线性回归对齐（align_fmllr.sh）的三音子模型共5个阶段。

下面以训练单音子隐马尔科夫模型为例，详细讲解训练的过程。

- 根据训练数据和语言模型训练声学模型：

```shell
steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/mono || exit 1;
```

训练的最终模型为exp/mono/final.mdl，里面保存了GMM模型的参数，使用下面命令可以查看模型的内容：

```shell
gmm-copy --binary=false exp/mono/final.mdl - | less
```

- 构建单音子解码图：

```shell
# Monophone decoding
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
```

该程序主要生成了HCLG.fst和words.txt文件，在后续的训练阶段起到了关键的作用。

- 解码：分别针对开发集和测试集解码

```shell
  # Monophone decoding
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/dev exp/mono/decode_dev

steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/test exp/mono/decode_test
```

解码的结果和日志保存在exp/mono/decode_dev和exp/mono/decode_test目录下。

- Veterbi对齐

```shell
# Get alignments from monophone system.
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/mono exp/mono_ali || exit 1;
```

此阶段对训练数据利用已生成模型重新对齐，以作为新模型的输入，对齐结果保存在exp/mono_ali目录。

至此，单音子训练过程结束，以相似的流程继续GMM-HMM模型的其他训练阶段，生成声学模型和解码结果，保存在exp目录中。

**DNN-HMM模型训练**

在整体的训练过程中，通过 GMM-HMM 模型能够得到 DNN 声学模型的输出结果，即：DNN 的训练是依赖于 GMM-HMM 模型的，因此训练一个好的 GMM-HMM 模型是提升语音识别效果的关键之一。另一方面，随着深度学习的发展，DNN模型展现了出了明显超越GMM模型的性能，替代了GMM进行HMM状态建模。

在本示例中，Kaldi nnet3中的TDNN时延神经网络模型和Chain链式模型被用来对数据集进行训练和解码，如下所示：

```shell
# nnet3
local/nnet3/run_tdnn.sh
```

```shell
# chain
local/chain/run_tdnn.sh
```



**查看结果**

```shell
# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
```



### 总结

通过运行以上GMM-HMM混合模型和DNN-HMM混合模型，基于数据堂开源中文普通话语料集aidatatang_200zh的语音识别基准实验已经完美完成，其字识别准确率如下所示：

<table>
   <tr>
      <td colspan=6 align="center">GMM-HMM(%)</td>
      <td colspan=2 align="center">DNN-HMM(%)</td>
   </tr>
   <tr>
      <td>mono</td>
      <td>Tri1</td>
      <td>Tri2</td>
      <td>Tri3a</td>
      <td>Tri4a</td>
      <td>Tri5a</td>
      <td>TDNN</td>
      <td>Chain</td>
   </tr>  
   <tr>
      <td>62.91</td>
      <td>82.02</td>
      <td>82.06</td>
      <td>82.74</td>
      <td>85.84</td>
      <td>87.78</td>
      <td>92.86</td>
      <td>94.41</td>
   </tr>
</table>

其中，

- mono指进行单音子隐马尔科夫模型训练的结果；

- Tri1指以mono模型为输入进行与上下文相关的三音子模型训练的结果；

- Tri2指以Tri1模型为输入进行与上下文相关的三音子模型训练的结果；

- Tri3a指进行线性判别分析和最大似然线性变换的三音子模型训练的结果；

- Tri4a指进行发音人自适应训练的结果；

- Tri5a指进行更深度发音人自适应训练的结果；
- TDNN指进行TDNN-HMM训练的结果；
- Chain指利用Chain模型训练的结果；

据此基准实验的识别效果显示，经过几轮的GMM-HMM模型的训练，基于aidatatang_200zh语料集的语音识别准确率可以达到87.78%，在此基础上进行DNN-HMM模型训练之后，语音识别准确率可高达94.41%。这也说明，由数据堂发布的包含600位来自中国不同地区的说话人、总计200小时时长共237265条语音、经过人工精心标注的中文普通话语料集可以对中文语音识别研究提供良好的数据支持。

尽管语音技术的发展具有一定的历史，深度学习在语音识别等研究领域的成果也极大地促进了语音技术的发展，但数据依然是目前语音技术存在局限性的原因之一，此数据集的开源恰恰丰富了国内的中文普通话语料集，为广大语音技术爱好者提供优质的数据资源。期待有兴趣的研究人员下载、使用aidatatang_200zh中文普通话语料集。
## 更多资源

数据堂是一家专业的人工智能数据服务提供商，致力于为全球人工智能企业提供数据获取、处理及数据产品服务，覆盖语音、图像、文本等数据类型，涵盖生物识别、语音识别、自动驾驶、智能家居、智能制造、新零售、OCR场景、智能医疗、智能交通、智能安防、手机娱乐等领域。

- 获取更多开源数据集，请访问[这里](https://www.datatang.com/webfront/opensource.html)
- 了解更多商业数据集，请点击[这里](https://www.datatang.com/webfront/datatang_dataset.html)

面向高校及科研群体,数据堂将持续开源更多高质量商业数据集,帮助研究人员拓宽研究领域，丰富研究内容，加速迭代。敬请期待！
