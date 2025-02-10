<div align="center">


# Awesome Vision Models for Time Series
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![Paper](https://img.shields.io/badge/paper-leave--red)](https://www.overleaf.com/project/677f07201833b1a0ba012653)
[![PyPI - Version](https://img.shields.io/pypi/v/version)](#package-info)
</div>


This repository tracks the latest paper on "Vision Model for Time series Analysis" and serves as the official repo for [Harnessing Vision Models for Time Series Analysis: A Survey](https://www.overleaf.com/project/677f07201833b1a0ba012653)

This repository is under active maintenance by [Ziming Zhao](https://zhziming.github.io/) from **D2I Group@UH** led by [Dr.JingChao Ni](https://nijingchao.github.io/).

***

### Contribution
Resulting from the the discrepancy between continuous time series and the discrete token space of LLMs, and the challenges in explicitly modeling the correlations of variates in MTS, research attentions have shifted to LVMs and VLMs for time series analysis. To fill the blank in the existing literature, this survey discusses the advantages of vision models over LLMs in time series analysis.

|[<img src="./fig/structure.png" width="500"/>](./fig/structure.png) |[<img src="./fig/Pipeline.png" width="3000"/>](./fig/Pipeline.png)|
|:--:|:--:| 
|Figure 1: The general process of leveraging vision models for time series analysis|Figure 2: Illustration of different modeling strategies on imaged time series |

The overall structure of our survey follows the general process of applying vision models for time series analysis as delineated in Figure 1. Based on the proposed dual view taxonomy, primary imaging methods on time series and imaged modelling solutions shown in Figure 2, are reviewed in this survey, followed by the discussion including pre- & post- processing for time series recovery and future directions in this promising field.

***
### Taxonomy
Taxonomy are proposed as a dual view of *Time Series to Image Transformation* and *Imaged Time Series Modeling*. For the former, primary methods for imaging UTS or MTS are described and remarked on their pros and cons. For the latter, the existing methods are classified by conventional vision models, LVMs and LMMs. 

#### Image Transformation of Time Series

<sub>The top panel includes single-modal models. The bottom panel includes multi-modal models. TS-Type denotes type of time series. TS-Recover denotes recovering time series from predicted images. </sub> $*$<sub>: the method has been used to model the individual UTSs of an MTS. </sub>$^{\natural}$ <sub>: a new pre-trained model was proposed in the work.</sub> $^{\flat}$ <sub>: when pre-trained models were unused, Fine-tune refers to train a task-specific model from scratch. </sub>



Method|TS-Type|Imaging|Multimodal|Model|Pre-trained|Fine-tune|Prompt|TS-Recover|Task|Domain
:-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:--:|
<sub>Time series classification using compression distance of recurrence plots [[paper]()] </sub>|UTS|RP|✘|K-NN|✘|✘|✘|✘|Classification|General
<sub>Encoding time series as images for visual inspection and classification using tiled convolutional neural networks [[paper]()] </sub>
<sub>Imaging time-series to improve classification and imputation [[paper]()]</sub>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>
<small></small>




***
### Package Info

We have uploaded our code to package to PyPI, run the following command for installation.

```bash
pip install time2img
```

Our code compatible with all common benchmarks found in [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). You can run [example](./src/main.py) to reproduce our illustration of different time series imaging methods from our paper. 

![Time Series Imaging](./fig/image_plot.png)


***
### Citation