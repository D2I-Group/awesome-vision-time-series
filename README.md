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
Resulted from the the discrepancy between continuous time series and the discrete token space of LLMs, and the challenges in explicitly modeling the correlations of variates in MTS, shifted research attentions to LVMs and VLMs. To fill the blank in the existing literature, this survey discusses the advantages of vision models over LLMs in time series analysis.

|[<img src="./fig/s.png" width="50"/>](./fig/structure.png) |[<img src="./fig/Pipeline.png" width="1000"/>](./fig/Pipeline.png)|
|:--:|:--:| 
| *Figure 1: The Framework of Our Survey* | *Figure 2: Categorization of Component Design for Fine-tuning Time Series LLMs* |

Left: The overall structure of our survey follows the general process of applying vision models for time series analysis, as delineated in Fig 1. 


Taxonomy are proposed as a dual view of *Time Series to Image Transformation* and *Imaged Time Series Modeling*. For the former, primary methods for imaging UTS or MTS are described and remarked on their pros and cons. For the latter, the existing methods are classified by conventional vision models, LVMs and LMMs. 



***
### Package Info

We have uploaded our code to package to PyPI, run the following command for installation.

```bash
pip install time2img
```

Our code compatible with all common benchmarks found in [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). You can run [example](./src/main.py) to reproduce our illustration of different time series imaging methods from our paper. 

![Time Series Imaging](./fig/image_plot.png)