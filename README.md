# AnomalyCLIP (Train once and test other)
> [**ICLR 24**] [**AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**](https://arxiv.org/pdf/2310.18961.pdf)
>
> by [Qihang Zhou*](), [Guansong Pang*](https://www.guansongpang.com/),  [Yu Tian](https://yutianyt.com/), [Shibo He](https://scholar.google.com/citations?hl=zh-CN&user=5GOcb4gAAAAJ&view_op=list_works&sortby=pubdate), [Jiming Chen](https://scholar.google.com/citations?user=zK9tvo8AAAAJ&hl=zh-CN).


## Updates

- **04.13.2025**: We fixed the bug of identical local outputs from multiple intermediate feature maps. We updated the illustration of local features, as raised in issue 18, and the adaptation of the text encoder, as discussed in issue 75, in the new arXiv version (**These updates do not affect the performance of AnomalyCLIP**). We thank the community, especially Jonah Weiss and fangfangzk, for their efforts in improving AnomalyCLIP.
- **03.19.2024**: Code has been released !!!
- **08.08.2024**: Update the code for testing one image.

## Introduction 
Zero-shot anomaly detection (ZSAD) requires detection models trained using auxiliary data to detect anomalies without any training sample in a target dataset. It is a crucial task when training data is not accessible due to various concerns, e.g., data privacy, yet it is challenging since the models need to generalize to anomalies across different domains where the appearance of foreground objects, abnormal regions, and background features, such as defects/tumors on different products/organs, can vary significantly. Recently large pre-trained vision-language models (VLMs), such as CLIP,
have demonstrated strong zero-shot recognition ability in various vision tasks, including anomaly detection. However, their ZSAD performance is weak since the VLMs focus more on modeling the class semantics of the foreground objects rather than the abnormality/normality in the images.
In this paper we introduce a novel approach, namely AnomalyCLIP, to adapt CLIP for accurate ZSAD across different domains. The key insight of AnomalyCLIP is to learn object-agnostic text prompts that capture generic normality and abnormality in an image regardless of its foreground objects. This allows our model to focus on the abnormal image regions rather than the object semantics, enabling generalized normality and abnormality recognition on diverse types of objects. Large-scale experiments on 17 real-world anomaly detection datasets show that AnomalyCLIP achieves superior zero-shot performance of detecting and segmenting anomalies in datasets of highly diverse class semantics from various defect inspection and medical imaging domains. All experiments are conducted in PyTorch-2.0.0 with a single NVIDIA RTX 3090 24GB. 

## Overview of AnomalyCLIP
![overview](https://github.com/zqhang/AnomalyCLIP/blob/main/assets/overview.png)


## Analysis of different text prompt templates
![analysis](./assets/analysis.png) 


## How to Run
### Prepare your dataset
Download the dataset below:

* Industrial Domain:
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip), [SDD](https://www.vicos.si/resources/kolektorsdd/), [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection), [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)

* Medical Domain:
[HeadCT](https://drive.google.com/file/d/1lSAUkgZXUFwTqyexS8km4ZZ3hW89i5aS/view?usp=sharing), [BrainMRI](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), [COVID-19](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), [ISIC](https://drive.google.com/file/d/1UeuKgF1QYfT1jTlYHjxKB3tRjrFHfFDR/view?usp=sharing), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Endo](https://drive.google.com/file/d/1LNpLkv5ZlEUzr_RPN5rdOHaqk0SkZa3m/view), [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation?tab=readme-ov-file).

* Google Drive link (frequently requested dataset): [SDD](https://drive.google.com/drive/folders/1oqaxUZYi44jlLT4WtT6D5T6onPTNZXsu?usp=drive_link), [Br35H](https://drive.google.com/file/d/1l9XODMBm4X23K70LtpxAxgoaBbNzr4Nc/view?usp=drive_link), [COVID-19](https://drive.google.com/file/d/1ECwI8DJmhEtcVHatxCAdFqnSmXs35WFL/view?usp=drive_link)
### Generate the dataset JSON
Take MVTec AD for example (With multiple anomaly categories)

Structure of MVTec Folder:
```
mvtec/
│
├── meta.json
│
├── bottle/
│   ├── ground_truth/
│   │   ├── broken_large/
│   │   │   └── 000_mask.png
|   |   |   └── ...
│   │   └── ...
│   └── test/
│       ├── broken_large/
│       │   └── 000.png
|       |   └── ...
│       └── ...
│   
└── ...
```

```bash
cd generate_dataset_json
python mvtec.py
```

Take SDD for example (With single anomaly category)

Structure of SDD Folder:
```
SDD/
│
├── electrical_commutators/
│   └── test/
│       ├── defect/
│       │   └── kos01_Part5_0.png
|       |   └── ...
│       └── good/
│           └── kos01_Part0_0.png
│           └── ...  
│
└── meta.json
```

```bash
cd generate_dataset_json
python SDD.py
```
Select the corresponding script and run it (we provide all scripts for datasets that AnomalyCLIP reported). The generated JSON stores all the information that AnomalyCLIP needs. 

### Custom dataset (optional)
1. Create a new JSON script in fold [generate_dataset_json](https://github.com/zqhang/AnomalyCLIP/tree/main/generate_dataset_json) according to the fold structure of your own datasets.
2. Add the related info of your dataset (i.e., dataset name and class names) in script [dataset\.py](https://github.com/zqhang/AnomalyCLIP/blob/main/dataset.py)

### Run AnomalyCLIP
* Quick start (use the pre-trained weights)
```bash
bash test.sh
```
  
* Train your own weights
```bash
bash train.sh
```


## Main results (We test all datasets by training once on MVTec AD. For MVTec AD, AnomalyCLIP is trained on VisA.)

### Industrial dataset
![industrial](./assets/Industrial.png) 


### Medical dataset
![medical](./assets/medical.png) 


## Visualization

![hazelnut](./assets/hazelnut.png) 

![capusle](./assets/capusle.png) 

![skin](./assets/skin.png) 

![brain](./assets/brain.png) 


## We provide the reproduction of WinCLIP [here](https://github.com/zqhang/WinCLIP-pytorch)


* We thank for the code repository: [open_clip](https://github.com/mlfoundations/open_clip), [DualCoOp](https://github.com/sunxm2357/DualCoOp), [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery), and [VAND](https://github.com/ByChelsea/VAND-APRIL-GAN/tree/master).

## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```
@inproceedings{zhou2023anomalyclip,
  title={AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
