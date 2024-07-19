# Benchmarking 3D Point Cloud Classifiers in Synthetic to Real scenarios: A Cross-Domain Study with Qualitative Grad-CAM Insights


[\[Project Paper\]](paper_def.pdf)

Code for SemNov3Detection project -  Advanced Machine Learning course AY 2023/2024 - PoliTO

![Alt text](https://github.com/LucaFaieta/SemNov_AML_DAAI_23-24/blob/main/chairs.PNG)

> **Abstract: **"In recent years, significant progress has been made in the domain of 3D learning, particularly in the context of classification, detection and segmentation tasks on 3D point clouds. However, most existing literature at this time focuses on closed-set problems, while open-set classification models are much needed for their capability to reflect the open and dynamic nature of real-world environments, and they still suffer from a general lack of comprehensive and complete analysis. [...]
We propose: 1.) A quantitative comparison of two state-of-the-art models performance in a synthetic-to-real scenario, leveraging a number of different evaluation strategies, 2.) Results for the implementation of a large pre-trained model taken from OpenShape publication and 3.) Visual qualitative analysis of DGCNN using a version of GradCAM specifically adapted for 3D-based frameworks."*

This code could serve as a promising foundation for future work, implementing and adapting GradCAM for other state-of-the-art 3D models to obtain deeper understanding and potentially decisive insights into these models' inner workings.


## Installation

We perform our experiments with PyTorch 1.9.1+cu111 and Python 3.7. To install all the required packages simply run:

```
```
pip install -r requirements.txt
```
```

To install PointNet++ you also need to run:

```
```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```
```
N.B. system-wide CUDA version should match the PyTorch one

## Data 

All data used in our experiments can be downloaded as follows:
```
```
chmod +x download_data.sh
./download_data.sh
```
```

## Models Training
Here you can find code to train DGCNN or PointNet++ on synthetic dataset:
```
```
#example cell for training a model
!python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name Pointnet_SR2_cosine --src SR2 --loss cosine --wandb_proj AML_DAAI_proj23_24_Pointnet_SR2_cosine --batch_size 32 --ckpt_path /content/drive/MyDrive/SemNov_AML_DAAI_23-24/outputs/Pointnet_SR2_cosine/models/model_best.pth
```
```
You can train PointBERT checkpoints adding the  "--fine_tuning path/to/pointBERT/model" option to the previous command. Checkpoints can be downloaded cloning their repository. We tested the following:

```
```
git clone https://huggingface.co/OpenShape/openshape-pointbert-vitl14-rgb
```
```
```
```
git clone https://huggingface.co/OpenShape/openshape-pointbert-vitl14-rgb
```
```
```
```
git clone https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb
```
```
N.B. implementation of other PointBERT checkpoints from OpenShape publication is not currently plug and play. If you want to test them you should manually manage their implementation in the code.



## Evaluation
To evaluate a model you can run the following:

```
```
#cell example for evaluation 
!python /content/drive/MoyDrive/SemNov_AML_DAAI_23-24/classifiers/trainer_cla_md.py --config /content/drive/MyDrive/SemNov_AML_DAAI_23-24/cfgs/Pointbertb32.yaml --exp_name OpenShape_Pointbertb32_SR2 --src SR2 --loss CE --wandb_proj AML_DAAI_proj23_24_test --mode eval --batch_size 128

##eventually load checkpoint
-- ckpt_path /content/drive/MyDrive/SemNov_AML_DAAI_23-24/outputs/OpenShape_Pointbertb32_SR2/models/model_best.pth -
```
```

To save GradCAM plots (currently only for DGCNN) add "--gradcam path/to/your/plots" option to the previous script


## Acknowledgements
This work heavily relies on the code and findings from the 3DOS benchmark (Alliegro et al.) and the OpenShape publication. We provide links to their GitHub repositories below and extend our gratitude for their contributions.

- [3DOS Benchmark](https://github.com/antoalli/3D_OS)
-  [OpenShape Publication](https://github.com/Colin97/OpenShape_code)





