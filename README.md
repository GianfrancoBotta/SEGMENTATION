# Segmentation and classification of cell nuclei from histopathological images

## Description

Detecting various types of cells characterizing the Tumor MicroEnvironment (TME) is vital for cancer prognosis and research. However, segmenting and classifying nuclei from microscopy images consumes precious time for pathologists and is prone to subjectivity errors. Therefore, Deep Learning offers a valuable solution to automate these tasks and limit the variability of the procedure. Despite this, cells within the TME are usually non-homogenously distributed, leading to a strong class imbalance.
To tackle these issues, we build upon the HoVer-Net model to perform the segmentation and classification of four TME cell types within microscopy images, proposing a novel composite loss function specifically tailored to address class imbalance. Indeed, incorporating the Asymmetric Unified Focal loss term allows for achieving mean Panoptic Quality (mPQ) and mean Dice Similarity Coefficient (mDSC) of 0.329 and 0.493, along with 0.768 binary DSC (bDSC). Notably, in the case of underrepresented classes, our approach yields improvements up to 7.4\% and 9\% for the PQ and DSC, respectively. Besides, we also try to employ the weighted cross-entropy loss function, which performs poorerly on average when compared to the original loss function of HoVer-Net. Finally, we show how comparable results can be obtained by analyzing the considered images' blue channel.

![Comparison of the different employed loss functions using two random patches](https://drive.google.com/file/d/1mTKW0v1N18YEsqA1Xvbnz8DGVJ7be1mS/view?usp=drive_link)

## Data

The data we used for our project is retrieved from the [MoNuSAC challenge of 2020][1]. The entire training dataset can be found [here](https://drive.google.com/uc?id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq), while the testing dataset can be found [here](https://drive.google.com/uc?id=1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ).
The data comprises a training dataset with the images and the masks (provided as .xml files) of four  types of cell (epithelial cells, lymphocytes, macrophages, and neutrophils), while the testing dataset contains also a fifth class of ambigous masks. In our project, we only consider the training dataset and we split it using 80/10/10 rule. For further details, see the MoNuSAC challenge [paper][1].

## Environment configuration

We choose to carry out our project in Python (version 3.8.10.), exploiting the [PyTorch][2] library to train our model. To train the network we use a 24GB NVIDIA RTX A5000 GPU. However, the code can be run both in Google Colab and on a local machine, since we want our analysis to be reproducible even for people that cannot have access to a private GPU.

## Code structure

1. Data exploration
   
   We explore the images in the dataset and we note a strong class imbalance within the dataset. In fact, epithelial cells and lymphocytes are much more numerous than macrophages and neutrophils. The exploratory data analysis can be found in the file "Dataset_analysis.ipynb".

2. Pre-processing and data augmentation

   We split the original training dataset from the MoNuSAC challenge in a training set corresponding to 80% of the patients, and in a validation and a test set comprising roughly 10% of the cases. Subsequently, we obtain the masks from the .xml files using the authors' code. Moreover, we divide the images into patches of dimension 256x256 pixels and using a step of 200 both horizontally and vertically, maintaining the use of mirror padding. In this context, we customize the normalization approach by performing it at a patch level, rather than at an image level, to better exploit the entire range of conversion. Finally, we obtain the horizontal-vertical and the binary masks from the original masks. To achieve these previous steps, we mostly rely on the content of the original [HoVer-Net][3] repository.
   As for the data augmentation step, we implement the data augmentation step exploiting [Albumentations][4].

4. Training

   Concerning the network architecture implementation we use a pre-configured model from the [PathML][5] library, as it provides the basis to develop a straightforward and efficient pipeline for applying HoVer-Net to our data. The provided pre-configured network is also easy to reconfigure to meet the specificities of our dataset. We train our models for 75 epochs, using the Adam optimizer with a starting
learning rate of 10−3, decreased to 10−4 after 40 epochs. The newly loss function we employ is the asymmetric unified focal loss, where we set the parameters as follows: γ = 0.5, δ = 0.6, λ = 0.5, as recommended [here][6].

5. Metrics

   To comprehensively evaluate our results, we exploit both the Dice score and the Panoptic quality. Furthermore, we provide also the binary Dice score, which only indicates the quality of the segmentation step. Note that two papers claim that both the [Dice score][7] and the [Panoptic quality][8] are not suitable to evaluate the accuracy of the network when the objects are cell nuclei. However, as for now, the community has not found a solution yet.

[1]: https://ieeexplore.ieee.org/document/9446924
[2]: https://arxiv.org/abs/1912.01703
[3]: https://arxiv.org/abs/1812.06499
[4]: https://arxiv.org/abs/1809.06839
[5]: https://www.medrxiv.org/content/10.1101/2021.07.07.21260138v1
[6]: https://www.sciencedirect.com/science/article/pii/S0895611121001750
[7]: https://arxiv.org/abs/1801.00868
[8]: https://www.nature.com/articles/s41598-023-35605-7
