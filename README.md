# SEGMENTATION

Detecting various types of cells characterizing the Tumor MicroEnvironment (TME) is vital for cancer prognosis and research. However, segmenting and classifying nuclei from microscopy images consumes precious time for pathologists and is prone to subjectivity errors. Therefore, Deep Learning offers a valuable solution to automate these tasks and limit the variability of the procedure. Despite this, cells within the TME are usually non-homogenously distributed, leading to a strong class imbalance.
To tackle these issues, we build upon the HoVer-Net model to perform the segmentation and classification of four TME cell types within microscopy images, proposing a novel composite loss function specifically tailored to address class imbalance. Indeed, incorporating the Asymmetric Unified Focal loss term allows for achieving mean Panoptic Quality (mPQ) and mean Dice Similarity Coefficient (mDSC) of 0.329 and 0.493, along with 0.768 binary DSC (bDSC). Notably, in the case of underrepresented classes, our approach yields improvements up to 7.4\% and 9\% for the PQ and DSC, respectively. Besides, we show how comparable results can be obtained by analyzing the considered images' blue channel.
