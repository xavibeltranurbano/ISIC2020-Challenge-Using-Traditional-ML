# CAD Project: A SKIN LESION CLASSIFICATION APPROACH WITH MACHINE LEARNING

Contributors:

Xavi Beltran Urbano, Muhammad Zain Amin

Program Files
============

```
.
├── dataset               # Not available in the git repository due to large file size.
├── preprocessing.py       
├── featureExtraction.py           
├── training.py
├── utils.py
├── main_Binary.py
└── main_Multiclass.py

```
# Dependencies

Used Python programming language for the code.
Everything is implemented in the pycharm which will hopefully make it easier to understand the code.

1) Numpy
2) Matplotlib
3) Sklearn
4) Python
5) Pandas

Project Methodology and Result
============
The proposed approach for skin lesion classification using machine learning consists of four major steps, which are image pre-processing, data sampling, feature extraction and, classification. Sklearn library was used for obtaining classification scores using a 5-fold cross-validation for both binary and multiclass classification tasks. The SMOTE approach helps as it is robust to class imbalance.

![image](https://github.com/xavibeltranurbano/ISIC-Challenge-A-Conventional-Skin-Lesion-Classification-Approach/assets/21214562/03164afe-964d-4b18-83ab-34d0b4354631)
![image](https://github.com/xavibeltranurbano/ISIC-Challenge-A-Conventional-Skin-Lesion-Classification-Approach/assets/21214562/fd3f0802-aecc-4005-a6dd-38170b0952cb)

Our proposed classification approach was implemented and evaluated on the ISIC Skin Lesion dataset, containing melanoma, bcc, and scc lesions. Various features were extracted from preprocessed images and used for training the machine learning algorithms. In general, the proposed CAD system achieves promising results in both the binary and multiclass classification tasks. The results are given below:-

![image](https://github.com/xavibeltranurbano/ISIC-Challenge-A-Conventional-Skin-Lesion-Classification-Approach/assets/21214562/97a73acd-ea51-4d6e-b085-91061b54fbdb)


## References
[1] https://github.com/hl0d0w1g/skin_lesion_segmentation/blob/main/segmentation.py

[2] Li, W., Joseph Raj, A. N., Tjahjadi, T., & Zhuang, Z. (2021). Digital hair removal by deep learning for skin lesion segmentation. Pattern Recognition, 117, 107994. https://doi.org/10.1016/j.patcog.2021.107994

[3] https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

[4] https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html

[5] C. Barata, M. E. Celebi and J. S. Marques, "Improving Dermoscopy Image Classification Using Color Constancy," in IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 3, pp. 1146-1152, May 2015, doi: 10.1109/JBHI.2014.2336473.

[6] https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation/blob/master/color_constancy.py

[7] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

[8] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

[9] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

[10] https://xgboost.readthedocs.io/en/stable/python/python_intro.html
