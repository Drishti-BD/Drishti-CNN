# [Drishti Convolutional Network](https://arxiv.org/abs/2108.04358)

This repository contains the code and algorithm for the paper ["Convolutional Nets for Diabetic Retinopathy Screening in Bangladeshi Patients"](https://arxiv.org/abs/2108.04358) by Ayaan Haque, Ipsita Sutradhar, Mahziba Rahman, Mehedi Hasan, and Malabika Sarker from Saratoga High School and BRAC University.

This paper is a validation study on the algorithm used for [Drishti](https://drishtiai.org/docs.html), an organization and application that provides free and accurate screening of Diabetic Retinopathy for rural areas of Bangladesh. The paper shows the effectiveness of our algorithm on performing stage-based classification of DR on Bangladeshi retinal scans collected from hospitals and field studies.

The paper is available on ArXiv [here](https://arxiv.org/abs/2108.04358) and is in review for a double-blind journal.

## Abstract

Diabetes is one of the most prevalent chronic diseases in Bangladesh, and as a result, Diabetic Retinopathy (DR) is widespread in the population. DR, an eye illness caused by diabetes, can lead to blindness if it is not identified and treated in its early stages. Unfortunately, diagnosis of DR requires medically trained professionals, but Bangladesh has limited specialists in comparison to its population. Moreover, the screening process is often expensive, prohibiting many from receiving timely and proper diagnosis. To address the problem, we introduce a deep learning algorithm which screens for different stages of DR. We use a state-of-the-art CNN architecture to diagnose patients based on retinal fundus imagery. This paper is an <i>experimental evaluation</i> of the algorithm we developed for DR diagnosis and screening specifically for Bangladeshi patients. We perform this validation study using separate pools of retinal image data of real patients from a hospital and field studies in Bangladesh. Our results show that the algorithm is effective at screening Bangladeshi eyes even when trained on a public dataset which is out of domain, and can accurately determine the stage of DR as well, achieving an overall accuracy of 92.27\% and 93.02\% on two validation sets of Bangladeshi eyes. The results confirm the ability of the algorithm to be used in real clinical settings and applications due to its high accuracy and classwise metrics. Our algorithm is implemented in the application <a href="https://drishtiai.org/"><i>Drishti</i></a>, which is used to screen for DR in patients living in rural areas in Bangladesh, where access to professional screening is limited.

## Data

Our training data is publically available [here](https://www.kaggle.com/c/aptos2019-blindness-detection/data). Our, run the following command:

```
kaggle competitions download -c aptos2019-blindness-detection
```

Once the data is downloaded, the file paths will be correct for the provided code.

Our validation data is confidential and cannot be made public.

## Method

![](https://github.com/ayaanzhaque/Drishti-CNN/blob/main/images/model_fig.png?raw=true)

We use a DenseNet-121 CNN as the base network. Importantly, we do not use a pre-trained version of the DenseNet-121 and rather train entirely with randomized parameters, as we found this increased the efficiency of training and deployment. The base DenseNet with the output layer removed is used as the primary architecture of our network. A 2D Global Average Pooling is then added after the DenseNet module, followed by a Dropout layer with a dropout rate of 0.5. For the output layer, a dense layer with 5 output nodes and a sigmoid activation function are used.

## Results

A brief summary of our results are shown below. The table shows the overall Accuracy and metrics for our performance on the two validation sets of Bangladeshi Eyes.

![](https://github.com/ayaanzhaque/Drishti-CNN/blob/main/images/results.png?raw=true)

## Code

To train the CNN, use the [```train.py```](https://github.com/ayaanzhaque/Drishti-CNN/blob/main/train.py) file and run it with the following command.

```
$ python train.py
```

All the data augmentation parameters and training hyperparameters are set. The DenseNet is not pre-trained, but in the DenseNet module, it can be modified to be pre-trained. For just the CNN code, use the [```CNN.py```](https://github.com/ayaanzhaque/Drishti-CNN/blob/main/CNN.py) file.

## Citation

If you find this repo or the paper useful, please cite:

```
@article{haque2021convolutional,
  title={Convolutional Nets for Diabetic Retinopathy Screening in Bangladeshi Patients},
  author={Haque, Ayaan and Sutradhar, Ipsita and Rahman, Mahziba and Hasan, Mehedi and Sarker, Malabika},
  journal={arXiv preprint arXiv:2108.04358},
  year={2021}
}
```
