# Intracranial Hemorrhage Classification
Computer-Aided Diagnosis Project: A version of the RSNA Intracranial Hemorrhage Detection

The presented project follows the 2019 Kaggle challenge entitled “RSNA Hemorrhage Detection: Identify acute intracranial hemorrhage and its subtypes”. The main goal was to develop an algorithm to detect acute intracranial hemorrhage in CT scans and classify it in its subtypes: epidural, intraparenchymal, intraventricular, subarachnoid, or subdural. Three methods were tested: two different CNNs, the VGG16 and ResNeXt-101, and a kNN classifier with CNN feature extraction. These methods were tested in two tasks: binary classification to detect the hemorrhage and multiclass classification to identify its subtypes. A final two-stage pipeline was proposed where the hemorrhage would be first detected and, if present, then classified. The results showed thatpreprocessing each of the DICOM images to a RGB PNG image,with relevant windows as each channel, improved the neural networks training and testing results. Besides that, the use of GradCAM showed some promising results regarding the location of certain types of hemorrhage. From the chosen metrics, the VGG16 model showed the best performance in the detection and classification task. Regarding the proposed two-stage pipeline, the results showed some improvement when compared with a simple multiclass classification. 

**Keywords** – Intracranial hemorrhage; multiclass classification; deep learning; k-nearest neighbors 

## Introduction 

Intracranial hemorrhage is a life-threatening condition that accounts for 15-30% of strokes. Bleeding inside the cranium can damage the brain tissue and even increase intracranial pressure, which further damages the brain itself. The treatment of this condition consists of stopping the bleeding and relieving the pressure. These actions must be done immediately to avoid irreversible damage [1]. 

The diagnosis is performed with a non-contrast computer tomography (CT) and the resulting medical images must be evaluated by specialists, to decide on the best course of treatment. However, this process can be complex and time-consuming. Therefore, automatically identifying the spot of a hemorrhage from a CT scan may help to speed the treatment. Additionally, the characterization of the hemorrhage subtype is essential to evaluate the possible damage and need for immediate surgery. Some features that help to distinguish the subtypes are location, shape, and proximity to other structures. Figure 1 shows five hemorrhage subtypes and their characteristics. [2]

![inbox_603584_56162e47358efd77010336a373beb0d2_subtypes-of-hemorrhage](https://user-images.githubusercontent.com/105933447/217233739-79d71a6e-bce3-43fe-829b-a7297ca09485.png)

Deep learning approaches are vastly used to perform this task in the available literature, with VGG16 and ResNeXt being two of the most promising and prominent networks. 

VGG16 is a deep convolutional neural network first introduced by K. Simonyan and A.Zisserman from the Visual Geometry Group of University of Oxford in the ImageNet Large Scale Visual Recognition Challenge 2014, where it achieved 92.7% top-5 test accuracy [3]. The architecture of this network is made of 16 convolutional layers distributed in 3 blocks, comprising 2 layers of 3x3 convolutions, followed by 2x2 max pooling, and 2 blocks containing 3 layers of 3x3 convolutions, followed by 2x2 max pooling, finishing with 2 fully connected layers of 4096 hidden layers each [4], as presented in Figure 2. 

![image](https://user-images.githubusercontent.com/105933447/217234181-0c67503c-ed5a-4a33-a440-f66f6ff18589.png)


Although it has about 138 million parameters, the processing of smaller receptive fields (area of the input image being analyzed by a node in the network), using 3x3 convolutions with stride 1, decreased the number of parameters when compared to other architectures, such as AlexNet [5]. In addition, after each convolution, the ReLU activation function is performed twice, making the decision function more discriminative [4].  

Other CNN largely used is ResNet, short for Residual Network, introduced in 2015 in [6]. This network was proposed to reduce model complexity while also keeping a good performance [7]. Typical ResNet models are implemented with skip connections that contain nonlinearities (ReLU) and batch normalization in between.  

ResNeXt is a homogeneous neural network that reduces the number of hyperparameters, suggested in [8]. This is achieved by the introduction of “cardinality”, an additional dimension, on top of the width and depth of ResNet, that defines the size of the set of transformations [9]. Cardinality is an essential dimension and increasing it is a more effective way of gaining accuracy than going deeper or wider [8]. The characteristics of ResNeXt are inherited from ResNet, VGG, and Inception, including shortcuts from the previous block to next block, stacking layers and adapting split-transform-merge strategy [10]. This neural network was used for the same referred challenge in [11] with promising results. 

![image](https://user-images.githubusercontent.com/105933447/217234358-a648446e-29a3-4f30-ad4d-8913301e0255.png)

Deep learning techniques can achieve remarkable performance in many tasks. However, the justification of the process decision is still a “black box”, making these techniques not explainable. Gradient-weighted Class Activation Mapping (GradCAM) is a method which provides visual explanations using gradient-based locations [12]. It is applied to an already-trained neural network, using the feature maps produced by a convolutional layer, with authors affirming that the last ones have the best compromise between high-level semantics and spatial information. The output of GradCAM is a heatmap where the relevant parts of the image for the model to be able to perform classification are highlighted. 

On the other hand, the combination of a neural network to extract relevant features and simpler classifiers has been explored in several applications, including studies for medical images [13], [14] and even for intracranial hemorrhage detection [15]. 

This project follows the 2019 Kaggle challenge entitled “RSNA Hemorrhage Detection: Identify acute intracranial hemorrhage and its subtypes”. The main goal was to develop an algorithm to detect the presence of acute intracranial hemorrhage in CT scans. Additionally, the model should be able to classify the hemorrhages according to its subtypes: epidural, intraparenchymal, intraventricular, subarachnoid, or subdural and, ideally, provide some insight about the possible location of the hemorrhage. 

## Methods 

### Data Pre-Processing  

The dataset consists of CT scans provided by the Radiological Society of North America in collaboration with members of the American Society of Neuroradiology and MD.ai. Since this dataset was extremely large, and due to limitations in space of our resources, a smaller version of the same dataset [16] was used.  

The smaller dataset contained 4930 images with detected hemorrhages and around 30 000 images without detected hemorrhages. For this project, only the first 5001 images from the “no” group were selected to balance the dataset and having into consideration the restraints of usage of the Google Drive/Collab. From the 4930 images with detected hemorrhages, the classification information was extracted from the original set. The subtype labels include: epidural, intraparenchymal, intraventricular, subarachnoid, subdural and any, the latter being always true if any of the subtype labels is true.  According to the provided classification data, an image can have more than one subtype label associated (therefore, being referred to as “multi label”).  

The initial dataset distribution is presented in Figure 4, with 9931 images in total, 5001 with no hemorrhage detected and 4930 with one or more types of hemorrhages. 
Figure 5 describes the distribution of the hemorrhage subtypes in this dataset. In the same figure, the number of images with multiple labels is also presented. For simplification purposes, only the single label images were used in this project, making the final task multiclass classification. In conclusion, the final dataset had 5001 images with no hemorrhage and 3496 images presenting hemorrhage of a single subtype, with the number of samples for each label being presented also in Figure 5. 

![image](https://user-images.githubusercontent.com/105933447/217234940-a5b5ee9c-e18b-4252-9351-1a5f4d0d35f4.png)

Medical CT images are usually stored in DICOM format, containing metadata alongside pixel data, which was the case of the RSNA challenge. The dataset images had 65 536 levels of gray (16-bit images), where each pixel’s value represents the density of the tissue in Hounsfield units (HU) [17], a relative quantitative measurement of radio density. 

In a CT scan, the absorption/attenuation of the X-ray beam is directly proportional to physical density of tissue. Therefore, the HU is defined as the linear transformation of the baseline linear attenuation coefficient of the X-ray beam, considering distilled water at standard temperature and pressure conditions 0 HU and air -1000 HU [18].  

Since monitors can only display 256 levels of gray, radiologists usually use software to focus on specialized intensity windows of HU to try to identify a specific pathology. These windows are characterized by a center (L) and a width (W) that specifies a linear conversion from HU to pixel values to be displayed, with the upper and lower gray levels (UGL and LGL, respectively) being calculated using the following system of equations: 

 ![image](https://user-images.githubusercontent.com/105933447/217235206-c256a704-04ee-40c2-b172-0f1ba1015666.png)


where the values above UGL are presented as white and all the values below LGL are presented as black. The following windows are the ones usually focused by radiologists: 

* Brain window – W:80 L:40 
* Subdural window – W:200 L:80 
* Bone window – W:2000 L:600 

For our project, we decided to extract each of the three windows previously referred and build an RGB image where each channel corresponded to one of those windows, as shown in Figure 6, since it demonstrated improvement in the classification of brain hemorrhage images [17], [19]. Those images were saved as PNG and used as input to our pipeline. 

![image](https://user-images.githubusercontent.com/105933447/217236074-6dcf85b4-7560-4f08-bb14-05265ef179d2.png)

## Pipeline Proposal 

To separate the steps of identifying the presence of a hemorrhage and, if positive, its subtype (naming those tasks as detection and classification, respectively), we proposed a two-stage pipeline presented in Figure 7. 

The detection task is performed in stage 1, where the positive cases of hemorrhage identified by this stage would continue to stage 2 to be classified into its subtype. 

![image](https://user-images.githubusercontent.com/105933447/217235785-4e3abd52-f40e-4bd0-ae5f-30bbd2f31b01.png)

## CNN feature extraction with kNN classification  

To compare a deep-learning approach to a traditional machine learning method, we also propose a CNN feature extraction with kNN classification. 4096 features were extracted from the first convolutional layer of VGG16 and used in the kNN classifier. Figure 8 shows the workflow for this method. 

![image](https://user-images.githubusercontent.com/105933447/217235745-79191d2a-fecc-4b13-ae78-cfc84299a13a.png)

## Training 

The dataset used has patients with and without hemorrhage. Therefore, for the detection task, the whole dataset was split into 70% training data, 15% validation data and 15% test data. To guarantee that the test set of the pipeline (detection test set) did not include positive images that belonged to the training set of the classification task, the classification test set was constituted by the images with hemorrhage presented in the detection test set. 

For the classification task, a sub-dataset with only the positive cases from the detection dataset, and without the images from the classification test set, was divided in order to achieve also 70% training data and 15% validation data regarding the whole classification dataset. A scheme of the data splitting is presented in Figure 9. 

![image](https://user-images.githubusercontent.com/105933447/217236140-5c9c3e04-e11e-45cd-a501-a20117d9fbbc.png)

In order to increase the size of the training set and improve the accuracy and generalization of our models, some data augmentation transformations were applied with the Albumentations library [20] to the training set: horizontal flip, brightness modification between -0.1 and 0.1, and rotation. Every transformation had a probability value of 0.5. 

All the images were resized to a square of size 224+5% and random cropped to a square of 224. The optimizer used was Adam, with a batch size of 16, and the loss function was cross entropy loss. Since the dataset and sub-dataset were imbalanced, the class weights for each task dataset were calculated and used to weight the loss function. 

## Results and Discussion 

In the following section, we present and analyze several results. Firstly, we introduce the performance metrics and a summary of graphic results. Then, we discuss these results and their possible meanings. 

### Performance Metrics 

To summarize our results, we present Table 1 with relevant metrics to evaluate the different proposed methods. The metrics used were the precision and recall value as well as the Matthew’s Correlation Coefficient, being all of them weighted by class weights. 

The Matthews Correlation Coefficient (MCC) is a metric used to evaluate a model’s performance in a classification task, with particular interest when there is imbalanced data because includes all the entries of a confusion matrix in both the numerator and the denominator. The equation for this metric is:  

![image](https://user-images.githubusercontent.com/105933447/217236365-a99ef41a-f213-48c6-813a-ed19dc8fd43e.png)
 
where c is the total number of elements predicted, s is the total number of elements, pk is the number of times class k was predicted and tk is the number of times class k truly occurred. 

MCC can have values between -1 and 1, with 1 representing a strong positive correlation between the predictions and the true labels, therefore the model has learned correctly. A MCC of value 0 would imply no correlation between the variables, with the classifier randomly assigning a class to each unit, and a value -1 would imply an inverse type of correlation, with the model learning but systematically switching all the labels [21]. 

![image](https://user-images.githubusercontent.com/105933447/217236500-71fa618e-7841-40b0-a3dc-9dfffd888e3c.png)

![Daco-Report-4](https://user-images.githubusercontent.com/105933447/217236769-e217b006-2be3-4aee-bcb9-9a8dc70dc0e0.png)

### Comparison between CNNs 

For the convolutional neural networks analyzed, several learning rates were tested. Figures 10 and 13 show the evolution of the learning rates tested for VGG16 in the detection and classification, respectively. In both cases, the loss was the lowest for 0.000005 as the learning rate parameter. 

The detection task using VGG16 was only possible to 95 epochs, due to time constraints of Google Collab, and Figure 11 illustrates the diminishing loss along those epochs. Regarding the classification task with VGG16, the final model was trained for 180 epochs and the loss decrease is shown in Figure 14. The instability of the graph suggests that the loss could be further decreased if only we had more computational power. 

Shifting the focus to ResNeXt, the results achieved were not satisfactory, especially when it comes to the classification task. As Table 1 shows, the precision, recall and Matthew’s coefficient in the detection task was slightly lower for the ResNeXt architecture comparing to the remaining ones, but much lower when it comes to classification task. Figure 16 displays the graph of loss vs. epoch for the classification task and it is possible to see that the minimum value of loss is still above 1.5, after 100 epochs. The confusion matrixes for the detection and classification task are Figures 17 and 18, respectively. In the latter, it is visually noticeable the imprecision of the model to classify the five types of hemorrhage. Ideally, the main diagonal should have lighter tones and be of higher value. In fact, the accuracy never exceeds 0.50, which is undesirable. The running time of the ResNeXt model was extremely high, which limited the amount of epochs that could be implemented, compromising the fine tuning of the parameters, and consequently the goal of getting optimal results. The overall bad results of ResNeXt in an earlier stage of the work lead to opt for VGG16 in the pipeline and in the feature extraction phase for the kNN model. 

### kNN Model  

For the kNN method the extracted training features were used to fit the model and the validation features were used to choose the k value (number of nearest neighbors) by testing the accuracy in several k values, as seen in Figure 19 and 21.  

In the detection task, the k parameter was set at 33 with an accuracy of 0.83. In the classification task, the k parameter was set at 14 with an accuracy of 0.64.  

From the confusion matrixes, we can conclude that the detection task was sufficiently successful, while the classification task failed largely in classifying the epidural hemorrhages.  

This is also supported by the results on Table 1, where the kNN model obtained better metrics in the detection task. 

### 3-window vs original images 

In addition to the two VGG-16 models trained for the detection and the classification task using the 3-window RGB images, another two VGG-16 models were trained on those tasks using the gray-scale images with the original window. Table 1 shows that the windowing images improved the performance of the classifier, in all metrics, for the task of classification, while showing no significant differences for the detection task. When training the CNN for the classification task, the training loss decreased quicker for the 3-window images, as presented in Figure 26. Therefore, the 3-window preprocessing improved our model’s performance, as already proved in some other implementations. 

![image](https://user-images.githubusercontent.com/105933447/217237005-57f65899-774d-4391-b63a-94bbfd6e78ea.png)

### GradCAM 

To make the decision process of the VGG-16 classification model more explainable, some of the results using GradCAM were extracted and presented in Figure 27, as well as the probability of the correct prediction for each case.  

For the images of labels epidural and subarachnoid the model was moderately certain of those decisions, with the heat map providing no information about a possible location of the hemorrhage since almost all the cranial area is in hot colors. For the examples of classes intraparenchymal and intraventricular, the model was quite certain of those predictions. Analyzing the heat map, a few portions of the CT were highlighted as determinant in this decision that could be a potential location of the hemorrhage. Since we don’t have the ground truth of the location, we cannot infer if those locations are correct or not, except for the second image, where the hemorrhage is clearly visible in the hot zone. 

![image](https://user-images.githubusercontent.com/105933447/217237154-923b11e2-20be-4d8a-8010-f9829a1d29fb.png)

### Analysis of Two-stage Pipeline  

After testing the separate tasks of detection and classification, we then applied these models to a two-stage pipeline, as described in the Methods section. Since VGG16 showed better results in both tasks, it was the chosen CNN for the pipeline.  

In this pipeline, the full detection test dataset first went through the detection task and those who were classified as positive were then used as an input for the classification task. The images classified as presenting no hemorrhages were given a new class (5) to create the final confusion matrix with all the possible classes (no hemorrhage and 5 types of hemorrhage) as seen in Figures 23 to 25.  

In Figure 23 we present the results from a baseline method, achieved by using the VGG16 to classify the six classes in one single task (without separating detection and classification). This method was used to work as a comparison with our proposed pipeline. In Figure 24 we present the results from the two-stage pipeline using only the VGG16 and in Figure 25 the results from the two-stage pipeline using the VGG-kNN method.  

Looking at the test confusion matrix, we can see that both the baseline method and VGG16 method do a reasonably good job since the diagonal values, corresponding to the percentages of rightfully classified images, are higher than those surrounding. In the baseline method, the intraventricular, subarachnoid, and subdural classes were the most successfully classified, whereas in the VGG16 method this was seen for the epidural, intraventricular and no hemorrhage classes. The fact that the two-stage pipeline can identify best those images without hemorrhage proves an advantage in separating the two tasks (by first identifying the images with or without hemorrhage present).  

For the VGG-kNN model, there is a clear failure in classifying the epidural images correctly. This is consistent with the classification results presented in Figure 22 for the VGG-kNN model. Thus, in this two-stage pipeline, as one of the tasks performs poorly, this is then reflected in the pipeline results accordingly.  

With the results in Table 1, when comparing the baseline method with the VGG16 method, every metric (precision, recall and Matthew’s coefficient) is higher in the VGG16 method. These results support our proposal of using the two-stage pipeline. Interestingly, the values reported for the VGG-kNN show higher values of recall and Matthew’s coefficient when compared to the previous methods. There is some uncertainty on the interpretation of these values.  

## Conclusion 

Results showed that mimicking the radiologist process in the evaluation of a brain CT, by providing the neural networks with RGB images with 3 of the most relevant windows in each channel, enhanced the neural network performance, contributing to smaller training loss and higher evaluation metrics, such as MCC. 

From the CNNs tested, the VGG16 showed better performance in both tasks (detection and classification) than the ResNeXt. The latter probabily needed more runtime in training to showcase better results.  

The pipeline method provided better results in classifying the images without hemorrhage, when compared to the baseline, denoting some advantage in this task separation. However, the kNN classifier failed in classifying the class epidural, not improving the pipeline containing only the CNNs. 

GradCAM results give a CNN model some explainability for a decision, with the heat map and the probability values being promising indicators of possible locations of the hemorrhage, at least in some of the labels. 

Some improvements could be done in our proposal. Regarding the models training, it is clear from the training loss vs epochs plots that the loss value did not stagnate, having margin for improvement if more epochs were run, which was not possible due to Google Collab usage limitation. Analyzing the confusion matrix for the training and test of each model, it seems to not overfit the training data. However, if eventually that was the case, adding a L2 regularization would be a possible approach. Regarding the kNN classifier, the quality of the features is critical for a good result. Therefore, other features extraction methods could be explored, such as the use of an autoencoder to extract the features from its bottleneck layer, where the relevant features for reconstructing an image are present. Another possible approach would be to take advantage of the fact that VGG16 was already trained in large datasets such as ImageNet, using transfer learning to retrain it to our task, instead of doing it from scratch. 

## References 

[1]	N. T. Patel and S. D. Simon, “Intracerebral Hemorrhage – Symptoms, Causes, Diagnosis and Treatments,” American Association of Neurogical Surgeons. 2020. 

[2]	A. Stein et al., “RSNA Intracranial Hemorrhage Detection,” Kaggle, 2019. [Online]. Available: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/overview. 

[3]	K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” 2014. 

[4]	A. Ajit, K. Acharya, and A. Samanta, “A Review of Convolutional Neural Networks,” in International Conference on Emerging Trends in Information Technology and Engineering, ic-ETITE 2020, 2020. 

[5]	A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks.” . 

[6]	K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 2016-Decem, pp. 770–778, 2016. 

[7]	G. L. Team, “Introduction to resnet or residual network,” Great Learning Blog, Mar-2022. [Online]. Available: https://www.mygreatlearning.com/blog/resnet/. 

[8]	S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated residual transformations for deep neural networks,” Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, vol. 2017-Janua, pp. 5987–5995, 2017. 

[9]	V. Kurama, “A guide to DenseNet, ResNeXt, and ShuffleNet V2,” Paperspace Blog, Apr-2021. [Online]. Available: https://blog.paperspace.com/popular-deep-learning-architectures-densenet-mnasnet-shufflenet/. 

[10]	E. Ma, “Enhancing resnet to resnext for image classification,” Medium, Mar-2020. [Online]. Available: https://medium.com/dataseries/enhancing-resnet-to-resnext-for-image-classification-3449f62a774c. 

[11]	M. Burduja, R. T. Ionescu, and N. Verga, “Accurate and efficient intracranial hemorrhage detection and subtype classification in 3D CT scans with convolutional and long short-term memory neural networks,” Sensors (Switzerland), vol. 20, no. 19, pp. 1–21, 2020. 

[12]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” International Journal of Computer Vision, vol. 128, no. 2, pp. 336–359, 2016. 

[13]	S. Benyahia, B. Meftah, and O. Lézoray, “Multi-features extraction based on deep learning for skin lesion classification,” Tissue and Cell, vol. 74, 2022. 

[14]	J. Zhuang, J. Cai, R. Wang, J. Zhang, and W. S. Zheng, “Deep knn for medical image classification,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 12261 LNCS, pp. 127–136, 2020. 

[15]	A. Sage and P. Badura, “Intracranial hemorrhage detection in head CT using double-branch convolutional neural network, support vector machine, and random forest,” Applied Sciences (Switzerland), vol. 10, no. 21, pp. 1–13, 2020. 

[16]	P. Amaro, “Smaller RSNA’s DS,” Kaggle, May-2022. [Online]. Available: https://www.kaggle.com/datasets/pedroamaro/smaller-rsnas-ds-ver-10. 

[17]	H. Ye et al., “Precise diagnosis of intracranial hemorrhage and subtypes using a three-dimensional joint convolutional and recurrent neural network,” European Radiology, vol. 29, no. 11, pp. 6191–6201, 2019. 

[18]	T. D. DenOtter and J. Schubert, “Hounsfield Unit - StatPearls - NCBI Bookshelf,” StatPearls Publishing, Treasure Island (FL). . 

[19]	R. EPP, “Gradient & Sigmoid Windowing | Kaggle.” . 

[20]	A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, “Albumentations: Fast and Flexible Image Augmentations,” Information 2020, Vol. 11, Page 125, vol. 11, no. 2, p. 125, 2020. 

[21]	M. Grandini, E. Bagli, and G. Visani, “Metrics for Multi-Class Classification: an Overview,” pp. 1–17, 2020. 


