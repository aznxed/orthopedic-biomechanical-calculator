# Convolutional Neural Network for Disk Hernia and Spondilolysthesis Classification
Using PyTorch and Amazon SageMaker, I've created a convolutional neural network capable of classifying disk hernia and spondilolysthesis with 88% accuracy given 6 biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine: pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis.

**Architecture**

![](https://raw.githubusercontent.com/aznxed/orthopedic-biomechanical-calculator/master/architecture.PNG)

**Future Development**

I will be creating an web application to allow for use as a clinical calculator. 

**Data Set Information:**

Biomedical data set built by Dr. Henrique da Mota during a medical residence period in the Group of Applied Research in Orthopaedics (GARO) of the Centre MÃ©dico-Chirurgical de RÃ©adaptation des Massues, Lyon, France. The data have been organized in two different but related classification tasks. The first task consists in classifying patients as belonging to one out of three categories: Normal (100 patients), Disk Hernia (60 patients) or Spondylolisthesis (150 patients). For the second task, the categories Disk Hernia and Spondylolisthesis were merged into a single category labelled as 'abnormal'. Thus, the second task consists in classifying patients as belonging to one out of two categories: Normal (100 patients) or Abnormal (210 patients). We provide files also for use within the WEKA environment.

**Attribute Information:**

Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (in this order): pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis. The following convention is used for the class labels: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and Abnormal (AB).

**Dataset Source:**

* Guilherme de Alencar Barreto (guilherme '@' deti.ufc.br) & Ajalmar RÃªgo da Rocha Neto (ajalmar '@' ifce.edu.br), Department of Teleinformatics Engineering, Federal University of CearÃ¡, Fortaleza, Ceará¡, Brazil.
* Henrique Antonio Fonseca da Mota Filho (hdamota '@' gmail.com), Hospital Monte Klinikum, Fortaleza, Ceará¡, Brazil.
* Kaggle Link - https://www.kaggle.com/caesarlupum/vertebralcolumndataset
