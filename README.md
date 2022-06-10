# MSGazeNet
This is the official implementation of our work entitled "Multistream Gaze Estimation with Anatomical Eye Region Isolation by Synthetic to Real Transfer Learning" in PyTorch (Version 1.9.0).
![Alt text](/Figures/msgazenet.png?raw=true "Optional Title") 

The repository contains the source code of our paper where we use the following datasets: 

* [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild): The dataset provides eye images and their corresponding head pose and gaze annotations from 15 subjects. It was collected in an unconstrained manner over the course of several months. The standard evaluation protocol of this dataset is leave-one-subject-out.  
* [Eyediap](https://www.idiap.ch/en/dataset/eyediap): The dataset was collected in a laboratory environment from 16 subjects. Each participant participated in three different sessions and the standard evaluation protocol of this dataset is 5-fold validation. In this work, we have used the VGA videos from the continuous/discrete screen target session (CS/DS).  
* [UTMultiview](https://www.ut-vision.org/datasets/): The dataset was collected from 50 subjects in a laboratory setup. The collection procedure involved 8 cameras which generated multiview eye image samples and the corresponding gaze labels was also recorded. 


