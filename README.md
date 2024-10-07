# AKD-IQA

The paper is under review.

# Abstract

Compared with no-reference image quality assessment (IQA), full-reference IQA often achieves higher consistency with human subjective perception due to the reference information for comparison. A natural idea is to design strategies that allow the latter to guide the former's learning to achieve better performance. However, how to construct the reference information and how to transfer prior knowledge are two important issues we are going to face that have not been fully explored. To this end, a novel method called no-reference IQA via inter-level adaptive knowledge distillation (AKD-IQA) is proposed. The core of AKD-IQA lies in transferring image distribution difference information from the full-reference teacher model to the no-reference student model through inter-level AKD. First, the teacher model is constructed based on multi-level feature discrepancy extractor and cross-scale feature integrator. Then, it is trained on a large synthetic distortion dataset to establish a comprehensive difference prior distribution. Finally, the image re-distortion strategies and inter-level AKD are introduced into the student model for effective learning. Experimental results on six standard IQA datasets demonstrate that the AKD-IQA achieves state-of-the-art performance. In addition, cross-dataset experiments confirm the superiority of it in generalization ability.

# Requirements

- einops
- numpy
- openpyxl
- pandas
- Pillow
- scipy
- torch
- torchvision
- Util

More information please check the requirements.txt.