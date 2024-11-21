# IBI
Source code of 'Instance by Instance: An Iterative Framework for Multi-instance 3D Registration.'

## Introduction
In this paper, we propose the first iterative framework for multi-instance 3D registration (MI-3DReg) in this work, termed instance-by-instance (IBI). It successively registers instances while systematically reducing outliers, starting from the easiest and progressing to more challenging ones. This enhances the likelihood of effectively registering instances that may have been initially overlooked, allowing for successful registration in subsequent iterations. Under the IBI framework, we further propose a Sparse-to-Dense Correspondence-based multi-instance registration method (IBI-S2DC) to enhance the robustness of MI-3DReg. Experiments on both synthetic and real datasets have demonstrated the effectiveness of IBI and suggested the new state-of-the-art performance with IBI-S2DC, _e.g._, our mean registration F1 score is **12.02%/12.35%** higher than the existing state-of-the-art on the synthetic/real datasets.
![image](https://github.com/user-attachments/assets/8aba13a9-c402-46fe-86f9-cac7b649a306)

## Datasets
All tested datasets can be found at this link[https://pan.baidu.com/s/1J_QvUobTmf-B4S0d8vrKvA], password：s2dc

## Results
If you find this code useful for your work or use it in your project, please consider citing:

**Synthetic&Real**

![image](https://github.com/user-attachments/assets/d2008d8e-736c-4f63-bde2-81e61868c098)

**Module effective analysis**

![image](https://github.com/user-attachments/assets/ce878661-8503-41e5-80c0-bd419caa58ee)

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

