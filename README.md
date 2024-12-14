# Rhythm-SNN
Codes for ***Efficient and robust temporal processing with neural oscillations modulated spiking neural networks***


Yinsong Yan<sup>†</sup>, Qu Yang<sup>†</sup>, [Yujie Wu](https://yjwu17.github.io/), [Haizhou Li](https://www.colips.org/~eleliha/), [Kay Chen Tan](https://www.polyu.edu.hk/dsai/people/academic-staff/tankaychen/) and [Jibin Wu](https://www.jibinwu.com/)*





## 🏊 Usage
## **1. Virtual Environment**
```
# create virtual environment
conda create -n RhythmSNN python=3.8.18
conda activate RhythmSNN

# select pytorch version=2.0.1
# install RhythmSNN requirements
pip install -r requirements.txt
```

## **2. Data Preparation**


All data used in this paper are publicly available. After downloading, please put the dataset in the folder with the corresponding dataset name. 

   1. S-MNIST and PS-MNIST datasets can be downloaded from http://yann.lecun.com/exdb/mnist/
   2. SHD dataset can be downloaded from https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
   3. ECG dataset can be downloaded from https://physionet.org/content/qtdb/1.0.0/
   4. GSC dataset is available at https://tensorflow.google.cn/datasets/catalog/speech_commands/
   5. DVS-Gesture dataset can be downloaded from https://research.ibm.com/interactive/dvsgesture/
   6. VoxCeleb1 dataset can be accessed at https://www.tensorflow.org/datasets/catalog/voxceleb
   7. PTB dataset can be obtained from https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset
   8. Intel N-DNS Challenge dataset can be downloaded from https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge




## **3. Pre-Processing**


The datasets (SHD and GSC) are required to arrange the data before training. The pre-processing codes and instructions can be found in the folder that corresponds to the task.




## **4. Model Training**


This section provides instructions on how to train models on various datasets using the provided scripts. Follow the steps below for each dataset:


### **1. PS-MNIST Dataset**
Run the following scripts for training on the PS-MNIST dataset, located in the `Spiking_pmnist` folder:
```
python main_psmnist-skipFFSNN.py # for Rhythm-FFSNN
python main_psmnist-skipASRNN.py # for Rhythm-ASRNN
``` 


### **2. S-MNIST Dataset**
Run the following scripts for training on the S-MNIST dataset, located in the `spiking_smnist` folder:
```
python main_seqmnist-skipFFSNN.py # for Rhythm-FFSNN
python main_seqmnist-skipASRNN.py # for Rhythm-ASRNN
``` 

### **3. SHD Dataset**
Run the following script for training on the SHD dataset, located in the `SHD` folder:
``` 
python main_dense_general_rhy.py # for Rhythm-DH-SFNN
``` 

### **4. ECG Dataset**
Run the following scripts for training on the ECG dataset, located in the `ECG` folder:
```
python main_ecg-skipFFSNN.py # for Rhythm-FFSNN
python main_ecg-skipASRNN.py # for Rhythm-ASRNN
```

### **5. DVS-Gesture Dataset**
Run the following scripts for training on the DVS-Gesture dataset, located in the `DVS-Gesture` folder:
```
python main_DVS-skipSRNN_general_cosA.py  # for Rhythm-FFSNN
python main_DVS-skipASRNN_general_cosA.py # for Rhythm-ASRNN
```

### **6. VoxCeleb1 Dataset**
Run the following script for training on the VoxCeleb1 dataset, located in the `VoxCeleb1` folder:
- `run_exp_spk.py`

Alternatively, you can use the shell script:
- `run_rhy_exp_spk.sh`

### **7. PTB Dataset**
Run the following scripts for training on the PTB dataset:
```
python RhythmSRNN-ptb.py # for Rhythm-SRNN
python RhythmALIF-ptb.py # for Rhythm-ASRNN
```

### 8. Intel N-DNS Challenge Dataset
The `Intel_N-DNS_Challenge` folder contains code to implement the Rhythm-GSNN model, as described in our paper, by incorporating the proposed rhythm mechanisms into the GSNN. This is based on [Spiking-FullSubNet](https://github.com/haoxiangsnr/spiking-fullsubnet), the winner of Intel N-DNS Challenge Track 1.

See the [Documentation](https://haoxiangsnr.github.io/spiking-fullsubnet/) for installation and usage.






## 🏄 Demo Results
More listening samples of Intel DNS Challenge are provided in [this Google drive folder](https://drive.google.com/drive/folders/1UPuXIr7RGcy911hJrXlWyLwd1NyYfsYg?usp=drive_link) (est.wav is our model's denoising result, raw.wav and ref.wav are raw and clean audios, respectively).
