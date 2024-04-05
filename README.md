# How to Run
Map-Aware Human Pose Prediction for Robot Follow-Ahead

To run the minimum code, clone the repository and go into the repository directory:

```
cd iros2024_poseForecasting
```

Install the requirements by running:

```
pip install -r requirements.txt
```

Then, download pre-processed dataset for GTA-IM and Real-IM and put them under folders in data/ directory (please follow the instructions in the Dataset section below). Next, download the model weights [real_Path_GRU](https://drive.google.com/drive/folders/1Yi0MfLp8jZs7W50jKT-a8n_wZBVuol9l?usp=sharing) and [gta_Path_GRU](https://drive.google.com/drive/folders/1Yi0MfLp8jZs7W50jKT-a8n_wZBVuol9l?usp=sharing) under results/real_Path_GRU/models and results/gta_Path_GRU/models directories, respectively.

You can run evaluation code on Real-IM dataset by:

```
python eval.py --cfg real_Path_GRU --epoch 3600
```

Similarly, to run on GTA-IM dataset:

```
python eval.py --cfg gta_Path_GRU --epoch 1200
```

To train the network, you can run the training script.

```
python train.py --cfg gta_Path_GRU --epoch 1200 --train_poseet --train_pathnet
```

# Dataset

To get permission to use datasets in the links, you will need to comply with a procedure. 

### For GTA-IM Dataset

Please follow the link to [their repository](https://github.com/ZheC/GTA-IM-Dataset) and navigate under 'Requesting Dataset' and request their dataset. Then, after getting a response from them, please forward it to us so that we can provide you a subset of pre-processed dataset for testing and demonstration.

### For Real-IM Dataset

We provide a subset of the pre-processed version of the dataset in this [link](https://drive.google.com/drive/folders/1jyxGQxHXxyNPKGvHbZcZ0bTUme2Bk3dV?usp=sharing). We will publish the full version our dataset in the future.
