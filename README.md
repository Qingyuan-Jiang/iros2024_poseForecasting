# iros2024_poseForecasting
Map-Aware Human Pose Prediction for Robot Follow-Ahead

To run the minimum code, go into minimum_code directory:

```
cd minimum_code
```

Install the requirements by:

```
pip install -r requirements.txt
```

Then, download pre-processed dataset for [GTA-IM](https://drive.google.com/drive/folders/1wSgFpP_rE1wEgO5R0_LuSH-2XVWuJmn7?usp=drive_link) and [Real-IM](https://drive.google.com/drive/folders/1R51LSCWSjrgCoQf--JE_5pv4gyXungEs?usp=drive_link) and put them under folders in data/ directory. Next, download the model weights [real_Path_GRU](https://drive.google.com/drive/folders/1tIKpXmtf2xDYqQkLYK7gQJyisDmuxOTX?usp=drive_link) and [gta_Path_GRU](https://drive.google.com/drive/folders/1FsJOAN6hw4UU-LAzd94w0uv_OTQ7PMyJ?usp=drive_link) under results/real_Path_GRU and results/gta_Path_GRU directories, respectively.

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
python train.py --cfg gta_Path_GRU --epoch 1200
```
