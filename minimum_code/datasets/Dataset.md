# Human Motion Prediction Dataset.

This is a description for the Human Motion Prediction in Large-Scale Indoor Environment Dataset. (HMPLIED).

## Structure

The data folder should look like this:

```
humanPosePrediction
├── data
│   ├── GTA-IM-Dataset
│   │   ├── 2020-06-03-13-31-46
│   │   │   ├── 0000.jpg
│   ├── pf_preprocessed
│   │   ├── 2020-06-03-13-31-46_r001_sf0.npz
│   ├── Real-IM-Dataset-raw
│   │   ├── shepherd-floor1_hm0_seq0
│   │   │   ├── 000000_f.jpg
│   │   │   ├── 000000_b.jpg
│   │   │   ├── info_frames.npz
│   │   │   ├── info_frames.pickle
│   ├── Real-IM-Dataset
│   │   ├── shepherd-floor1_hm0_seq0.npz
├── datasets
├── results
```

## Data content.

The data should for each `npz` file should contain content includes `joints`, `scene_points`, `scene_colors`.