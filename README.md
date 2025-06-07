# LARES

## Requirements

```
torch==2.4.0
recbole==1.2.1
numpy
tqdm
```

## Dataset
You can find all the datasets we used in [Google Drive](https://drive.google.com/file/d/1P9ihX3L8zCYjg6c9p6EkiFRh0C9s2Qem). Please download the file and unzip it to the `dataset/` folder.

## Self-Supervised Pre-Training

```shell
bash run_sl.sh
```

Filter data

```shell
bash run_filter.sh
```

## Reinforcement Post-Training

```shell
bash run_rl.sh
```