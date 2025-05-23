

GPU_ID=3
DATASET=Instruments
MODEL=LARES
DATA_PATH=./dataset


python sl_train.py \
  -d $DATASET \
  -m $MODEL \
  --gpu_id=$GPU_ID \
  --data_path=$DATA_PATH \
  --learning_rate=0.001 \
  --attn_dropout_prob=0.5 \
  --hidden_dropout_prob=0.5 \
  --alpha=0.1 \
  --gamma=0.5 \
  --n_pre_layers=2 \
  --n_core_layers=2 \
  --mean_recurrence=4 \
  --train_batch_size=1024 \
  --tau=1 \
  --sem_func=dot \
  --state_std=1.0 \
  --state_scale=3


