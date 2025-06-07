

GPU_ID=3

python filter_data.py \
  --ckpt_path=./saved/LARES-SL-Instruments.pth \
  --device=cuda:$GPU_ID \
  --n_iter=3 \
  

