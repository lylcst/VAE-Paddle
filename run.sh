CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --result_dir=result \
    --save_dir=checkpoint \
    --batch_size=128 \
    --epoches=100 \
    --lr=1e-3 \
    --z_dim=20 \
    --input_dim=28*28 \
    --input_channels=1
