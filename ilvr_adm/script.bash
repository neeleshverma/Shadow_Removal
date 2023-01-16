pip install blobfile
pip install mpi4py
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --schedule_name linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
python image_train.py --data_dir "/content/drive/MyDrive/Fall 2022/CV/Shadow Removal DDPM Project/ISTD_Dataset/train/train_C" $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS