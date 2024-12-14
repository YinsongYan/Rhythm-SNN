CUDA_VISIBLE_DEVICES=2 python run_exp_spk.py \
    --dataset_name vox1 \
    --data_folder /home/zeyang/data/VoxCeleb1\
    --nb_epochs 200 \
    --nb_hiddens 512 \
    --nb_layers 3 \
    --batch_size 256 \
    --frontend fbank\
    --model_type RhyadLIF \
    --exp_name RhyadLIF_b0-3_c1-30_dc07-1_p0 \
    --log_tofile True \
    --use_augm True \
    --normalization batchnorm \
    --scheduler_type ReduceLROnPlateau \
    --nb_inputs 40 \
    --lr 0.005 \


# CUDA_VISIBLE_DEVICES=0 python run_exp_spk.py \
#     --dataset_name vox1 \
#     --data_folder /home/zeyang/data/VoxCeleb1\
#     --nb_epochs 200 \
#     --nb_hiddens 512 \
#     --nb_layers 3 \
#     --batch_size 256 \
#     --frontend fbank\
#     --model_type PLIF \
#     --log_tofile False \
#     --use_pretrained_model True \
#     --normalization batchnorm \
#     --scheduler_type ReduceLROnPlateau \
#     --nb_inputs 40 \
#     --lr 0.001 \
#     --only_do_testing True \
#     --load_exp_folder exp/spk_id_exps/vox1_RhyPLIF_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_001_fbank_RhyPLIF_DC0_95_1_0 \
