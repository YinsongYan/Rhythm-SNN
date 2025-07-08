import os

### RhythmASRNN
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 2 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 3 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 4 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 5 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 6 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 7 --lr 1e-4 --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=1 python ./main_delayed_recall.py --te RhythmALIF  --seq_len 8 --lr 1e-4 --fc 512 512 512")


# ### ASRNN
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 10 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 12 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 14 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 16 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 18 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 20 --lr 1e-4  --fc 512 512 512")

#
# ### RhythmASRNN
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 10 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 12 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 14 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 16 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 18 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 20 --lr 1e-4 --fc 512 512 512")

