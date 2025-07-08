import os

### ASRNN
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 2 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 3 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 4 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 5 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 6 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 7 --lr 1e-4  --fc 512 512 512")
os.system("CUDA_VISIBLE_DEVICES=2 python ./main_delayed_recall.py --te ALIF  --seq_len 8 --lr 1e-4  --fc 512 512 512")


### ASRNN
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 10 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 12 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 14 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 16 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 18 --lr 1e-4  --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 20 --lr 1e-4  --fc 512 512 512")


# # ### RhythmASRNN
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 10 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 12 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 14 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 16 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 18 --lr 1e-4 --fc 512 512 512")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 20 --lr 1e-4 --fc 512 512 512")


### ASRNN
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 2 --lr 1e-4  --fc 512 512 512")    # 70.65
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 2 --lr 1e-4  --fc 1024 1024")    # 53.90
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 100 --fc 256 256")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 200  --fc 256 256")
# # os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 400 --beta 0.0025  --fc 256 256")
#
#
# ### RhythmASRNN
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 2 --lr 1e-4 --fc 512 512 512")   # 99.85
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 2 --lr 1e-3 --fc 1024 1024")   # 98.05
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 50  --fc 256 256")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 100 --fc 256 256")
# os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 200  --fc 256 256")
# # os.system(" python ./main_delayed_recall.py --te RhythmALIF  --seq_len 400 --beta 0.0025  --fc 256 256")
#
#
#
# ### DEXAT
# os.system(" python ./main_delayed_recall.py --te DEXAT  --seq_len 2 --lr 1e-4 --fc 1024 1024")
# os.system(" python ./main_delayed_recall.py --te DEXAT  --seq_len 100 --fc 256 256")
# os.system(" python ./main_delayed_recall.py --te DEXAT  --seq_len 200  --fc 256 256")
# # os.system(" python ./main_delayed_recall.py --te DEXAT  --seq_len 400 --beta 0.0025  --fc 256 256")
#
#
# ### RhythmDEXAT
# os.system(" python ./main_delayed_recall.py --te RhythmDEXAT  --seq_len 2 --lr 1e-3 --fc 1024 1024")
# os.system(" python ./main_delayed_recall.py --te RhythmDEXAT  --seq_len 100 --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te RhythmDEXAT  --seq_len 200  --fc 280 280")
# # os.system(" python ./main_delayed_recall.py --te RhythmDEXAT  --seq_len 400 --beta 0.0025  --fc 280 280")



# ### LSTM
# os.system(" python ./main_delayed_recall.py --te LSTM  --seq_len 50 --fc 114 114 --grad-clip 1e-1")
# os.system(" python ./main_delayed_recall.py --te LSTM  --seq_len 100 --fc 114 114")
# os.system(" python ./main_delayed_recall.py --te LSTM  --seq_len 200 --fc 114 114")
# os.system(" python ./main_delayed_recall.py --te LSTM  --seq_len 400 --fc 114 114")
#
# ### SRNN
# os.system(" python ./main_delayed_recall.py --te LIF  --seq_len 50  --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te LIF  --seq_len 100 --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te LIF  --seq_len 200  --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te LIF  --seq_len 400 --beta 0.0025  --fc 280 280")
#
# ### ASRNN
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 50  --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 100 --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 200  --fc 280 280")
# os.system(" python ./main_delayed_recall.py --te ALIF  --seq_len 400 --beta 0.0025  --fc 280 280")


