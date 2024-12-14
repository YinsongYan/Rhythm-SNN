
from torchaudio import datasets

datasets.SPEECHCOMMANDS(
    root="/home/ysyan/yysproject/RhythmSNN/data",
    url='speech_commands_v0.02',
    folder_in_archive='SpeechCommands',
    download=True
)
