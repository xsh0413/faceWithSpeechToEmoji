import os
# import sys
# import argparse
import numpy as np
import pickle
# from collections import Counter
# import pandas as pd
# from database import SER_DATABASES
import random
from transformers import Wav2Vec2Processor
# from tqdm import tqdm
from collections import defaultdict
import pickle
import librosa
import librosa.display
import math



def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def extract_logspec(x, sr, params):
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:nfreq]
    
    #Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec,0)

    return spec

def segment_nd_features(input_values, mfcc, data, segment_size):
    '''
    将特征分割成 <segment_size> 帧。
    如果数据帧数小于 segment_size，则用0填充。

    输入:
    ------
        - data: 形状为 (Channels, Fime, Time)
        - emotion: 当前语音数据的情感标签
        - segment_size: 每个分段的长度
    
    返回:
    -------
    元组 (分段数量, 帧, 分段标签, 语音标签)
        - 帧: 形状为 (N, C, F, T) 的 ndarray
                    - N: 分段数量
                    - C: 通道数量
                    - F: 频率索引
                    - T: 时间索引
        - 分段标签: 每个分段的标签列表
                    - 分段标签的长度 == 分段数量
    '''
    segment_size_wav = segment_size * 160
    # Transpose data to C, T, F
    
    data = data.transpose(0,2,1)
    time = data.shape[1]
    time_wav = input_values.shape[0]
    nch = data.shape[0]
    start, end = 0, segment_size
    start_wav, end_wav = 0, segment_size_wav
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    #if num_segs > 1:
    #    num_segs = num_segs - 1
    mfcc_tot = []
    audio_tot = []
    data_tot = []
    sf = 0
    
    processor = Wav2Vec2Processor.from_pretrained("pretrained_model/wav2vec2-base-960h")
    
    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        if end_wav > time_wav:
            end_wav = time_wav
            start_wav = max(0, end_wav - segment_size_wav)
        """
        if end-start < 100:
            num_segs -= 1
            print('truncated')
            break
        """
        # Do padding
        mfcc_pad = np.pad(
                mfcc[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
        
        audio_pad = np.pad(input_values[start_wav:end_wav], ((segment_size_wav - (end_wav - start_wav)), (0)), mode="constant")
  
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
                #data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant",
                #constant_values=((-80,-80),(-80,-80)))
            data_pad.append(data_ch)

        
        #audio_wav = processor(audio_wav.cpu(), sampling_rate=16000, return_tensors="pt").input_values# [1, batch, 48000] 
        #audio_wav = audio_wav.permute(1, 2, 0) # [batch, 48000, 1] 
        #audio_wav = audio_wav.reshape(audio_wav.shape[0],-1) # [batch, 48000] 
        
        
        data_pad = np.array(data_pad)
        
        # Stack
        mfcc_tot.append(mfcc_pad)
        data_tot.append(data_pad)

        audio_pad_np = np.array(audio_pad)
        audio_pad_pt = processor(audio_pad_np, sampling_rate=16000, return_tensors="pt").input_values
        audio_pad_pt = audio_pad_pt.view(-1)
        audio_pad_pt_np = audio_pad_pt.cpu().detach().numpy()
        audio_tot.append(audio_pad_pt_np)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)
        start_wav = end_wav
        end_wav = min(time_wav, end_wav + segment_size_wav)      
    
    mfcc_tot = np.stack(mfcc_tot)
    data_tot = np.stack(data_tot)
    audio_tot = np.stack(audio_tot)
    # utt_label = emotion
    # segment_labels = [emotion] * num_segs
    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, mfcc_tot, audio_tot)




def extract_features(audio_data, sr, params):
    # processor = Wav2Vec2Processor.from_pretrained("pretrained_model/wav2vec2-base-960h")
    speaker_features = defaultdict()

    data_tot, labels_tot, labels_segs_tot, segs, data_mfcc, data_audio = list(), list(), list(), list(), list(), list()
    # for wav_path, emotion in speaker_files[speaker_id]:
        
    # 读取波形数据
    # x, sr = librosa.load(wav_path, sr=None)

    # 应用预加重滤波器
    audio_data = librosa.effects.preemphasis(audio_data, zi=[0.0])

    # 提取所需特征为 (C,F,T)
    features_data = extract_logspec(audio_data, sr, params)
    
    hop_length = 160 # hop_length 越小，序列长度越大
    # f0 = librosa.feature.zero_crossing_rate(x, hop_length=hop_length).T # (seq_len, 1)
    # cqt = librosa.feature.chroma_cqt(y=x, sr=sr, n_chroma=24, bins_per_octave=72, hop_length=hop_length).T # (seq_len, 12)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T # (seq_len, 20)
    
    # wav2vec
    # input_values = processor(x, sampling_rate=sr, return_tensors="pt").input_values
    
    # 将特征分割为 (N,C,F,T)
    features_segmented = segment_nd_features(audio_data, mfcc, features_data, params['segment_size'])

    # 收集所有分段
    data_tot.append(features_segmented[1])
    # labels_tot.append(features_segmented[3])
    # labels_segs_tot.extend(features_segmented[2])
    segs.append(features_segmented[0])
    data_mfcc.append(features_segmented[2])
    data_audio.append(features_segmented[3])




    # Post process
    data_tot = np.vstack(data_tot).astype(np.float32)
    data_mfcc = np.vstack(data_mfcc).astype(np.float32)
    data_audio = np.vstack(data_audio).astype(np.float32)
    # labels_tot = np.asarray(labels_tot, dtype=np.int8)
    # labels_segs_tot = np.asarray(labels_segs_tot, dtype=np.int8)
    segs = np.asarray(segs, dtype=np.int8)
    
    # Make sure everything is extracted properly
    # assert len(labels_tot) == len(segs)#+ == data_mfcc.shape[0]
    # assert data_tot.shape[0] == labels_segs_tot.shape[0] == sum(segs)


    #Put into speaker features dictionary
    print(data_tot.shape)
    # print(labels_segs_tot.shape)
    print(data_audio.shape)
    # print(labels_tot.shape)
    print(segs.shape)
    # print(labels_tot.shape)
    audio_features = defaultdict()
    audio_features["seg_spec"] = data_tot
    # audio_features["utter_label"] = labels_tot
    # audio_features["seg_label"] = labels_segs_tot
    audio_features["seg_num"] = segs
    audio_features["seg_mfcc"] = data_mfcc
    audio_features["seg_audio"] = data_audio
    # speaker_features[speaker_id] = audio_features #(data_tot, labels_tot, labels_segs_tot, segs)
    speaker_features['audio'] = audio_features

    print(speaker_features)

    return speaker_features



def extract(audio_data,sr=44100):
    params={'window'        : "hamming",
            'win_length'    : 18,
            'hop_length'    : 10,
            'ndft'          : 800,
            'nfreq'         : 200,
            'nmel'          : 128,
            'segment_size'  : 300,
            'mixnoise'      : False
            }
    
    dataset  = "IEMOCAP"
    features = "logspec"
    dataset_dir = "../../Datasets/IEMOCAP"
    mixnoise = False 

    out_filename = './features_extraction/IEMOCAP_logspec200.pkl'

    seed_everything(42)

    features_data = extract_features(audio_data, sr, params)

    print(type(features_data['audio']))

    return features_data

    # if out_filename is not None:
    #     with open(out_filename, 'wb') as fout:
    #         pickle.dump(features_data, fout)

