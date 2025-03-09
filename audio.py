import pyaudio
import numpy as np
import torch
import torch.nn.functional as f
import time
import pickle
from audio_extraction import extract
from ser_models.ser_model import Ser_Model
from data_utils import SERInput
# from audio_extraction import extract

from transformers import logging
logging.set_verbosity_error()



# 音频参数设置
FORMAT = pyaudio.paInt16    # 采样格式
CHANNELS = 1                # 单声道
RATE = 44100                # 采样率
CHUNK = 1024                # 数据块大小
WINDOW_DURATION = 3         # 处理窗口时长（秒）
PROCESS_INTERVAL = 1     # 处理间隔（秒）
TARGET_SAMPLE_RATE = 22050  # librosa默认采样率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMO_MAP = {0:'ang', 1:'dis', 2:'fea', 3:'hap', 4:'neu', 5:'sad', 6:'sur'}

# 计算窗口大小
WINDOW_SIZE = int(RATE * WINDOW_DURATION)

class AudioProcessor:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_process_time = 0
        self.initial_buffer = True
        
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback
        )

        # 音频特征
        self.features = None
        self.result = None
        self.result_array = None
        self.result_flag = False

        self.is_paused = False  # 新增暂停状态标志
        self.pause_buffer = np.array([], dtype=np.int16)  # 暂停时临时保存数据

        self.streamTimer = time.time()

    def pause(self):
        """暂停音频收集"""
        if not self.is_paused:
            self.is_paused = True
            # 保存当前缓冲区内容
            # self.pause_buffer = self.audio_buffer.copy()
            # 清空主缓冲区
            self.audio_buffer = np.array([], dtype=np.int16)
            print("麦克风已暂停")

    def resume(self):
        """恢复音频收集"""
        if self.is_paused:
            self.is_paused = False
            # 恢复暂停前的数据
            # self.audio_buffer = self.pause_buffer
            # self.pause_buffer = np.array([], dtype=np.int16)
            print("麦克风已恢复")


    def callback(self, in_data, frame_count, time_info, status):
        """音频采集回调函数"""
        if self.is_paused:
            # 暂停时不处理数据
            return (None, pyaudio.paContinue)

        # 将新数据转换为numpy数组
        new_data = np.frombuffer(in_data, dtype=np.int16)
        
        # 更新缓冲区
        self.audio_buffer = np.concatenate((self.audio_buffer, new_data))
        
        # 保持缓冲区不超过窗口大小
        if len(self.audio_buffer) > WINDOW_SIZE:
            self.audio_buffer = self.audio_buffer[-WINDOW_SIZE:]
        
        # 处理控制逻辑
        self.process_control()
        return (None, pyaudio.paContinue)

    def process_control(self):
        """处理时机控制"""
        if self.is_paused:
            return  # 暂停时不处理数据

        current_time = time.time()
        
        # 初始缓冲区填充检查
        if self.initial_buffer:
            if len(self.audio_buffer) >= WINDOW_SIZE:
                self.initial_buffer = False
                self.last_process_time = current_time
                self.process_data()
            return
        
        # 常规处理间隔检查
        if (current_time - self.last_process_time) >= PROCESS_INTERVAL:
            self.last_process_time = current_time
            # print("time")
            self.process_data()
            self.cal_result()
    
    def no_result(self):
        print(len(self.audio_buffer))
        if len(self.audio_buffer) < WINDOW_SIZE :
            print("wind")
            return True
        return False

        
    def cal_result(self):
        data = SERInput(self.features).get_data()

        model = Ser_Model().to(DEVICE)
        with torch.no_grad():
            model.load_state_dict(torch.load("model.pth"))

        data_spec = torch.tensor(data["seg_spec"].copy(), dtype=torch.float32).to(DEVICE)
        data_mfcc = torch.tensor(data["seg_mfcc"].copy(), dtype=torch.float32).to(DEVICE)
        data_audio = torch.tensor(data["seg_audio"].copy(), dtype=torch.float32).to(DEVICE)
        # data_num = torch.tensor(data['audio']["seg_num"], dtype=torch.int8).to(DEVICE)
        outputs = model(data_spec, data_mfcc, data_audio)


        # print(f"result:{f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy()}")
        # print(f"模型结果：{outputs['M']}")

        result = np.array(f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy())

        # 这里突然发现之前提取特征的时候，batch=2
        # 可能是由于原项目是两个人的对话，所以原代码中保留了batch至少为偶数？

        self.result_array = np.sum(result,axis=0)

        result = np.argmax(self.result_array)

        # print(f"{result}")
        # print(f"{EMO_MAP[result]}")

        self.result = EMO_MAP[result]
        self.result_flag = True

    # def extract(self, audio_data, sr):
    #     # print(f"Extracting features from audio data: {audio_data}")
    #     # print(f"{len(audio_data)}")

    def process_data(self):
        """数据处理方法"""
        if len(self.audio_buffer) < WINDOW_SIZE:
            return
        
        # 获取当前窗口数据
        window_data = self.audio_buffer[-WINDOW_SIZE:]

        # 转换为librosa兼容格式
        window_data = window_data.astype(np.float32) / 32768.0
        
        # 调用特征提取函数
        self.features = extract(window_data, RATE)

    def run(self):
        try:
            print("开始音频采集...")
            self.stream.start_stream()
            while self.stream.is_active():
                time.sleep(0.1)
                # self.pause()
                # time.sleep(1)
                # self.resume()
                # self.save()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("\n正在停止...")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("程序已安全退出")

    def get_features(self):
        return self.features
    
    def get_result(self):
        if self.result_flag == False:
            return np.array([0]*7)
        return self.result_array
    
    def save(self):
        with open('features.pkl','wb') as file:
            pickle.dump(self.features,file)

if __name__ == "__main__":
    start_time = time.time()
    processor = AudioProcessor()
    processor.run()

    
    # print(f"features:{processor.features}")