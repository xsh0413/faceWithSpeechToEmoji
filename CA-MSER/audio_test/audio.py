import pyaudio
import numpy as np
import time
import pickle
from audio_test.audio_extraction import extract

# 音频参数设置
FORMAT = pyaudio.paInt16    # 采样格式
CHANNELS = 1                # 单声道
RATE = 44100                # 采样率
CHUNK = 1024                # 数据块大小
WINDOW_DURATION = 3         # 处理窗口时长（秒）
PROCESS_INTERVAL = 0.03     # 处理间隔（秒）
TARGET_SAMPLE_RATE = 22050  # librosa默认采样率

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

        self.streamTimer = time.time()

        # self.load_data()

    # def load_data(self):
    #     """加载 pkl 存储的数据"""
    #     try:
    #         with open('data.pkl', 'rb') as file:
    #             self.saved_data = pickle.load(file)
    #             print("数据加载成功")
    #     except FileNotFoundError:
    #         self.saved_data = None
    #         print("未找到数据文件")

    def callback(self, in_data, frame_count, time_info, status):
        """音频采集回调函数"""
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
            self.process_data()

    def extract(self, audio_data, sr):
        print(f"Extracting features from audio data: {audio_data}")
        print(f"{len(audio_data)}")

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

if __name__ == "__main__":
    processor = AudioProcessor()
    processor.run()
    print(f"features:{processor.features}")