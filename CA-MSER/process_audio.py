from audio_test.audio import AudioProcessor
from models.ser_model import Ser_Model
from data_utils import SERInput
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np

LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAVEPATH = "model.pth"

def main():

    processor = AudioProcessor()
    processor.run()

    features = processor.get_features()
    data = SERInput(features)

    model = Ser_Model().to(DEVICE)
    with torch.no_grad():
        model.load_state_dict(torch.load("pretrained_model/model.pth"))
    # optimizer = optim.AdamW(model.parameters(), lr=LR)
    # criterion_ce = nn.CrossEntropyLoss()
    # criterion_mml = nn.MultiMarginLoss(margin=0.5) 

    data_spec = torch.tensor(data['audio']["seg_spec"], dtype=torch.float32).to(DEVICE)
    data_mfcc = torch.tensor(data['audio']["seg_mfcc"], dtype=torch.float32).to(DEVICE)
    data_audio = torch.tensor(data['audio']["seg_audio"], dtype=torch.float32).to(DEVICE)
    # data_num = torch.tensor(data['audio']["seg_num"], dtype=torch.int8).to(DEVICE)
    outputs = model(data_spec, data_mfcc, data_audio)


    print(f"result:{f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy()}")


