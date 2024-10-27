import numpy as np
import torch
import torch.nn as nn
from model.fusingnet import FusingNet


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, max_k=400, emb_data_path='datasore_profile', topk=1):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.topk = topk

        self.output_layer = nn.Linear(self.ssl_features, 1)
        self.classify_layer = nn.Linear(self.ssl_features, 16)
        self.fusing_net = FusingNet(emb_data_path=emb_data_path, max_k=max_k, top_k=self.topk)

    def forward(self, wav, syslist):
        wav = wav.squeeze(1)  # [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        
        x_mean = torch.mean(x, 1)

        x_1 = self.output_layer(x_mean)
        x_2 = self.classify_layer(x_mean)

        x_fusing = self.fusing_net(x_mean, x_1, x_2, syslist)
        return x_mean, x_fusing