import torch
import torch.nn as nn
from model.datastore import Datastore


class FusingNet(nn.Module):
    def __init__(self, emb_data_path=None, max_k=None, top_k=8):
        super(FusingNet, self).__init__()
        self.input_dim = max_k
        self.output_dim = max_k
        self.top_k = top_k

        self.knn_datastore = Datastore(emb_data_path, max_k)

        self.get_k_prob = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(32, self.output_dim),
            nn.Softmax(dim=-1),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 3 , 4 , ... , ]
        )

        self.get_lamda_from_knn = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )
        self.get_lamda_from_wav = nn.Sequential(
            nn.Linear(self.input_dim+top_k+2, 32),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def get_flag_idx(self, x):
        x_idx = x * 4
        x_idx = x_idx - 4
        x_idx = torch.clamp(x_idx, 0, 15)
        return x_idx.to(torch.int64)
    
    def forward(self, input_future, wav2vec_score, wav2vec_prob, syslist):
        # input:[B, future], wav2vec_prob:[B, 16]
        # knn_dists:[B, max_k], knn_scores:[B, max_k],
        knn_dists, knn_scores = self.knn_datastore(input_future, syslist)

        # k-net: get knn result
        k_prob = self.get_k_prob(knn_dists)  # [B,max_k]
        knn_result = torch.sum(knn_scores * k_prob, dim=-1) # [B]

        # λ-net: get final output
        knn_idx = self.get_flag_idx(knn_result)  # [B]
        wav2vec_idx = self.get_flag_idx(wav2vec_score) # [B]
        
        # get the probablity of result from two methods
        idx = torch.cat((wav2vec_idx, knn_idx.unsqueeze(-1)), dim=-1)  # [B,2]
        wav2vec_tar_prob = torch.gather(wav2vec_prob, dim=-1, index=idx) # [B,X,2]
        wav2vec_top_prob, top_index = torch.topk(wav2vec_prob, self.top_k) # [B,X,top_k]
        wav2vec_tar_prob = torch.cat((wav2vec_tar_prob, wav2vec_top_prob), dim=-1)  # [B,X,2+top_k]

        knn_lambda = self.get_lamda_from_knn(knn_dists) # [B,X,1]
        wav_input = torch.cat((wav2vec_tar_prob, knn_dists), dim=-1)
        wav_lambda = self.get_lamda_from_wav(wav_input)

        pre_lambda = torch.softmax(torch.cat((knn_lambda, wav_lambda), dim=-1), -1) # [B,X,2]
        result = pre_lambda[:,:1].squeeze(1) * knn_result + pre_lambda[:,1:2].squeeze(1) * wav2vec_score.squeeze(1)


        return result
