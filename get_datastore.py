import os
import numpy as np
import argparse
import pickle
import torch, torchaudio
import faiss
import fairseq
from model.ramp import MosPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='example_samples/label.txt', help='Path of your DATA label')
    parser.add_argument('--checkpoint', type=str, default='model_ckpt/ramp_ckpt', help='Path of MOS prediction checkpoint.')
    parser.add_argument('--datastore_path', type=str, default='new_domain_datastore/', help='Path of MOS prediction checkpoint.')
    args = parser.parse_args()
    
    cp_path = 'model_ckpt/wav2vec_small.pt'
    my_checkpoint = args.checkpoint
    datastore_path = args.datastore_path
    datadir = args.datadir
    SSL_OUT_DIM = 768

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))
    print('Starting prediction')

    emb_list = []
    label_list = []
    sys_list = []
    
    with open(datadir, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(',')
        audio_path = parts[0]
        label = float(parts[1])
        system_name = (audio_path.split('/')[1].split('-')[0],)
        inputs = torchaudio.load(audio_path)[0].unsqueeze(0)
    
        inputs = inputs.to(device)

        with torch.no_grad():
            embs, _ = model(inputs, system_name)

        emb = embs.cpu().detach().numpy()[0]

        emb_list.append(emb)
        label_list.append(label)
        sys_list.append(system_name[0])

    emb_array = np.array(emb_list)
    label_array = np.array(label_list)
    sys_array = np.array(sys_list)

    print("emb_array.shape: ", emb_array.shape)
    print("label_array.shape: ", label_array.shape)
    print("sys_array.shape: ", sys_array.shape)

    os.makedirs(datastore_path, exist_ok=True)
    with open(os.path.join(datastore_path, "emb_array.pkl"), "wb") as f:
        pickle.dump(emb_array, f)
    with open(os.path.join(datastore_path, "label_array.pkl"), "wb") as f:
        pickle.dump(label_array, f)
    with open(os.path.join(datastore_path, "sys_array.pkl"), "wb") as f:
        pickle.dump(sys_array, f)

    return


if __name__ == '__main__':
    main()
