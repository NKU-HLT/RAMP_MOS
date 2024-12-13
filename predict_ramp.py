import os
import faiss
import fairseq
import argparse
import torch, torchaudio

from model.ramp import MosPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='model_ckpt/ramp_ckpt',
                        help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--datastore_path', type=str, default='datastore_profile',
                        help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--wavdir', type=str, default='example_samples',
                        help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='test_answer.txt',
                        help='Output filename for your answer.txt.')
    args = parser.parse_args()
    
    max_k = 400
    topk = 1
    SSL_OUT_DIM = 768
    cp_path = 'model_ckpt/wav2vec_small.pt'
    my_checkpoint = args.checkpoint
    wavdir = args.wavdir
    outfile = os.path.join(args.outfile)
    datastore_path = args.datastore_path
    
    if not os.path.exists(cp_path):
        os.system('wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P model_ckpt')
        os.system('wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P model_ckpt/')
        
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MosPredictor(ssl_model, SSL_OUT_DIM, max_k, datastore_path, topk=topk).to(device)
    model.eval()
    model.load_state_dict(torch.load(my_checkpoint))
    
    wavdir = args.wavdir
    predictions = {}

    print('start evaluate')
    for filename in os.listdir(wavdir):
        if filename.endswith('.wav'):  # Ensure we process only .wav files
            testwav = os.path.join(wavdir, filename)
            
            # Load the wav file
            wav, _ = torchaudio.load(testwav)
            inputs = wav.to(device)
            filenames = (filename.split('-')[0],)
            
            # Perform prediction
            with torch.no_grad():
                _, outputs = model(inputs, syslist=filenames)
            
            # Store predictions
            outputs = outputs.to('cpu').numpy()[0]
            predictions[filename] = outputs

    # generate answer.txt
    ans = open(outfile, 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    print(f"saved at {outfile} success!")
    ans.close()


if __name__ == '__main__':
    main()
