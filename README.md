# 📈 **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

Welcome to the official implementation of:

- **RAMP: Retrieval-Augmented MOS Prediction via Confidence-based Dynamic Weighting**
- **RAMP+: Retrieval-Augmented MOS Prediction with Prior Knowledge Integration**

This repository provides everything you need to evaluate and predict MOS (Mean Opinion Scores) efficiently with the **RAMP+ model**, leveraging **prior knowledge integration** to improve accuracy and handling out-of-domain (OOD) data gracefully. 

---

## 🚀 **Quick Evaluation Guide**

### 1. **Download Code and Checkpoint**

Get started by cloning the repository and downloading [the necessary checkpoint file](https://drive.google.com/file/d/1-l5huyOHWXFtSlGfHnHJVA7dcVS2RSdM/view?usp=sharing) to `RAMP_MOS/model_ckpt`:



```bash
git clone https://github.com/jiusansan222/RAMP_MOS.git
cd RAMP_MOS
```

### 2. **Set Up the Environment**

Get the environment ready to go with Python 3.9.12 and required dependencies:

```bash
conda create -n RAMP python=3.9.12
conda activate RAMP

# Clone and install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# Install additional requirements
pip install -r requirements.txt
```

### 3. **Run Predictions**

Use `predict_ramp.py` to generate predictions. Just point to the checkpoint, datastore, and WAV files!

```bash
python predict_ramp.py --wavdir path/to/wav --outfile path/to/answer
```
### Parameters:

- **`--checkpoint`**: The path to the downloaded model checkpoint. 

- **`--datastore_path`**: The path to the datastore. In this case, we have provided a BVCC datastore in `datasore_profile` as default, which makes it easier for you to evaluate the model. 

- **`--wavdir`**: The path to the directory containing the WAV files you want to predict on. 

- **`--outfile`**: The path where the prediction results will be saved. 


---

## 🌐 **Cross-Domain Prediction**

One of the key strengths of **RAMP+** is its robust performance on **out-of-domain (OOD) data**, making it easy to evaluate new domain speech without retraining. Let’s see how to set it up for OOD evaluation. 

### 1. **Prepare Labels**

Create a `label.txt` file with the format below to include the paths to your WAV files and their respective MOS scores.

Example format:

```
path_to_wav/systemid-uttid.wav, mos

# Example
example_samples/sys64e2f-utt491a78a.wav,4.0
example_samples/sys64e2f-utt8485f83.wav,3.625
example_samples/sys7ab3c-utt1417b69.wav,3.375
example_samples/sys7ab3c-uttb548b8d.wav,2.0
...
```

### 2. **Create Datastore**

Generate the datastore by running the command below:

```bash
python get_datastore.py --datadir path/to/label.txt --checkpoint path/to/ckpt --datastore_path path/to/datastore
```

### 3. **Evaluate**

Evaluate OOD data by running the prediction script with the new datastore:

```bash
python predict_ramp.py --checkpoint path/to/ckpt --datastore_path path/to/new_datastore --wavdir path/to/wav --outfile path/to/answer
```

---

## 📚 **Acknowledgments**

This project builds upon prior work from the [nii-yamagishilab/mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl) repository. We thank them for their contributions! 

## 📑 **Citation**

If you use RAMP+ in your research, please cite us as follows:

```bibtex
@inproceedings{wang23r_interspeech,
  title     = {RAMP: Retrieval-Augmented MOS Prediction via Confidence-based Dynamic Weighting},
  author    = {Hui Wang and Shiwan Zhao and Xiguang Zheng and Yong Qin},
  year      = {2023},
  booktitle = {INTERSPEECH 2023},
  pages     = {1095--1099},
  doi       = {10.21437/Interspeech.2023-851},
  issn      = {2958-1796},
}
```

---

Enjoy using **RAMP+** for enhanced and efficient MOS predictions! 🎉
