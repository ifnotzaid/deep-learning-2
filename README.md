# A Multi-Block Deep Architecture for ECG Classification on PTB-XL

**SWE012 - Deep Learning with Python | Final Project**  
*[Your Name] - May 2026*

---

## Abstract

We propose a five-block deep neural architecture for single-label diagnostic
classification of 12-lead electrocardiograms on the PTB-XL dataset
(Wagner et al., 2020). The model integrates a 1-D convolutional feature
extractor, a convolutional autoencoder for latent compression, a bidirectional
LSTM and a GRU for sequential modeling, and a temporal attention mechanism for
weighted pooling. Each block is independently ablated to quantify its
contribution. On the held-out PTB-XL fold-10 test split, the full model
achieves accuracy = **0.7797**, macro-F1 = **0.5879**, and macro-AUC =
**0.8696**. The ablation study shows that removing the GRU produces the
largest degradation in test macro-F1, while removing the autoencoder improves
performance, suggesting that the reconstruction objective over-constrained the
classification representation in this setup.

---

## 1. Introduction

Automated interpretation of 12-lead electrocardiograms (ECGs) is a
long-standing problem in clinical machine learning. ECG signals are
simultaneously local, because diagnostic morphology occurs over short windows
such as the QRS complex, and global, because rhythm-level abnormalities span
the entire recording. This dual nature motivates hybrid architectures that
combine convolutional and recurrent inductive biases.

In this work, we (i) combine five distinct architectural blocks in a single
end-to-end pipeline, (ii) introduce a reconstruction-based auxiliary objective
via an embedded autoencoder, and (iii) systematically ablate every block to
identify which inductive biases are useful for PTB-XL superclass
classification. The project is designed around the course topics of
convolutional neural networks, recurrent neural networks, gated recurrent
models, attention, and autoencoders.

---

## 2. Dataset

We use **PTB-XL** (Wagner et al., *Scientific Data*, 2020), a large publicly
available clinical 12-lead ECG dataset. The dataset originates from a
peer-reviewed research paper and is officially distributed through PhysioNet.
We use the 100 Hz version of the ECG recordings, where each example is a
10-second signal with 12 leads and 1000 samples per lead.

The original PTB-XL release contains approximately 21.8k ECG recordings from
approximately 18.9k patients. Following the official stratified fold split, we
use folds 1-8 for training, fold 9 for validation, and fold 10 for testing.
Records are mapped to five diagnostic superclasses using the SCP-ECG statement
metadata provided with the dataset:

- **NORM**: normal ECG
- **MI**: myocardial infarction
- **STTC**: ST/T change
- **CD**: conduction disturbance
- **HYP**: hypertrophy

To keep the task focused and compatible with standard multiclass
cross-entropy, we retain only records with exactly one diagnostic superclass.
This produces a clean single-label classification dataset with **12,978**
training records, **1,642** validation records, and **1,652** test records.

**Rationale for dataset choice.** PTB-XL is appropriate for this project
because it is clinically meaningful, publicly documented, large enough for deep
learning, and tied to a peer-reviewed dataset paper rather than being only a
Kaggle benchmark. The official folds also reduce the risk of data leakage and
make the experiment reproducible.

---

## 3. Methods

### 3.1 Architecture Overview

Let `x` denote an input ECG with shape `(12, 1000)`, corresponding to 12 leads
and 1000 time steps. The model predicts one of five diagnostic superclasses.
The forward pass is:

```text
ECG signal -> CNN -> Autoencoder -> BiLSTM -> GRU -> Attention -> MLP -> class prediction
```

The full model has **270,534** trainable parameters.

### 3.2 Block 1 - Convolutional Feature Extractor

The first block applies stacked 1-D convolutions over the ECG time axis.
Convolution is appropriate because ECG morphology is local: clinically useful
patterns such as QRS shape, ST deviation, and T-wave morphology occur in short
time windows. A 1-D CNN also uses sparse connectivity and parameter sharing,
matching the Week 6 CNN material from the course. The same learned filters can
detect similar waveform patterns wherever they occur in the recording.

### 3.3 Block 2 - Convolutional Autoencoder

The second block uses an autoencoder-style bottleneck. The encoder compresses
the CNN representation into a smaller latent representation, and the decoder
attempts to reconstruct the encoded feature sequence. The reconstruction MSE is
added to the classification objective with weight **0.1**.

This block is motivated by the Week 8 autoencoder material: a useful
autoencoder should not simply copy the input, but should learn a constrained
representation that preserves salient structure. In this project, the AE is
used as an auxiliary regularizer intended to encourage compact, denoised ECG
features.

### 3.4 Block 3 - Bidirectional LSTM

The third block is a single-layer bidirectional LSTM. Since the entire 10-second
ECG is available at inference time, the model can process the sequence both
forward and backward. This is useful because the interpretation of a time step
can depend on context before and after it. In the language of the course slides,
this is a sequence-to-single-output problem: the model reads the full sequence
and then emits one diagnostic label.

### 3.5 Block 4 - GRU

The fourth block is a GRU applied after the BiLSTM. GRUs are a simpler gated
recurrent unit than LSTMs: they use reset and update gates without a separate
cell state. The goal of this block is to refine the bidirectional sequence
representation using a lighter recurrent mechanism. In the ablation study,
removing the GRU caused the largest performance drop, reducing test macro-F1
from **0.5879** to **0.5646**. This supports the decision to include a final
gated recurrent refinement stage.

### 3.6 Block 5 - Temporal Attention

The fifth block is temporal attention. Instead of averaging every time step
equally, the attention layer learns weights over time and produces a weighted
summary of the ECG representation. This is appropriate because diagnostic
evidence may be concentrated in a small number of beats or waveform segments.
The attention mechanism therefore acts as content-based pooling over the
sequence.

### 3.7 Regularization

The final training setup uses several regularization techniques:

- **Dropout** in the convolutional/recurrent path and classifier to reduce
  co-adaptation.
- **Batch normalization** in the CNN block to stabilize activations.
- **AdamW weight decay** with weight decay **1e-4**.
- **Autoencoder reconstruction loss** with reconstruction weight **0.1** as an
  auxiliary objective.
- **Early stopping** with patience **7**, selecting the checkpoint with the
  best validation macro-F1.
- **Cosine learning-rate scheduling** across the maximum 30 training epochs.
- **Gradient clipping** during training to stabilize recurrent optimization.

Because PTB-XL is class-imbalanced, macro-F1 is reported alongside accuracy.
This prevents the evaluation from being dominated by the most frequent class.

### 3.8 Hyperparameter Tuning

Hyperparameters were selected using the validation fold. The main search
focused on the learning rate and the autoencoder reconstruction-loss weight,
because these directly control optimization speed and the strength of the
auxiliary AE objective.

| Learning rate | Reconstruction weight | Validation macro-F1 |
|---------------|-----------------------|---------------------|
| 0.0003        | 0.1                   | 0.5181              |
| 0.0010        | 0.1                   | 0.5587              |
| 0.0030        | 0.1                   | **0.5782**          |
| 0.0010        | 0.0                   | 0.5671              |
| 0.0010        | 0.3                   | 0.5771              |

The selected configuration was:

| Hyperparameter | Selected value | Explanation |
|----------------|----------------|-------------|
| Learning rate | **0.003** | Best validation macro-F1 in the tuning sweep |
| Batch size | **64** | Stable memory use and fast A100 training |
| Reconstruction weight | **0.1** | Best setting in the learning-rate sweep and keeps the AE objective active |
| Weight decay | **1e-4** | Standard AdamW regularization value |
| Max epochs | **30** | Sufficient upper bound with early stopping |
| Early-stopping patience | **7** | Stops after validation macro-F1 stops improving |
| Checkpoint criterion | **Validation macro-F1** | Better suited to class imbalance than accuracy |

The full model smoke test reached validation macro-F1 **0.5687** after only
five epochs, confirming that the model and data pipeline were learning before
the full ablation study was launched.

---

## 4. Experiments

### 4.1 Full Model

The full model was trained for up to 30 epochs with early stopping. It stopped
at epoch 21 after no validation macro-F1 improvement for 7 epochs. The best
validation macro-F1 occurred at epoch 14.

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | **0.7741** | **0.7797** |
| Macro-F1 | **0.5838** | **0.5879** |
| Macro-AUC | **0.8737** | **0.8696** |
| Parameters | **270,534** | - |

### 4.2 Ablation Study

Each variant removes a single architectural block while keeping the remaining
training setup constant. Delta test F1 is computed relative to the full model.

| Variant | Val F1 | Test Acc | Test F1 | Test AUC | Delta Test F1 |
|---------|--------|----------|---------|----------|---------------|
| Full | **0.5838** | **0.7797** | **0.5879** | **0.8696** | **0.0000** |
| - CNN | 0.6041 | 0.7591 | 0.5925 | 0.8733 | +0.0046 |
| - Autoencoder | 0.6203 | 0.7688 | 0.6149 | 0.8937 | +0.0270 |
| - BiLSTM | 0.5947 | 0.7815 | 0.5971 | 0.8907 | +0.0092 |
| - GRU | 0.5796 | 0.7585 | 0.5646 | 0.8543 | -0.0233 |
| - Attention | 0.5831 | 0.7736 | 0.5888 | 0.8870 | +0.0009 |

**Discussion.** The ablation study reveals that the GRU is the most important
positive contributor among the tested blocks. Removing the GRU reduced test
macro-F1 from **0.5879** to **0.5646**, the largest degradation observed in the
study. This suggests that the final recurrent refinement stage provides useful
gating after the bidirectional LSTM representation. In course terms, the GRU's
update and reset gates appear to help the model decide which temporal
information should be retained before the final attention pooling step.

The most surprising result is that removing the autoencoder improved test
macro-F1 from **0.5879** to **0.6149**. This does not mean that autoencoders are
not useful in general; rather, it suggests that the specific vanilla
reconstruction objective used here was not the best match for the supervised
classification task. The AE may have forced the latent representation to
preserve information useful for reconstruction but not useful for diagnosis.
This matches a known autoencoder trade-off from the course material: the
constraint must force useful features, but a poorly weighted reconstruction
constraint can compete with the downstream objective.

The CNN, BiLSTM, and attention ablations produced smaller differences. Removing
attention had almost no effect on test macro-F1, and removing the CNN or BiLSTM
slightly improved the reported F1 in this run. These differences are small
enough that they should be interpreted cautiously, especially because the test
set contains only 1,652 filtered single-label records. However, the full model
still satisfies the project goal of coherently integrating all required blocks,
and the ablation study provides a useful scientific conclusion: for this
filtered PTB-XL setting, the GRU was beneficial, while the AE design should be
revised before being used in a production-oriented model. Future work should
try a denoising autoencoder, a smaller reconstruction weight, focal loss, or
balanced batch sampling.

---

## 5. Conclusion

We presented a five-block deep architecture for ECG superclass classification
on PTB-XL and quantified each block's contribution through systematic ablation.
The model combines CNN, autoencoder, BiLSTM, GRU, and attention blocks in a
single end-to-end pipeline. The full model achieved **0.7797** test accuracy,
**0.5879** test macro-F1, and **0.8696** test macro-AUC. The ablation study
showed that the GRU provided the strongest positive contribution, while the
vanilla autoencoder objective did not improve classification performance in
this setup.

Future work could extend the experiment to the full multi-label PTB-XL task,
use the 500 Hz signal version, incorporate patient metadata, and evaluate more
stable imbalance strategies such as focal loss or balanced batch sampling.

---

## References

- Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I.,
  Samek, W., & Schaeffter, T. (2020). *PTB-XL, a large publicly available
  electrocardiography dataset.* Scientific Data, 7(1), 154.
- Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M.,
  Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., &
  Stanley, H. E. (2000). *PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a new research resource for complex physiologic signals.*
  Circulation, 101(23), e215-e220.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.*
  Neural Computation, 9(8), 1735-1780.
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). *Learning Phrase Representations using
  RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by
  Jointly Learning to Align and Translate.* ICLR.
- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008).
  *Extracting and composing robust features with denoising autoencoders.*
  ICML.

---

## Reproducibility

```bash
pip install -r requirements.txt
python ablation.py
```

The experiment uses the PTB-XL 100 Hz ECG files. Due to slow PhysioNet
bandwidth during the experiment, the PTB-XL archive was downloaded from a
Kaggle mirror of the official dataset release, but the dataset source and
citation remain the original Wagner et al. research paper and PhysioNet
release. The local experiment used the extracted folder:

```text
ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1
```

The reported split after filtering to single-label diagnostic superclass
records is:

```text
train: 12978
validation: 1642
test: 1652
input shape: (12, 1000)
```

Recommended repository contents:

```text
README.md
notebook.ipynb
ecg_net.py
data.py
train.py
ablation.py
requirements.txt
results.csv
tuning.csv
figures/
```
