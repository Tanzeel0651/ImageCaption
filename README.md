# 🖼️ Image Caption Generator using ResNet + LSTM (PyTorch)

This project implements an end-to-end **image captioning model** from scratch using a **ResNet encoder** and a **stacked LSTM decoder** in PyTorch. It takes an input image and generates a natural language caption. The system is trained on custom image-caption pairs and includes support for GPU acceleration and sampling-based generation.

## ✨ Features

- ✅ Custom ResNet-18 image encoder (headless)
- ✅ Stacked LSTM decoder for sequential caption generation
- ✅ TorchText-based tokenizer and vocabulary
- ✅ Captions generated using greedy decoding
- ✅ GPU support with CUDA-enabled PyTorch builds

## 🧠 Architecture

```
[Image] ──► ResNet18 Encoder ──► Feature Vector
                                ↓
                Word Embedding + Image Feature → LSTM Decoder → Caption
```

- **Encoder**: Custom-built ResNet-18 (no classifier head)
- **Decoder**: 3-layer LSTM stack + linear projection to vocab
- **Input**: Caption tokens prepended with `<start>` and appended with `<end>`
- **Output**: Generated token sequence decoded into a sentence

## 📁 Project Structure

```
image_caption/
├── dataloader.py         # Dataset + Vocabulary + Collate function
├── architecture.py       # ResNet encoder and LSTM decoder definitions
├── trainer.py            # Training + generation logic
├── captions.txt          # Text file with image-caption pairs
├── Images/               # Folder of images used for training
└── README.md             # This file
```

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/image-captioning-lstm
cd image-captioning-lstm
```

### 2. Create and activate the environment

```bash
conda create -n imagecap python=3.11
conda activate imagecap
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu121
```

Make sure your system has compatible NVIDIA drivers (CUDA ≥ 12.1) and `nvidia-smi` works.

## 📄 Prepare Your Dataset

1. Place all training images inside the `Images/` directory.
2. Create a `captions.txt` file in the following format:

```
image1.jpg, A person riding a red bicycle.
image2.jpg, Two dogs running through a field.
```

Each line maps an image filename to its corresponding caption.

## 🚀 Train the Model

```bash
python trainer.py
```

Training will begin and print output like:

```
Epoch: 200/200 -- Batch: (1011/1012) -- Loss: 0.3252 -- Epoch Loss: 329.0056
```

## 🧪 Generate Captions

After training, the model can generate captions interactively. Example outputs:

```
Generated Caption: <start> the man is wearing a white suit and walking on the street . <end>
Actual Caption:   The man is wearing a white suit and walking on the street .

Generated Caption: <start> two dogs play with a bottle . <end>
Actual Caption:   Two dogs play with a bottle .

Generated Caption: <start> young woman jumping on trampoline . <end>
Actual Caption:   Young woman jumping on trampoline .

Generated Caption: <start> a young boy in a brown jacket playing with snow . <end>
Actual Caption:   A young boy in a brown jacket playing with snow .

Generated Caption: <start> two tan and white dogs compete outside . <end>
Actual Caption:   Two yellow dogs running on sand .

Generated Caption: <start> the man holds the tennis mother in the parasail . <end>
Actual Caption:   The man has a black and white scarf around his neck .
```

> 🧠 The model shows high alignment on familiar visual scenes and nouns, with some hallucination in complex cases — indicating future benefit from attention mechanisms or object reasoning.

## 💾 Saving and Loading

### Save:
```python
torch.save(model.state_dict(), "caption_model.pth")
torch.save(vocab, "vocab.pth")
```

### Load:
```python
model.load_state_dict(torch.load("caption_model.pth"))
vocab = torch.load("vocab.pth")
```

Tokenizer can be reloaded with:
```python
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")
```

## 🧠 Future Improvements (TODO)

- [ ] Add BLEU, CIDEr, or METEOR evaluation metrics
- [ ] Implement top-k and top-p (nucleus) sampling
- [ ] Add attention mechanism to decoder
- [ ] Move to PyTorch Lightning for modular training
- [ ] Export results for validation dataset

## 🙌 Acknowledgments

Built by [Tanzeel](https://github.com/yourusername), using PyTorch, TorchText, and torchvision.

## 📄 License

This project is licensed under the MIT License.