import torch
from torch.utils.data import Dataset

import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from torchvision import transforms

import os
#os.chdir("/home/tanzeel/Videos/image_caption/")


def image_caption_dict():
    caption = {}
    with open("captions.txt", 'r') as file:
        for line in file.readlines():
            if ".jpg" in line:
                line = line.split(",")
                image = 'Images/'+line[0]
                cap = ' '.join(line[1:]).strip()
                caption[image] = cap
        return caption

def read_image(image_path):
    img = Image.open(image_path).resize((224,224))
    return img

def make_vocab(all_captions, basic_tokenizer):
    def yield_tokens(data_iter):
        for caption in data_iter:
            yield ["<start>"] + basic_tokenizer(caption) + ["<end>"]

    vocab = build_vocab_from_iterator(yield_tokens(all_captions), specials=["<pad>", "<start>", "<end>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

class ImageCaptionDataset(Dataset):
    def __init__(self):
        self.image_caption = image_caption_dict()
        self.images = list(self.image_caption.keys())
        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                                std=[0.229, 0.224, 0.225])
        ])
        self.basic_tokenizer = get_tokenizer("basic_english")
        self._vocab = make_vocab(list(self.image_caption.values()), self.basic_tokenizer)

    def __len__(self):
        return len(self.images)

    @property
    def vocab_size(self):
        return self._vocab.__len__()

    @property
    def vocab(self):
        return self._vocab

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.image_caption[image]

        image_tensor = self.transform(read_image(image))
        tokens = ["<start>"] + self.basic_tokenizer(caption) + ["<end>"]
        token_idx = torch.tensor([self._vocab[token] for token in tokens])
        
        return {
            "image": image_tensor, 
            "caption_idx": token_idx,
            "image_path": image,
            "caption": caption
        }

def collate_fn(batch, pad_token_idx):
    image_stack = []
    caption_idx_stack = []
    path_stack = []
    caption_stack = []

    for item in batch:
        image_stack.append(item["image"])
        caption_idx_stack.append(item["caption_idx"])
        path_stack.append(item["image_path"])
        caption_stack.append(item["caption"])

    padded_caption = pad_sequence(caption_idx_stack, batch_first=True, padding_value=pad_token_idx)
    return {
        "image": torch.stack(image_stack),
        "caption_idx": padded_caption,
        "image_path": path_stack,
        "caption: ": caption_stack
        }



if __name__ == '__main__':
    dataloader = Dataloader()

