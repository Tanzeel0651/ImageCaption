import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import ImageCaptionDataset, collate_fn
from architecture import ResNet_18, RNN

import random
from PIL import Image

class Trainer():
    def __init__(self, model, optimizer, loss_function, vocab_size):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compute_loss(self, inputs):
        image_tensor = inputs["image"].to(self.device)
        caption = inputs["caption_idx"].to(self.device)

        if caption.ndim==1:
            caption = caption.unsqueeze(0)

        input_caption = caption[:, :-1]
        target_caption = caption[:, 1:]

        logits = self.model(image_tensor, input_caption)
        return self.loss_function(logits.view(-1, logits.size(-1)), target_caption.reshape(-1))
    
    def train(self, dataloader, num_epochs):
        self.model.train()
        total_batches = len(dataloader)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for idx, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                if idx%2==0:
                    print(f"\rEpoch: {epoch+1}/{num_epochs} -- Batch: ({idx+1}/{total_batches}) -- Loss: {loss.item():.4f} -- Epoch Loss: {epoch_loss:.4f}", end="", flush=True)
                epoch_loss += loss.item()
            print()
        

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super(ImageCaptionModel, self).__init__()
        self.encoder = ResNet_18(3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, 512).to(self.device)
        self.decoder = RNN(input_size=512, hidden_size=512, vocab_size=vocab_size).to(self.device)

    def forward(self, image, caption):
        image_feat = self.encoder(image)
        caption_embed = self.embedding(caption)
        
        T = caption.shape[1]
        image_feat = image_feat.unsqueeze(1)
        image_feat_broadcast = image_feat.repeat(1,T,1)

        enriched_input = caption_embed + image_feat_broadcast
        logits = self.decoder(enriched_input) 

        return logits

    def generate(self, image, start_token, end_token, vocab):
        itos = vocab.get_itos()
        image_feat = self.encoder(image.to(self.device)).unsqueeze(1)
        caption = [start_token]
        index = 0
        while caption[-1] != end_token and index<30:
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(self.device)
            caption_embed = self.embedding(caption_tensor).to(self.device)
            T = len(caption)
            image_feat = image_feat
            image_feat_broadcast = image_feat.repeat(1,T,1)

            enriched_input = caption_embed + image_feat_broadcast
            logits = self.decoder(enriched_input)[:, -1, :]
            logits_softmax = nn.functional.softmax(logits/0.7, dim=-1)
            #top_p, top_class = torch.max(logits_softmax, dim=-1)
            next_token = torch.multinomial(logits_softmax, num_samples=1)
            caption.append(next_token.item())
            index += 1
        
        return ' '.join([itos[idx] for idx in caption]) 

 
def test_random_instance(model, dataset, vocab, start_index, end_index):
    ## Testing on random instances
    rand_index = random.randint(0, len(dataset)-1)
    image, _, image_path, caption = dataset[rand_index].values()

    gen_caption = model.generate(image.unsqueeze(0), start_index, end_index, vocab)
    print("Generated Caption:", gen_caption)
    print("Actual Caption:", caption)
    
    #Image.open(image_path).show()

def main():
    dataset = ImageCaptionDataset()
    vocab_size = dataset.vocab_size
    vocab = dataset.vocab

    start_index = vocab["<start>"]
    end_index = vocab["<end>"]
    pad_index = vocab["<pad>"]

    model = ImageCaptionModel(vocab_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_index)

    dataloader = torch.utils.data.DataLoader(
                    dataset = dataset,
                    batch_size=8,
                    shuffle=True,
                    collate_fn=lambda batch: collate_fn(batch, pad_token_idx=pad_index)
    )
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      vocab_size=vocab_size
    )

    trainer.train(dataloader, num_epochs=200)
    
    torch.save(model.state_dict(), "caption_model.pth")
    torch.save(vocab, "vocab.pth")

    model.eval()
    with torch.no_grad():
        while True:
            query = input("Try another instance (Y/n): ")
            if query.lower().startswith("n"):
                break
            test_random_instance(model, dataset, vocab, start_index, end_index)

        
if __name__=="__main__":
    main()
