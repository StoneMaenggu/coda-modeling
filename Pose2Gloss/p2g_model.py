import torch 
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, emb_dim*4),
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim*2),
            nn.ReLU(),
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            )
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [src_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.embedding =nn.Sequential(
        #                             nn.Linear(output_dim, emb_dim*2),
        #                             nn.ReLU(),
        #                             nn.Linear(emb_dim*2, emb_dim),
        #                             nn.ReLU(),
        #                             )
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input.argmax(2)))
        # embedded: [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [seq_len, batch_size, hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        
        prediction = self.fc_out(output.squeeze(0))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        # teacher_forcing_ratio: probability to use teacher forcing
        
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = trg.shape[2]
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = F.one_hot(output.argmax(1), trg_vocab_size)
            input = trg[t] if teacher_force else top1
        
        return outputs
    
    def init_weights(self):
        pass
        # for name, param in self.named_parameters():
        #     if 'weight_ih' in name:
        #         nn.init.xavier_uniform_(param.data)
        #     elif 'weight_hh' in name:
        #         nn.init.orthogonal_(param.data)
        #     elif 'bias' in name:
        #         nn.init.constant_(param.data, 0)
        #         if 'bias_hh_l0' in name:
        #             nn.init.constant_(param.data, 1.0)
        # print()

                    
if __name__ == '__main__':
    from signlang_dataloader import AIHUB_SIGNLANG_DATASET, collate_fn
    from torch.utils.data import DataLoader
    train_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='train',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN'])

    train_loader = DataLoader(train_dataset,32,shuffle=True, collate_fn=collate_fn)
    pose_batch, gloss_batch = next(iter(train_loader))

    enc = Encoder(input_dim=50, 
                  emb_dim=256, 
                  hid_dim=512, 
                  n_layers=2, 
                  dropout=0.5)
    dec = Decoder(output_dim=3332,
                  emb_dim=256,
                  hid_dim=512,
                  n_layers=2,
                  dropout=0.5)
    model = Seq2Seq(enc,dec,'cuda:0').to('cuda:0')
    gloss_batch = gloss_batch.to('cuda:0')    
    pose_batch = pose_batch[:,:,:,:2].reshape(-1,32,50).to('cuda:0')    

    pred = model(pose_batch, gloss_batch)

    print('end')