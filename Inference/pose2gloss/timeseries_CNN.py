import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, embedding_dim):
        super(TimeSeriesCNN, self).__init__()
        # self.emb1 = nn.Linear(input_channels, 512)
        # self.emb2 = nn.Linear(512, 256)
        # self.emb3 = nn.Linear(256, 128)

        self.conv1 = nn.Conv1d(input_channels, num_filters,
                               kernel_size=7,stride=2,padding=3)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2,
                               kernel_size=5,stride=2,padding=2)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4,
                               kernel_size=3,stride=2,padding=1)
        self.fc1 = nn.Linear(6144, embedding_dim)
        self.relu = nn.ReLU()

        self.cls1 = nn.Linear(embedding_dim, embedding_dim)
        self.cls2 = nn.Linear(embedding_dim, 210)
    def forward(self,src):
        # x = self.emb1(src.permute(0,2,1))
        # x = self.relu(x)
        # x = self.emb2(x)
        # x = self.relu(x)
        # x = self.emb3(x)
        # x = self.relu(x)

        # x = x.permute(0,2,1)

        x = self.conv1(src)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    input_channels = 68
    num_filters = 256
    kernel_size = 3
    embedding_dim = 64
    seq_len = 45
    model = TimeSeriesCNN(input_channels,
                          num_filters, 
                          kernel_size, 
                          embedding_dim)
    
    x = torch.ones((16,input_channels,seq_len))

    pred = model(x)

    print('end')

    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy input tensors (batch_size, input_channels, seq_length)
    input1 = torch.randn(32, input_channels, 45)
    input2 = torch.randn(32, input_channels, 45)
    labels = torch.randint(0, 2, (32,))  # Binary labels for contrastive loss

    # Forward pass
    output1 = model(input1)
    output2 = model(input2)

    # Compute the loss
    loss = criterion(output1, output2, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Contrastive loss: {loss.item()}')