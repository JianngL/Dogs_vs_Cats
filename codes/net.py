import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import create_dataloader
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(46656, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, train_loader, val_loader, use_cuda, epoches=10, lr=1e-4):
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epoches + 1):
        print("Epoch", epoch)
        loss_list = []
        # train_loss
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_list = np.array(loss_list)
        loss_cur_epoch = np.mean(loss_list)
        # val_loss and accuracy
        val_loss, corr_num = 0.0, 0
        for batch_idx, (data, label) in enumerate(val_loader):
            with torch.no_grad():
                data, label = data.to(device), label.to(device)
                output = model(data)
                corr_num += sum(output.argmax(dim=1) == label).item()
                loss = F.nll_loss(output, label)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        accuracy = corr_num / (len(val_loader.dataset))
        print("train_loss: {:.4f} val_loss:{:.4f} val_acc:{:.4f}".format(loss_cur_epoch, val_loss, accuracy))
        torch.save(model.state_dict(), "cats_dogs_cnn_val_acc_{:.4f}.pt".format(accuracy))

if __name__ == "__main__":
    model = Net()
    use_cuda = torch.cuda.is_available()
    train_loader = create_dataloader(is_trian=True)
    val_loader = create_dataloader(is_trian=False)
    train(model, train_loader, val_loader, use_cuda)