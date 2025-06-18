import torch, json, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image

training = True

class BoardReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 27)

    def forward(self, x):
        return self.backbone(x).view(-1, 9, 3)


class TicTacToeDataset(Dataset):
    def __init__(self, image_path, labels_file, transform=None):
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.image_path = image_path
        self.transform = transform
        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_loc = os.path.join(self.image_path, img_name)

        image = Image.open(image_loc)
        if self.transform:
            image = self.transform(image)
        
        grid = torch.tensor(self.labels[img_name], dtype=torch.long).flatten()
        return image, grid


def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TicTacToeDataset('data/images', 'data/labels.json', transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = BoardReader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(50):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    torch.save(model.state_dict(), 'board_reader.pth')
    return model


def predict_board(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=2).squeeze().numpy()
        return predicted.reshape(3, 3)

if training:
    model = train_model()

model = BoardReader()
model.load_state_dict(torch.load('board_reader.pth'))
board_state = predict_board('test_image.jpeg', model)
print(board_state)
