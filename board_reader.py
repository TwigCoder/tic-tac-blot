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

        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,27)
        )

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
        transforms.RandomRotation(5),
        transforms.RandomPerspective(0.5),
        transforms.ColorJitter(0.4, 0.4, 0.3),
        transforms.RandomAffine(0, (0.1, 0.1), (0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TicTacToeDataset('data/images', 'data/labels.json', transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = BoardReader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    model.train()
    best_loss = float('inf')

    for epoch in range(50):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'board_reader_inc.pth')
    
    torch.save(model.state_dict(), 'board_reader_inc.pth')
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
    predictions = []

    for angle in [0,-2,2]:
        rotated = image.rotate(angle)
        tensor = transform(rotated).unsqueeze(0)    

        with torch.no_grad():
            output = model(tensor)
            predicted = torch.argmax(output, dim=2).squeeze().numpy()
            predictions.append(predicted)
            return predicted.reshape(3, 3)
        
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    final_pred = torch.argmax(avg_pred, dim=2).squeeze().numpy()

    return final_pred.reshape(3, 3)

if training:
    model = train_model()

model = BoardReader()
model.load_state_dict(torch.load('board_reader_inc.pth'))
board_state = predict_board('test.jpeg', model)
print(board_state)
