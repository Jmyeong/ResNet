import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
from model import ResNet_My
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, test_loader, criterion, device):
    model.eval()  # 평가 모드로 전환
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad(): 
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

def plot_loss(train_loss, valid_loss, filename="loss_plot.png"):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
feature_map = None
       
def visualize_tensorboard(writer, feature_map, epoch, step, name='feature_map'):
    num_channels = feature_map.size(0)  # 64개 채널
    height, width = feature_map.size(-1), feature_map.size(-2)  # 8x8 크기

    # 8x8 그리드로 배열
    grid_size = int(np.sqrt(num_channels)) + 1  # 8x8 그리드로 가정
    image_grid = torch.zeros((grid_size * height, grid_size * width))  # 64개의 채널을 하나의 텐서로 합침
    feature_map_reshaped = feature_map.reshape(64, -1)[0].unsqueeze(0)
    image_grid_reshaped = image_grid.reshape(1, -1)
    image_grid_reshaped = image_grid_reshaped.squeeze(0)
    image_grid_reshaped[:feature_map_reshaped.size(-1)] = feature_map_reshaped.squeeze(0)
    total_len = image_grid_reshaped.size(-1)
    image_grid_reshaped = image_grid_reshaped.unsqueeze(0).reshape(-1, int(total_len ** 0.5), int(total_len ** 0.5))

    # TensorBoard에 이미지로 기록
    writer.add_image(name, image_grid_reshaped, global_step=step, dataformats='CHW')

 
def main():
    device = torch.device('cuda')
    
    summary_save_path = "./summary"
    writer = SummaryWriter(summary_save_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    
    batch_size = 64
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = ResNet_My(32, 1, 2, 3).to(device)

    print(model)
    
    epochs = 20
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    train_loss_list = []
    valid_loss_list = []
    
    def forward_hook(module, input, output):
        global feature_map
        feature_map = output
        return output
    
    try:
        hook_handle = model.layer4[2].conv2.register_forward_hook(forward_hook)
        print("Hook registered successfully")
    except AttributeError:
        print("Error: The specified layer (MLP) or encoder block is not available.")    


    for epoch in range(epochs):
        # Train
        model.train()
        base_loss = 0.0
        for iters, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if feature_map is not None:
                # print(feature_map[0][0].unsqueeze(0).shape)
                if (iters+1) % 100 == 0:
                    visualize_tensorboard(writer, feature_map, epoch, iters, name=f"feature_map/{epoch}/{iters}")
            pred = model(inputs)
            # category = torch.argmax(pred, dim=1)
            # print(category)
            
            loss = criterion(pred, labels)            
            loss.backward()
            base_loss += loss.item()
            optimizer.step()
        
        # Validation
        model.eval()
        valid_loss = 0.0
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            valid_loss += loss.item()
        
        print(f"epoch : {epoch+1}/{epochs}, Valid_loss : {valid_loss / len(valid_loader)}")
        print(f"epoch : {epoch+1}/{epochs}, Total_loss : {base_loss / len(train_loader)}")
    
        train_loss_list.append(base_loss / len(train_loader))
        valid_loss_list.append(valid_loss / len(valid_loader))

    plot_loss(train_loss=train_loss_list, valid_loss=valid_loss_list, filename="./Loss.png")
    # torch.save(model.state_dict(), "./checkpoint/ViT.pth")
    accuracy, avg_loss = evaluate(model=model, test_loader=test_loader, criterion=criterion, device=device)
    print(f"accuracy : {accuracy}, avg_loss : {avg_loss}")
    
    # visualize_vit_feature_map(model, test_loader, device, save_path="./")

if __name__ == "__main__":
    main()