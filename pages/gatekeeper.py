import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def train_guard_model():
    # 1. Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load dataset
    dataset_path = r"C:\Users\user\Desktop\CODE4FOOD\capstone_project\Capstone-Project_Group_1_C4FS\pages\Guardrail Dataset"
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # 3. Print mapping so the user knows exactly how PyTorch assigned the indices
    print("\n" + "="*50)
    print("🚨 CRITICAL MAPPING INFORMATION 🚨")
    print("Copy the dictionary below into your poultry_guard_app.py script!")
    print(f"GATEKEEPER_CLASSES = {{")
    for class_name, index in dataset.class_to_idx.items():
        print(f"    {index}: '{class_name}',")
    print(f"}}")
    print("="*50 + "\n")
    
    print(f"Total images found: {len(dataset)}")

    # 4. Prepare data loader and model architecture
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    model = models.mobilenet_v3_small(weights="DEFAULT")
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(5): 
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f}")

    # 6. Save weights
    torch.save(model.state_dict(), 'gatekeeper_guard.pth')
    print("Guard model successfully trained and saved as 'gatekeeper_guard.pth'.")

if __name__ == '__main__':
    train_guard_model()