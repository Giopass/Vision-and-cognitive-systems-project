import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_data(train_val_dir, test_dir):
    """Load and merge all data from both directories using only folder names as labels"""
    
    all_image_paths = []
    all_labels = []
    
    # Load from train_val directory (city/query/images/)
    logger.info(f"Loading data from train_val: {train_val_dir}")
    for city_folder in sorted(os.listdir(train_val_dir)):
        city_path = os.path.join(train_val_dir, city_folder)
        if not os.path.isdir(city_path):
            continue
            
        images_dir = os.path.join(city_path, 'query', 'images')
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in os.listdir(images_dir) 
                      if any(f.endswith(ext) for ext in image_extensions)]
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            all_image_paths.append(image_path)
            all_labels.append(city_folder)  # Use folder name as label
        
        logger.info(f"  {city_folder}: {len(image_files)} images from train_val")
    
    # Load from test directory (city/dataset/)
    logger.info(f"Loading data from test: {test_dir}")
    for city_folder in sorted(os.listdir(test_dir)):
        city_path = os.path.join(test_dir, city_folder)
        if not os.path.isdir(city_path):
            continue
            
        images_dir = os.path.join(city_path, 'dataset')
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in os.listdir(images_dir) 
                      if any(f.endswith(ext) for ext in image_extensions)]
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            all_image_paths.append(image_path)
            all_labels.append(city_folder)  # Use folder name as label
        
        logger.info(f"  {city_folder}: {len(image_files)} images from test")
    
    logger.info(f"Total loaded: {len(all_image_paths)} images from {len(set(all_labels))} cities")
    
    # Print class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.info(f"  {label}: {count} images")
    
    return all_image_paths, all_labels

class SimpleGeoLocationDataset(Dataset):
    """Simple dataset that takes pre-loaded image paths and labels"""
    
    def __init__(self, image_paths, labels, transform=None, label_encoder=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = label_encoder
        
        # Encode labels
        if self.label_encoder is not None:
            self.encoded_labels = self.label_encoder.transform(self.labels)
        else:
            raise ValueError("label_encoder is required")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PlaNetCNN(nn.Module):
    """PlaNet-style CNN for geolocation classification"""
    
    def __init__(self, num_classes, pretrained=True):
        super(PlaNetCNN, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """Get data transforms"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=5, device='cuda', save_path='best_model.pth'):
    """Train the model"""
    
    model.to(device)
    
    # Tracking metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_preds += target.size(0)
            correct_preds += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_preds/total_preds:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_preds / total_preds
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for data, target in test_pbar:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.numpy())
    
    accuracy = accuracy_score(targets, predictions)
    return accuracy, predictions, targets

def plot_results(history, class_names, predictions, targets):
    """Plot training history and confusion matrix"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    ax4.bar(range(len(class_names)), class_acc)
    ax4.set_title('Per-Class Accuracy')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training pipeline"""
    
    config = {
        'train_val_dir': r'C://Users//giopa//Downloads//Compressed//train_val',  
        'test_dir': r'C://Users//giopa//Downloads//Compressed//test',            
        'train_ratio': 0.8,  
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 2
    }
    
    print("Starting PlaNet Geolocation Training")
    print(f"Using device: {config['device']}")
    
    # Load and merge all data
    print("\nLoading and merging all data...")
    all_image_paths, all_labels = load_all_data(config['train_val_dir'], config['test_dir'])
    
    if len(all_image_paths) == 0:
        print("No images found!")
        return
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Found {num_classes} cities: {list(label_encoder.classes_)}")
    
    # Split all data into train and test (80/20)
    print(f"\nSplitting data: {config['train_ratio']*100:.0f}% train, {(1-config['train_ratio'])*100:.0f}% test")
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=(1 - config['train_ratio']),
        random_state=42,
        stratify=all_labels  # Maintain class balance
    )
    
    # Further split training data into train and validation (80/20 of the training set)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=0.2,  # 20% of training set for validation
        random_state=42,
        stratify=train_labels
    )
    
    print(f"Final dataset sizes:")
    print(f"   Training: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")  
    print(f"   Test: {len(test_paths)} images")
    
    # Print class distribution for each split
    print(f"\nClass distribution:")
    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"  {split_name}:")
        for city, count in zip(unique, counts):
            print(f"    {city}: {count}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SimpleGeoLocationDataset(train_paths, train_labels, train_transform, label_encoder)
    val_dataset = SimpleGeoLocationDataset(val_paths, val_labels, val_transform, label_encoder)
    test_dataset = SimpleGeoLocationDataset(test_paths, test_labels, val_transform, label_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'])
    
    # Create model
    model = PlaNetCNN(num_classes=num_classes, pretrained=True)
    print(f"Model created with {num_classes} output classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['num_epochs'], device=config['device']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, predictions, targets = evaluate_model(model, test_loader, config['device'])
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Print detailed results
    print("\nDetailed Classification Report:")
    print(classification_report(targets, predictions, target_names=label_encoder.classes_))
    
    # Plot results
    plot_results(history, label_encoder.classes_, predictions, targets)
    
    # Save everything
    save_dict = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'test_accuracy': test_acc,
        'history': history,
        'config': config,
        'class_names': label_encoder.classes_.tolist()
    }
    
    torch.save(save_dict, 'planet_geolocation_model.pth')
    print("Model saved as 'planet_geolocation_model.pth'")
    
    print("Training completed successfully!")

# Updated inference function
def predict_image(model_path, image_path, device='cuda'):
    """Predict location for a single image"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    label_encoder = checkpoint['label_encoder']
    num_classes = checkpoint['num_classes']
    
    # Create and load model
    model = PlaNetCNN(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare image
    _, transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_city = label_encoder.inverse_transform([predicted.cpu().item()])[0]
        confidence_score = confidence.cpu().item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
        top3_cities = label_encoder.inverse_transform(top3_indices.cpu().numpy()[0])
        top3_confidences = top3_probs.cpu().numpy()[0]
    
    return predicted_city, confidence_score, list(zip(top3_cities, top3_confidences))

if __name__ == "__main__":
    main()
