import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import seaborn as sns
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def haversine_distance_torch(pred_latlon, target_latlon):
    """
    Torch version of Haversine distance for loss calculation
    Input: tensors of shape (batch_size, 2) where columns are [lat, lon]
    Returns: mean Haversine distance in kilometers
    """
    # Convert to radians
    pred_lat_rad = torch.deg2rad(pred_latlon[:, 0])
    pred_lon_rad = torch.deg2rad(pred_latlon[:, 1])
    target_lat_rad = torch.deg2rad(target_latlon[:, 0])
    target_lon_rad = torch.deg2rad(target_latlon[:, 1])
    
    # Haversine formula
    dlat = target_lat_rad - pred_lat_rad
    dlon = target_lon_rad - pred_lon_rad
    
    a = torch.sin(dlat/2)**2 + torch.cos(pred_lat_rad) * torch.cos(target_lat_rad) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))  # Clamp to avoid numerical issues
    
    # Radius of earth in kilometers
    r = 6371
    
    return torch.mean(c * r)

def load_trainval_data_with_coordinates(train_val_dir):
    """Load data from train_val directory with lat/lon coordinates from raw.csv files"""
    
    all_image_paths = []
    all_coordinates = []
    
    logger.info(f"Loading data from train_val: {train_val_dir}")
    
    for city_folder in sorted(os.listdir(train_val_dir)):
        city_path = os.path.join(train_val_dir, city_folder)
        if not os.path.isdir(city_path):
            continue
            
        images_dir = os.path.join(city_path, 'query', 'images')
        csv_path = os.path.join(city_path, 'query', 'raw.csv')  # Changed to raw.csv
        
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            continue
            
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            continue
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  {city_folder}: CSV loaded with {len(df)} entries")
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            continue
        
        # Check required columns
        required_cols = ['key', 'lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns in {csv_path}. Found: {df.columns.tolist()}")
            continue
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        available_images = {f.split('.')[0]: f for f in os.listdir(images_dir) 
                          if any(f.endswith(ext) for ext in image_extensions)}
        
        # Match CSV entries with available images
        matched_count = 0
        for _, row in df.iterrows():
            key = str(row['key'])
            if key in available_images:
                image_path = os.path.join(images_dir, available_images[key])
                all_image_paths.append(image_path)
                # Store as [lat, lon] and validate ranges
                lat, lon = float(row['lat']), float(row['lon'])
                if -90 <= lat <= 90 and -180 <= lon <= 180:  # Validate coordinate ranges
                    all_coordinates.append([lat, lon])
                    matched_count += 1
                else:
                    logger.warning(f"Invalid coordinates: lat={lat}, lon={lon} for key={key}")
        
        logger.info(f"  {city_folder}: {matched_count}/{len(df)} entries matched with images")
    
    all_coordinates = np.array(all_coordinates)
    logger.info(f"Total loaded: {len(all_image_paths)} images with coordinates")
    logger.info(f"Coordinate ranges - Latitude: [{all_coordinates[:, 0].min():.6f}, {all_coordinates[:, 0].max():.6f}]")
    logger.info(f"                  - Longitude: [{all_coordinates[:, 1].min():.6f}, {all_coordinates[:, 1].max():.6f}]")
    
    # Print some sample coordinates for debugging
    logger.info(f"Sample coordinates:")
    for i in range(min(5, len(all_coordinates))):
        logger.info(f"  [{all_coordinates[i, 0]:.6f}, {all_coordinates[i, 1]:.6f}]")
    
    return all_image_paths, all_coordinates

class GeoLocationRegressionDataset(Dataset):
    """Dataset for lat/lon coordinate regression"""
    
    def __init__(self, image_paths, coordinates, transform=None, coord_scaler=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform
        self.coord_scaler = coord_scaler
        
        # Normalize coordinates if scaler is provided
        if self.coord_scaler is not None:
            self.normalized_coordinates = self.coord_scaler.transform(self.coordinates)
        else:
            self.normalized_coordinates = self.coordinates
    
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
        
        coordinates = torch.tensor(self.normalized_coordinates[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, coordinates

class PlaNetRegression(nn.Module):
    """PlaNet-style CNN for lat/lon coordinate regression"""
    
    def __init__(self, pretrained=True):
        super(PlaNetRegression, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer for regression (2 outputs: lat, lon)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # 2 outputs for lat, lon - removed Tanh
        )
    
    def forward(self, x):
        return self.backbone(x)

def calculate_distance_error_km(pred_coords, true_coords, coord_scaler):
    """Calculate Haversine distance error in kilometers"""
    # Denormalize coordinates
    pred_original = coord_scaler.inverse_transform(pred_coords)
    true_original = coord_scaler.inverse_transform(true_coords)
    
    # Calculate Haversine distances
    distances = []
    for i in range(len(pred_original)):
        pred_lat, pred_lon = pred_original[i]
        true_lat, true_lon = true_original[i]
        dist = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
        distances.append(dist)
    
    return np.array(distances)

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, coord_scaler,
                num_epochs=25, device='cuda', save_path='best_model.pth'):
    """Train the regression model"""
    
    model.to(device)
    
    # Tracking metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_distance_error': [], 'val_distance_error': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(target.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f} km'})
        
        train_loss = running_loss / len(train_loader)
        
        # Calculate training distance error
        train_predictions = np.vstack(train_predictions)
        train_targets = np.vstack(train_targets)
        train_distances = calculate_distance_error_km(train_predictions, train_targets, coord_scaler)
        train_distance_error = np.mean(train_distances)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(target.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f} km'})
        
        val_loss /= len(val_loader)
        
        # Calculate validation distance error
        val_predictions = np.vstack(val_predictions)
        val_targets = np.vstack(val_targets)
        val_distances = calculate_distance_error_km(val_predictions, val_targets, coord_scaler)
        val_distance_error = np.mean(val_distances)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_distance_error'].append(train_distance_error)
        history['val_distance_error'].append(val_distance_error)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} km, Train Distance Error: {train_distance_error:.2f} km")
        print(f"Val Loss: {val_loss:.4f} km, Val Distance Error: {val_distance_error:.2f} km")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Val Loss: {best_val_loss:.4f} km")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    logger.info(f"Best validation loss: {best_val_loss:.4f} km")
    
    return model, history

def evaluate_model(model, test_loader, coord_scaler, device='cuda'):
    """Evaluate regression model on test set"""
    
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for data, target in test_pbar:
            data = data.to(device)
            outputs = model(data)
            
            predictions.append(outputs.cpu().numpy())
            targets.append(target.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Calculate distance errors in kilometers
    distances = calculate_distance_error_km(predictions, targets, coord_scaler)
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    
    return mean_error, median_error, predictions, targets, distances

def plot_results(history, predictions, targets, distances, coord_scaler):
    """Plot training history and results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss (Haversine Distance)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (km)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_distance_error'], 'b-', label='Training Distance Error')
    ax2.plot(epochs, history['val_distance_error'], 'r-', label='Validation Distance Error')
    ax2.set_title('Training and Validation Distance Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Distance Error (km)')
    ax2.legend()
    ax2.grid(True)
    
    # Geographic scatter plot
    pred_original = coord_scaler.inverse_transform(predictions)
    targets_original = coord_scaler.inverse_transform(targets)
    
    # Plot actual vs predicted locations on map
    ax3.scatter(targets_original[:, 1], targets_original[:, 0], alpha=0.6, s=10, c='blue', label='Actual')
    ax3.scatter(pred_original[:, 1], pred_original[:, 0], alpha=0.6, s=10, c='red', label='Predicted')
    ax3.set_title('Geographic Distribution: Actual vs Predicted')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.legend()
    ax3.grid(True)
    
    # Distance error histogram
    ax4.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.2f} km')
    ax4.axvline(np.median(distances), color='orange', linestyle='--', 
                label=f'Median: {np.median(distances):.2f} km')
    ax4.set_title('Distance Error Distribution')
    ax4.set_xlabel('Distance Error (km)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training pipeline"""
    
    config = {
        'train_val_dir': 'C:\\Users\\giopa\\Downloads\\Compressed\\train_val', 
        'train_ratio': 0.8,  
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 2
    }
    
    print("Starting PlaNet Geolocation Regression Training (Lat/Lon with Haversine)")
    print(f"Using device: {config['device']}")
    
    # Load data from train_val directory with coordinates
    print("\nLoading data with lat/lon coordinates from raw.csv files...")
    all_image_paths, all_coordinates = load_trainval_data_with_coordinates(config['train_val_dir'])
    
    if len(all_image_paths) == 0:
        print(" No images found!")
        return
    
    # Create coordinate scaler for normalization
    coord_scaler = StandardScaler()
    coord_scaler.fit(all_coordinates)
    
    print(f"Loaded {len(all_image_paths)} images with coordinates")
    
    # Split data into train and test (80/20)
    print(f"\nSplitting data: {config['train_ratio']*100:.0f}% train, {(1-config['train_ratio'])*100:.0f}% test")
    
    train_paths, test_paths, train_coords, test_coords = train_test_split(
        all_image_paths, all_coordinates,
        test_size=(1 - config['train_ratio']),
        random_state=42
    )
    

    train_paths, val_paths, train_coords, val_coords = train_test_split(
        train_paths, train_coords,
        test_size=0.2,  
        random_state=42
    )
    
    print(f" Final dataset sizes:")
    print(f"   Training: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")  
    print(f"   Test: {len(test_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = GeoLocationRegressionDataset(train_paths, train_coords, train_transform, coord_scaler)
    val_dataset = GeoLocationRegressionDataset(val_paths, val_coords, val_transform, coord_scaler)
    test_dataset = GeoLocationRegressionDataset(test_paths, test_coords, val_transform, coord_scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'])
    
    # Create model
    model = PlaNetRegression(pretrained=True)
    print(f"Model created for lat/lon coordinate regression")
    
    # Loss and optimizer - using Haversine distance loss
    criterion = haversine_distance_torch
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Train model
    print("\nStarting training with Haversine distance loss...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, coord_scaler,
        num_epochs=config['num_epochs'], device=config['device']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    mean_error, median_error, predictions, targets, distances = evaluate_model(
        model, test_loader, coord_scaler, config['device']
    )
    
    print(f"Final Test Results:")
    print(f"   Mean Distance Error: {mean_error:.2f} km")
    print(f"   Median Distance Error: {median_error:.2f} km")
    print(f"   25th percentile: {np.percentile(distances, 25):.2f} km")
    print(f"   75th percentile: {np.percentile(distances, 75):.2f} km")
    print(f"   Max error: {np.max(distances):.2f} km")
    print(f"   Min error: {np.min(distances):.2f} km")
    print(f"   % predictions within 100km: {np.sum(distances < 100) / len(distances) * 100:.1f}%")
    print(f"   % predictions within 1000km: {np.sum(distances < 1000) / len(distances) * 100:.1f}%")
    
    # Plot results
    plot_results(history, predictions, targets, distances, coord_scaler)
    
    # Save everything
    save_dict = {
        'model_state_dict': model.state_dict(),
        'coord_scaler': coord_scaler,
        'mean_error': mean_error,
        'median_error': median_error,
        'history': history,
        'config': config
    }
    
    torch.save(save_dict, 'planet_regression_model.pth')
    print("Model saved as 'planet_regression_model.pth'")
    
    print("Training completed successfully!")

def predict_coordinates(model_path, image_path, device='cuda'):
    """Predict lat/lon coordinates for a single image"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    coord_scaler = checkpoint['coord_scaler']
    
    # Create and load model
    model = PlaNetRegression(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare image
    _, transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        normalized_coords = model(image_tensor).cpu().numpy()
        
        # Denormalize coordinates
        predicted_coords = coord_scaler.inverse_transform(normalized_coords)[0]
        lat, lon = predicted_coords[0], predicted_coords[1]
    
    return lat, lon

if __name__ == "__main__":
    main()

# Example usage:
# lat, lon = predict_coordinates('planet_regression_model.pth', 'test_image.jpg')
# print(f"Predicted coordinates: Lat={lat:.6f}, Lon={lon:.6f}")
# print(f"Google Maps link: https://www.google.com/maps?q={lat},{lon}")