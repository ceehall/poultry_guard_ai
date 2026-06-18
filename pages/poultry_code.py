import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

class PoultryDiseaseClassifier(nn.Module):
    """
    Custom MobileNetV3 Large architecture used for classifying 
    poultry fecal diseases across 4 classes.
    """
    def __init__(self, num_classes=4):
        super(PoultryDiseaseClassifier, self).__init__()
        try:
            self.backbone = models.mobilenet_v3_large(weights=None)
        except TypeError:
            self.backbone = models.mobilenet_v3_large(pretrained=False)
            
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def letterbox_resize(pil_img, target_size=(224, 224)):
    """Resizes an image maintaining its original aspect ratio using padding."""
    pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', target_size, (0, 0, 0))
    background.paste(pil_img, ((target_size[0] - pil_img.size[0]) // 2,
                               (target_size[1] - pil_img.size[1]) // 2))
    return background

def verify_structural_suitability(pil_image):
    """
    STAGE 2 GUARDRAIL (Structural):
    Analyzes contour geometry to ensure the image contains a distinct, 
    isolated object on a floor rather than a completely flat wall, empty space, 
    or blurry out-of-focus background.
    """
    img = np.array(pil_image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = img.shape[0] * img.shape[1]
    
    area_ratio = contour_area / total_area
    
    # If an object contour takes up more than 85% of the frame or less than 1%,
    # it is likely a flat surface background anomaly or camera noise.
    if area_ratio > 0.85 or area_ratio < 0.01:
        return False
    return True

def isolate_region_of_interest_from_pil(pil_image, output_size=(224, 224)):
    """Crops closely around the core fecal dropping matter to remove background noise."""
    img = np.array(pil_image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pad = 10
        crop = img[max(0, y-pad):min(img.shape[0], y+h+pad), 
                   max(0, x-pad):min(img.shape[1], x+w+pad)]
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return letterbox_resize(Image.fromarray(crop_rgb), output_size)
            
    return letterbox_resize(pil_image, output_size)

def calculate_flock_risk_score(cv_probability, disease_class, bird_age_weeks, humidity_pct, flock_density_high):
    """Combines model classification outputs with coop metadata for an operational risk score."""
    if disease_class.lower() == "healthy":
        return float(cv_probability * 0.1)
        
    multiplier = 0.0
    if disease_class.lower() == "cocci" and humidity_pct > 75:
        multiplier += 0.15
    if disease_class.lower() == "ncd" and flock_density_high:
        multiplier += 0.15
    if bird_age_weeks < 3:
        multiplier += 0.10
        
    final_score = (cv_probability * 0.65) + (multiplier * 0.35)
    return min(float(final_score), 1.0)\
    