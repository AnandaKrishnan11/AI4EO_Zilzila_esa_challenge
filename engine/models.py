import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionModule(nn.Module):
    """Attention mechanism for feature fusion"""
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feat1, feat2):
        combined = torch.cat([feat1, feat2], dim=1)
        attention_weights = self.attention(combined)
        attended_feat1 = feat1 * attention_weights[:, 0].unsqueeze(1)
        attended_feat2 = feat2 * attention_weights[:, 1].unsqueeze(1)
        return attended_feat1 + attended_feat2, attention_weights

class ResNetChangeDetector(nn.Module):
    """ResNet-based model for bitemporal scene classification and change detection"""
    def __init__(self, num_classes, depth, pretrained=False, feature_dim=512):
        super(ResNetChangeDetector, self).__init__()
        self.feature_dim = feature_dim
        
        # Feature extractor
        if depth == 50:
            self.feature_extractor = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            self.feature_extractor = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f'The provided model depth {depth} is not known')
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(2048, feature_dim)
        
    
        self.attention = AttentionModule(feature_dim)   # Attention module for feature fusion
        
       
        self.change_classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 256),  # Concatenated features + difference + attention
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    

    def extract_features(self, x):
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features
    
    def forward(self, x1, x2):

        features_t1 = self.extract_features(x1)
        features_t2 = self.extract_features(x2)
        
        # Fuse features using attention
        fused_features, attention_weights = self.attention(features_t1, features_t2)
        
        # Calculate feature difference for change detection
        feature_diff = torch.abs(features_t1 - features_t2)
        
        # Change detection or combined classification
        change_features = torch.cat([fused_features, feature_diff, features_t1, features_t2], dim=1)
        change_pred = self.change_classifier(change_features)
        return change_pred



class ResNetTransfermerModel(nn.Module):
    def __init__(self, num_classes, depth, pretrained=False):
        super(ResNetTransfermerModel, self).__init__()
       
        # Shared ResNet backbone
        if depth == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f'The provided model depth {depth} is not known!')
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
       
        # Feature dimension
        feature_dim = 2048
       
        # Transformer for temporal fusion
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )
       
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
   
    def forward(self, x1, x2):
        # Extract features
        feat1 = self.feature_extractor(x1).squeeze()
        feat2 = self.feature_extractor(x2).squeeze()
       
        # Stack temporal features
        temporal_features = torch.stack([feat1, feat2], dim=1)
       
        # Apply transformer
        fused_features = self.transformer(temporal_features)
       
        # Use features from both time steps
        final_features = fused_features.mean(dim=1)
       
        return self.classifier(final_features)
    


# Example usage and training setup
def create_model(num_classes, model_type, depth, pretrained):
    if model_type == 'resnet':
        return ResNetChangeDetector(
                    num_classes=num_classes,
                    depth=depth,
                    pretrained=pretrained,
                    feature_dim=512
                )
    else:
        return ResNetTransfermerModel(num_classes=num_classes,
                                      depth=depth,
                                      pretrained=pretrained)


if __name__ == '__main__':
    x1 = torch.rand(4,3, 128, 128)
    x2 = torch.rand(4,3, 128, 128)

    model = create_model(num_classes=2)
    out = model(x1, x2)
    print(out.shape)
