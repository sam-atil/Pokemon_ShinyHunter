import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------------------------------------
# MODEL 1: (Pokemon Classifier)
# ------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, num_class = 30):
        super().__init__()
        #Main part of CNN Layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        #Classifier section of CNN
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            #From Task 5
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
            #Don't need softmax here, because crossEntropy() function already includes it
        )

    #Forward pass function
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------
# MODEL 2A: CNN Feature Extractor (same as ShinyCNN)
# ------------------------------------------------------------

class ShinyCNN(nn.Module):
    def __init__(self, feature_dim = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )


        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

# ------------------------------------------------------------
# MODEL 2B: Shiny Modal Classifier (ShinyModal)
# ------------------------------------------------------------

class ShinyModal(nn.Module):
    def __init__(self, num_pokemon = 30, id_emb_dim = 32, img_feat_dim = 256, num_classes =2):
        super().__init__()

        self.cnn = ShinyCNN(feature_dim=img_feat_dim)
        self.id_proj = nn.Embedding(num_pokemon, id_emb_dim)

        fusion_dim = img_feat_dim + id_emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_data, id_data):
        img_features = self.cnn(img_data)
        id_features = self.id_proj(id_data)
        combined = torch.cat([img_features, id_features], dim = 1)
        return self.fusion(combined)