import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from monai.networks.nets import DenseNet121  # Example of a 3D CNN from MONAI

class AlvaroLLMNiBabies(nn.Module):
    def __init__(self):
        super(AlvaroLLMNiBabies, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.cnn3d = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)  # 3D CNN
        self.fusion_layer = nn.Linear(768 + 2, 512)  # Combine BERT and CNN outputs
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary classification: CCM or AIH
        )

    def forward(self, text_inputs, image_inputs):
        # Text Processing with BERT
        text_outputs = self.bert(**text_inputs).pooler_output  # [batch_size, 768]

        # Image Processing with 3D CNN
        image_outputs = self.cnn3d(image_inputs)  # [batch_size, 2]

        # Fusion
        fused_features = torch.cat((text_outputs, image_outputs), dim=1)  # [batch_size, 770]
        fused_features = self.fusion_layer(fused_features)  # [batch_size, 512]

        # Classification
        logits = self.classification_head(fused_features)
        return logits
