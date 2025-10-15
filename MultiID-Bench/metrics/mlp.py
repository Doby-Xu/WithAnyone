import torch
import torch.nn as nn
  
class Conv_MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        result = self.conv_head(image_embeds)
        return self.scoring_head(result)