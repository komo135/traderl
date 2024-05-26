from torch import nn


class CnnNet(nn.Module):
    def __init__(self, in_channels, out_channels=32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnnnet = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=8, stride=2, padding=4),
            nn.GroupNorm(min(self.out_channels // 2, 32), self.out_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.out_channels, self.out_channels * 2, kernel_size=4, stride=2, padding=2),
            nn.GroupNorm(min((self.out_channels * 2) // 2, 32), self.out_channels * 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.cnnnet(x)


# モジュールのファクトリ関数
def create_cnnnet(in_channels, out_channels=32, **kwargs):
    assert isinstance(in_channels, int), "in_channels must be int"
    assert isinstance(out_channels, int), "out_channels must be int"
    return CnnNet(in_channels, out_channels, **kwargs)


cnnnet_models = {
    "cnnnet": lambda in_channels, **kwargs: create_cnnnet(in_channels, 32, **kwargs),
    "cnnnet_64": lambda in_channels, **kwargs: create_cnnnet(in_channels, 64, **kwargs),
    "cnnnet_128": lambda in_channels, **kwargs: create_cnnnet(in_channels, 128, **kwargs),
}

__all__ = ["cnnnet_models"]
