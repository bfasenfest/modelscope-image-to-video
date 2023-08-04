import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Conditining class, open to ideas for improving it
class ConditionalNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.SiLU()
        )
        self.beta = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.SiLU()
        )

    def forward(self, x, y):
        batch, _, _, height, width = y.shape

        y = rearrange(y, "b c f h w -> (b h w) f c")

        gamma = self.gamma(y)
        beta = self.beta(y)

        gamma = rearrange(gamma, "(b h w) f c -> b c f h w", b=batch, h=height, w=width)
        beta = rearrange(beta, "(b h w) f c -> b c f h w", b=batch, h=height, w=width)

        return gamma * x + beta
    
class ConditioningBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()

        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conditioning_in = nn.Sequential(
            nn.Conv2d(4, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(in_dim // 4, 32), in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
        )

        self.conditioning_mid = ConditionalNorm(in_dim)

        self.conditioning_out = nn.Sequential(
            nn.GroupNorm(min(in_dim // 4, 32), in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, in_dim, (3, 3, 3), padding=(1, 1, 1))
        )

        nn.init.zeros_(self.conditioning_out[-1].weight)
        nn.init.zeros_(self.conditioning_out[-1].bias)

    def forward(self, hidden_states, num_frames=1, init_image=None):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )
        identity = hidden_states

        image_transformed = self.conditioning_in(init_image)
        image_transformed = image_transformed.unsqueeze(2)
        image_transformed = image_transformed.repeat(1, 1, num_frames, 1, 1)
        image_transformed = F.interpolate(image_transformed, size=hidden_states.shape[2:], mode='trilinear')
        
        hidden_states = self.conditioning_mid(image_transformed, hidden_states)
        hidden_states = self.conditioning_out(hidden_states)

        hidden_states = identity + hidden_states
        
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states
    
class TemporalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.0):
        super().__init__()

        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conditioning_in = ConditioningBlock(in_dim, in_dim)

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        self.conditioning_out = ConditioningBlock(in_dim, in_dim)

        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states, num_frames=1, init_image=None):
        hidden_states = self.conditioning_in(hidden_states, num_frames, init_image)

        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )

        hidden_states = self.conditioning_out(hidden_states, num_frames, init_image)
        
        return hidden_states