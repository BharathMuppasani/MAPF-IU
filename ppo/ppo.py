import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic 3×3 residual block (He et al. 2016)."""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride,
                               padding, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1,
                               padding, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class PPOActorCritic(nn.Module):
    """
    ResNet‐based Spatial Encoder + per‐step GRU actor/critic heads,
    *grid‐size agnostic*.
    Forward: logits, value, new_hx_actor, new_hx_critic = forward(obs, hx_a, hx_c)
    """
    def __init__(
        self,
        num_actions: int,
        fc_hidden: int = 256,
        low_feature_dim: int = 16
    ):
        super().__init__()
        self.fc_hidden     = fc_hidden

        # 1) conv trunk
        self.conv1 = nn.Conv2d(6, 32, 3, 1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.res1  = ResidualBlock(32, 32)
        self.res2  = ResidualBlock(32, 32, dilation=2)

        # 2) fuse low‐dim (dir+dist)
        self.low_to_conv = nn.Linear(3, low_feature_dim, bias=False)
        self.conv_fuse   = nn.Conv2d(32 + low_feature_dim, 32, 3, 1, padding=1, bias=False)
        self.bn_fuse     = nn.BatchNorm2d(32)

        # 3) downsample + more Res blocks
        self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.res3  = ResidualBlock(64, 64, dilation=2)
        self.res4  = ResidualBlock(64, 64, dilation=4)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.res5  = ResidualBlock(128, 128, dilation=4)
        self.res6  = ResidualBlock(128, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # 4) late‐fusion MLP
        self.fc_low     = nn.Linear(3, fc_hidden, bias=False)
        self.fc_conv    = nn.Linear(128, fc_hidden, bias=False)
        self.fc_combine = nn.Linear(fc_hidden * 2, fc_hidden, bias=False)

        # 5) GRUs + heads
        self.actor_gru  = nn.GRUCell(fc_hidden, fc_hidden)
        self.critic_gru = nn.GRUCell(fc_hidden, fc_hidden)
        self.action_head = nn.Linear(fc_hidden, num_actions)
        self.value_head  = nn.Linear(fc_hidden, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, hx_actor, hx_critic):
        # obs['grid']: (B,1,H,W) or (B,H,W)
        grid = obs['grid']
        if grid.ndim==4 and grid.size(1)==1:
            grid = grid.squeeze(1)           # → (B,H,W)
        B, H, W = grid.shape

        # a) build spatial channels + dynamic coords
        obstacles = ((grid==-1)|(grid==-3)).float()
        agent     = ((grid==1)|(grid==3)).float()
        goal      = ((grid==2)|(grid==3)).float()
        dynamic   = (grid==-2).float()

        # on‐the‐fly normalized coordinate channels
        ys = torch.linspace(-1,1,H, device=grid.device)
        xs = torch.linspace(-1,1,W, device=grid.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')         # (H,W)
        coords = torch.stack([gx, gy], dim=0).unsqueeze(0)      # (1,2,H,W)
        coords = coords.expand(B, -1, -1, -1)                   # (B,2,H,W)

        x = torch.stack([obstacles, agent, goal, dynamic], dim=1)  # (B,4,H,W)
        x = torch.cat([x, coords], dim=1)                         # (B,6,H,W)

        # b) ResNet trunk
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x);  x = self.res2(x)

        # c) fuse low‐dim
        direction = obs['direction']   # (B,2)
        distance  = obs['distance']    # (B,1)
        low = torch.cat([
            direction.view(B,2), distance.view(B,1)
        ], dim=1)                     # (B,3)
        lp = F.relu(self.low_to_conv(low)).view(B,-1,1,1).expand(-1,-1,H,W)
        x  = torch.cat([x, lp], dim=1)
        x  = F.relu(self.bn_fuse(self.conv_fuse(x)))

        # d) deeper ResNet
        x = F.relu(self.bn2(self.conv2(x))); x = self.res3(x); x = self.res4(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.res5(x); x = self.res6(x)

        # e) global pool + MLP
        x = self.global_pool(x).view(B, 128)
        low2      = F.relu(self.fc_low(low))      # (B,fc_hidden)
        conv_feat = F.relu(self.fc_conv(x))       # (B,fc_hidden)
        comb      = torch.cat([conv_feat, low2], dim=1)
        comb      = F.relu(self.fc_combine(comb)) # (B,fc_hidden)

        # f) GRUs
        new_hx_actor  = self.actor_gru(comb, hx_actor)
        new_hx_critic = self.critic_gru(comb, hx_critic)

        # g) heads
        logits = self.action_head(new_hx_actor)                # (B,A)
        value  = self.value_head(new_hx_critic).squeeze(-1)    # (B,)

        return logits, value, new_hx_actor, new_hx_critic
