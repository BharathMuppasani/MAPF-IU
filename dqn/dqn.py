import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=1):
#         super().__init__()
#         padding = dilation
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, stride=stride,
#                                padding=padding, dilation=dilation, bias=False)
#         self.bn1   = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, stride=1,     
#                                padding=padding, dilation=dilation, bias=False)
#         self.bn2   = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         return F.relu(out)

# class ResNetDQN(nn.Module):
#     def __init__(self, grid_size, num_actions=5, fc_hidden=256, low_feature_dim=16):
#         super().__init__()
#         self.grid_size = grid_size

#         # 1) Precompute normalized coordinate grid as a buffer
#         xs = torch.linspace(-1, 1, grid_size)
#         ys = torch.linspace(-1, 1, grid_size)
#         grid_y, grid_x = torch.meshgrid(xs, ys, indexing='ij')  # (H, W)
#         coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
#         self.register_buffer('coords', coords)

#         # 2) Initial conv: 6 channels = [obs, agent, goal, dyn, x_coord, y_coord]
#         self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.res1  = ResidualBlock(32, 32, stride=1, dilation=1)
#         self.res2  = ResidualBlock(32, 32, stride=1, dilation=2)

#         # 3) Early fusion of low-dim features
#         self.low_to_conv = nn.Linear(3, low_feature_dim, bias=False)
#         self.conv_fuse   = nn.Conv2d(32 + low_feature_dim, 32,
#                                      kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn_fuse     = nn.BatchNorm2d(32)

#         # 4) Downsample #1
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 1/2 resolution
#         self.bn2   = nn.BatchNorm2d(64)
#         self.res3  = ResidualBlock(64, 64, stride=1, dilation=2)
#         self.res4  = ResidualBlock(64, 64, stride=1, dilation=4)

#         # 5) Downsample #2
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)  # 1/4 resolution
#         self.bn3   = nn.BatchNorm2d(128)
#         self.res5  = ResidualBlock(128, 128, stride=1, dilation=4)
#         self.res6  = ResidualBlock(128, 128, stride=1, dilation=1)

#         # 6) Global pooling
#         self.global_pool = nn.AdaptiveAvgPool2d((1,1))

#         # 7) Late fusion heads
#         self.fc_low   = nn.Linear(3, fc_hidden, bias=False)
#         self.fc_conv  = nn.Linear(128, fc_hidden, bias=False)
#         self.fc_combine = nn.Linear(fc_hidden*2, fc_hidden, bias=False)
#         self.out       = nn.Linear(fc_hidden, num_actions, bias=True)

#         # 8) Weight initialization
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if getattr(m, 'bias', None) is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, obs):
#         # --- Build multi-channel grid input ---
#         grid = obs['grid']                           # (B, H, W)
#         B, H, W = grid.shape

#         obstacles = ((grid == -1) | (grid == -3)).float()
#         agent     = ((grid == 1) | (grid == 3)).float()
#         goal      = ((grid == 2) | (grid == 3)).float()
#         dynamic   = ((grid == -2) | (grid == -3)).float()

#         # coords: (1,2,H,W) → (B,2,H,W)
#         coords = self.coords.expand(B, -1, -1, -1)

#         x = torch.stack([obstacles, agent, goal, dynamic], dim=1)  # (B,4,H,W)
#         x = torch.cat([x, coords], dim=1)                         # (B,6,H,W)

#         # --- Conv / ResNet stem ---
#         x = F.relu(self.bn1(self.conv1(x)))  # (B,32,H,W)
#         x = self.res1(x)
#         x = self.res2(x)

#         # --- Early fuse low-dim features ---
#         low = torch.cat([obs['direction'], obs['distance']], dim=1)  # (B,3)
#         lp  = F.relu(self.low_to_conv(low))                         # (B, low_feature_dim)
#         lp  = lp.view(B, -1, 1, 1).expand(-1, -1, H, W)             # (B,low,H,W)
#         x   = torch.cat([x, lp], dim=1)                             # (B,32+low,H,W)
#         x   = F.relu(self.bn_fuse(self.conv_fuse(x)))               # (B,32,H,W)

#         # --- Downsample #1 + Res blocks ---
#         x = F.relu(self.bn2(self.conv2(x)))  # (B,64,H/2,W/2)
#         x = self.res3(x)
#         x = self.res4(x)

#         # --- Downsample #2 + Res blocks ---
#         x = F.relu(self.bn3(self.conv3(x)))  # (B,128,H/4,W/4)
#         x = self.res5(x)
#         x = self.res6(x)

#         # --- Global pooling + flatten ---
#         x = self.global_pool(x).view(B, 128)  # (B,128)

#         # --- Late fusion + output ---
#         low2 = F.relu(self.fc_low(low))        # (B,fc_hidden)
#         c2   = F.relu(self.fc_conv(x))         # (B,fc_hidden)
#         comb = torch.cat([c2, low2], dim=1)    # (B, 2*fc_hidden)
#         comb = F.relu(self.fc_combine(comb))   # (B,fc_hidden)
#         q    = self.out(comb)                  # (B, num_actions)

#         return q

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,     
                               padding=padding, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetDQN(nn.Module):
    def __init__(self, num_actions=5, fc_hidden=256, low_feature_dim=16):
        super().__init__()
        # 1) Initial conv stem (no coords yet)
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.res1  = ResidualBlock(32, 32, stride=1, dilation=1)
        self.res2  = ResidualBlock(32, 32, stride=1, dilation=2)

        # 2) Early fusion of low-dim features
        self.low_to_conv = nn.Linear(3, low_feature_dim, bias=False)
        self.conv_fuse   = nn.Conv2d(32 + low_feature_dim, 32,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_fuse     = nn.BatchNorm2d(32)

        # 3) Downsample #1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.res3  = ResidualBlock(64, 64, stride=1, dilation=2)
        self.res4  = ResidualBlock(64, 64, stride=1, dilation=4)

        # 4) Downsample #2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.res5  = ResidualBlock(128, 128, stride=1, dilation=4)
        self.res6  = ResidualBlock(128, 128, stride=1, dilation=1)

        # 5) Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # 6) Late fusion heads
        self.fc_low      = nn.Linear(3, fc_hidden, bias=False)
        self.fc_conv     = nn.Linear(128, fc_hidden, bias=False)
        self.fc_combine  = nn.Linear(fc_hidden * 2, fc_hidden, bias=False)
        self.out         = nn.Linear(fc_hidden, num_actions, bias=True)

        # 7) Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        # --- Build multi-channel grid input ---
        grid = obs['grid']           # (B, H, W)
        B, H, W = grid.shape

        # derive channels
        obstacles = ((grid == -1) | (grid == -3)).float()
        agent     = ((grid == 1) | (grid == 3)).float()
        goal      = ((grid == 2) | (grid == 3)).float()
        dynamic   = ((grid == -2) | (grid == -3)).float()

        # dynamically generate coords in [−1,1]
        device = grid.device
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        coords = torch.stack([grid_x, grid_y], dim=0)           # (2, H, W)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)       # (B, 2, H, W)

        # stack everything
        x = torch.stack([obstacles, agent, goal, dynamic], dim=1)  # (B, 4, H, W)
        x = torch.cat([x, coords], dim=1)                         # (B, 6, H, W)

        # --- Conv / ResNet stem ---
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, H, W)
        x = self.res1(x)
        x = self.res2(x)

        # --- Early fuse low-dim features ---
        low = torch.cat([obs['direction'], obs['distance']], dim=1)  # (B, 3)
        lp  = F.relu(self.low_to_conv(low))                          # (B, low_feature_dim)
        lp  = lp.view(B, -1, 1, 1).expand(-1, -1, H, W)              # (B, low, H, W)
        x   = torch.cat([x, lp], dim=1)                              # (B, 32+low, H, W)
        x   = F.relu(self.bn_fuse(self.conv_fuse(x)))                # (B, 32, H, W)

        # --- Downsample #1 + Res blocks ---
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, H/2, W/2)
        x = self.res3(x)
        x = self.res4(x)

        # --- Downsample #2 + Res blocks ---
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, H/4, W/4)
        x = self.res5(x)
        x = self.res6(x)

        # --- Global pooling + flatten ---
        feat = self.global_pool(x).view(B, 128)  # (B, 128)

        # --- Late fusion + output ---
        low2 = F.relu(self.fc_low(low))          # (B, fc_hidden)
        c2   = F.relu(self.fc_conv(feat))        # (B, fc_hidden)
        comb = torch.cat([c2, low2], dim=1)      # (B, 2*fc_hidden)
        comb = F.relu(self.fc_combine(comb))     # (B, fc_hidden)
        q    = self.out(comb)                    # (B, num_actions)

        return q

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, stride=stride, padding=1)
#         self.bn1   = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, stride=1, padding=1)
#         self.bn2   = nn.BatchNorm2d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         return F.relu(out)

# class ResNetDQN(nn.Module):
#     def __init__(self, grid_size, num_actions=4, fc_hidden=128, low_feature_dim=8):
#         """
#         Args:
#             grid_size: Height/width of the grid input.
#             num_actions: Number of discrete actions.
#             fc_hidden: Hidden dimension for fully connected layers.
#             low_feature_dim: Dimension to which low-dimensional features (direction + distance) are projected.
        
#         This network now expects a multi-channel input with 5 channels.
#         The grid is converted into 5 channels as follows:
#           - Channel 0: Free cells (grid == 0)
#           - Channel 1: Obstacles (grid == -1)
#           - Channel 2: Agent (grid == 1 or grid == 3)
#           - Channel 3: Goal (grid == 2 or grid == 3)
#           - Channel 4: Dynamic obstacles (currently always 0)
#         """
#         super(ResNetDQN, self).__init__()
#         # --- Convolutional branch (early stage) ---
#         # Update first conv to accept 5 channels.
#         self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.resblock1 = ResidualBlock(32, 32)
#         self.resblock2 = ResidualBlock(32, 32)
        
#         # --- Early fusion of low-dimensional features ---
#         self.low_to_conv = nn.Linear(3, low_feature_dim)
#         self.conv_fuse = nn.Conv2d(32 + low_feature_dim, 32, kernel_size=3, stride=1, padding=1)
#         self.bn_fuse   = nn.BatchNorm2d(32)
        
#         # --- Further convolutional branch ---
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn2   = nn.BatchNorm2d(64)
#         self.resblock3 = ResidualBlock(64, 64)
#         self.resblock4 = ResidualBlock(64, 64)
        
#         # --- Adaptive pooling to 2x2 ---
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # Output: (batch, 64, 2, 2)
        
#         # --- Late fusion of low-dimensional features ---
#         # Flattened conv output size: 64 * 2 * 2 = 256.
#         self.fc_conv = nn.Linear(64 * 4, fc_hidden)
#         self.fc_low = nn.Linear(3, fc_hidden)
        
#         # Optional dropout for regularization.
#         self.dropout = nn.Dropout(p=0.2)
        
#         # Combined fully connected layer.
#         self.fc_combined = nn.Linear(fc_hidden + fc_hidden, fc_hidden)
#         self.out = nn.Linear(fc_hidden, num_actions)
        
#         # Weight initialization.
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
                    
#     def forward(self, obs):
#         """
#         Expects obs as a dict with:
#           - obs['grid']: tensor of shape (batch, grid_size, grid_size) containing integer values:
#               0 for free, -1 for obstacles, 1 for agent, 2 for goal.
#           - obs['direction']: tensor of shape (batch, 2)
#           - obs['distance']: tensor of shape (batch, 1)
        
#         Converts the grid into a 5-channel representation:
#           - Channel 0: Free cells (grid == 0)
#           - Channel 1: Obstacles (grid == -1)
#           - Channel 2: Agent (grid == 1 or grid == 3)
#           - Channel 3: Goal (grid == 2 or grid == 3)
#           - Channel 4: Dynamic obstacles (always 0 for now)
#         """
#         batch_size = obs['grid'].size(0)
#         grid = obs['grid']  # (batch, grid_size, grid_size)
        
#         # Create multi-channel input.
#         free = (grid == 0).float()
#         obstacles = (grid == -1).float()
#         agent = ((grid == 1) | (grid == 3)).float()
#         goal = ((grid == 2) | (grid == 3)).float()
#         dynamic = torch.zeros_like(grid).float()  # Currently always 0.
#         # Stack channels -> (batch, 5, grid_size, grid_size)
#         x = torch.stack([free, obstacles, agent, goal, dynamic], dim=1)
        
#         # --- Convolutional branch ---
#         x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, grid_size, grid_size)
#         x = self.resblock1(x)
#         x = self.resblock2(x)
        
#         # --- Early fusion ---
#         low_extra = torch.cat([obs['direction'], obs['distance']], dim=1)  # (batch, 3)
#         low_proj = F.relu(self.low_to_conv(low_extra))  # (batch, low_feature_dim)
#         low_proj_expanded = low_proj.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
#         x = torch.cat([x, low_proj_expanded], dim=1)  # (batch, 32+low_feature_dim, grid_size, grid_size)
#         x = F.relu(self.bn_fuse(self.conv_fuse(x)))  # (batch, 32, grid_size, grid_size)
        
#         # --- Further convolutional branch ---
#         x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, grid_size/2, grid_size/2)
#         x = self.resblock3(x)
#         x = self.resblock4(x)
        
#         # --- Adaptive Pooling with MPS Workaround ---
#         # MPS requires input sizes to be divisible by output sizes.
#         # If not, temporarily move the tensor to CPU for pooling.
#         x = x.to("cpu")
#         x = self.adaptive_pool(x)  # (batch, 64, 2, 2)
#         x = x.to(obs['grid'].device)
#         x = x.view(batch_size, -1)  # (batch, 64*2*2 = 256)
        
#         # --- Late fusion ---
#         conv_features = F.relu(self.fc_conv(x))  # (batch, fc_hidden)
#         conv_features = self.dropout(conv_features)
#         low_late = F.relu(self.fc_low(low_extra))  # (batch, fc_hidden)
#         low_late = self.dropout(low_late)
#         combined = torch.cat([conv_features, low_late], dim=1)  # (batch, 2*fc_hidden)
#         combined = F.relu(self.fc_combined(combined))  # (batch, fc_hidden)
#         combined = self.dropout(combined)
#         q_values = self.out(combined)  # (batch, num_actions)
#         return q_values



# Path-Finding GCN version
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class GraphDQN(nn.Module):
    """
    Overall structure (per forward pass):

    1) Graph Branch (replacing the convolutional branch):
       - GCN1: node_feature_dim -> 32, then BN, LeakyReLU, Dropout
       - GCN2: 32 -> 64, then BN, LeakyReLU, Dropout
       - GCN3: 64 -> 128, then BN, LeakyReLU, Dropout
       - GCN4: 128 -> 128, then BN, LeakyReLU, Dropout
       - global_mean_pool over nodes yields a vector of size 128 per graph
       - Fully connected layer: Linear(128 -> fc_hidden) with LeakyReLU

    2) Low-Dim Branch (for direction and distance):
       - Fully connected layer: Linear(3 -> fc_hidden) with LeakyReLU

    3) Combined Branch:
       - Concatenate the graph branch output and low-dim branch output (dimension: 2 * fc_hidden)
       - Fully connected layer: Linear(2 * fc_hidden -> fc_hidden) with LeakyReLU

    4) Final Output:
       - Linear(fc_hidden -> num_actions) to yield Q-values.
    """
    def __init__(self, node_feature_dim, num_actions=4, fc_hidden=128):
        """
        Args:
            node_feature_dim (int): Number of features per node.
            num_actions (int): Number of discrete actions.
            fc_hidden (int): Hidden dimension for the fully connected layers.
        """
        super(GraphDQN, self).__init__()
        
        # --- Graph Branch ---
        self.gcn1 = GCNConv(node_feature_dim, 32)
        self.bn1  = BatchNorm(32)
        self.gcn2 = GCNConv(32, 64)
        self.bn2  = BatchNorm(64)
        self.gcn3 = GCNConv(64, 128)
        self.bn3  = BatchNorm(128)
        self.gcn4 = GCNConv(128, 128)
        self.bn4  = BatchNorm(128)
        
        # Graph-level MLP after pooling:
        self.fc_graph = nn.Linear(128, fc_hidden)
        
        # --- Low-dimensional Branch ---
        self.fc_low = nn.Linear(3, fc_hidden)
        
        # --- Combined Branch ---
        self.fc_combined = nn.Linear(2 * fc_hidden, fc_hidden)
        
        # --- Final Q-value Output ---
        self.out = nn.Linear(fc_hidden, num_actions)
        
        # Weight initialization for linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        """
        Expects a PyG Data object with:
          - data.x: Node features of shape [num_nodes, node_feature_dim]
          - data.edge_index: Graph connectivity (COO format)
          - data.batch: A vector assigning each node to a graph in the batch
          - data.low: Low-dimensional features of shape [batch_size, 3] (direction_x, direction_y, distance)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- Graph Branch ---
        x = F.leaky_relu(self.bn1(self.gcn1(x, edge_index)), negative_slope=0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = F.leaky_relu(self.bn2(self.gcn2(x, edge_index)), negative_slope=0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = F.leaky_relu(self.bn3(self.gcn3(x, edge_index)), negative_slope=0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = F.leaky_relu(self.bn4(self.gcn4(x, edge_index)), negative_slope=0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Global mean pooling over nodes.
        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.fc_graph(x), negative_slope=0.01)
        
        # --- Low-dimensional Branch ---
        low = F.leaky_relu(self.fc_low(data.low), negative_slope=0.01)
        
        # --- Combine Both Branches ---
        combined = torch.cat([x, low], dim=1)
        combined = F.leaky_relu(self.fc_combined(combined), negative_slope=0.01)
        
        # --- Final Output for Q-values ---
        q_values = self.out(combined)
        return q_values

