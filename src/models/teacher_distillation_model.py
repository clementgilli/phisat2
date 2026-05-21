import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Phisat2FeatureExtractor(nn.Module):
    def __init__(self, full_model: nn.Module):
        """
        Extracts only the encoding pathway from the full U-Net model.
        Drops the decoder and all task-specific heads to save VRAM.
        """
        super().__init__()
        # Extract only the modules needed to reach the latent space
        self.stem = full_model.stem
        self.encoder = full_model.encoder
        self.bridge = full_model.bridge
        self.pool_feats = full_model.pool_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Pooled feature vector (B, dims[-1])
        """
        x_stem = self.stem(x)
        bottom, _ = self.encoder(x_stem) # We ignore the skip connections
        bottom_feats = self.bridge(bottom)
        pooled_feats = self.pool_feats(bottom_feats)
        
        return pooled_feats

class DomainFeatureDistiller(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 640, hidden_dim: int = 512, use_predictor: bool = True, loss_type: str = "cosine"):
        """
        Cross-domain feature distillation wrapper.
        Expects a pure encoder that outputs a 1D feature vector.
        
        Args:
            encoder: The base feature extractor.
            feature_dim: Output dimension of the encoder.
            hidden_dim: Hidden dimension for the predictor MLP.
            use_predictor: If True, uses the SimSiam predictor. If False, bypasses it.
            loss_type: "cosine" for negative cosine similarity, "mse" for strict L2 distance.
        """
        super().__init__()
        self.loss_type = loss_type
        
        # CPU transfer to prevent CUDA fragmentation during deepcopy
        device_before = next(encoder.parameters()).device
        encoder.cpu() 
        
        # 1. TEACHER NETWORK (Target - Frozen)
        self.teacher_encoder = copy.deepcopy(encoder)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        self.teacher_encoder.eval()
        
        # 2. STUDENT NETWORK (Online - Trainable)
        self.student_encoder = encoder
        
        # 3. PREDICTOR (Ablation toggle)
        if use_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feature_dim)
            )
        else:
            # nn.Identity acts as a transparent pass-through layer
            self.predictor = nn.Identity()
        
        # Restore to original device
        self.to(device_before)

    def train(self, mode: bool = True):
        """
        Ensures the Teacher's BatchNorm layers remain frozen during training.
        """
        super().train(mode)
        self.teacher_encoder.eval()

    def forward(self, img_real: torch.Tensor, img_sim: torch.Tensor) -> torch.Tensor:
        """
        Computes the alignment loss based on selected configuration.
        """
        # --- Teacher Path (No gradients) ---
        with torch.no_grad():
            z_sim = self.teacher_encoder(img_sim)
            z_sim = z_sim.detach().clone()
            
        # --- Student Path (Requires gradients) ---
        z_real = self.student_encoder(img_real)
        p_real = self.predictor(z_real)
        
        # --- Loss Calculation ---
        if self.loss_type == "cosine":
            p_real_norm = F.normalize(p_real, dim=-1, p=2)
            z_sim_norm = F.normalize(z_sim, dim=-1, p=2)
            loss = - (p_real_norm * z_sim_norm).sum(dim=-1).mean()
            
        elif self.loss_type == "mse":
            loss = F.mse_loss(p_real, z_sim)
            
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose 'cosine' or 'mse'.")
        
        return loss