"""Goal-conditioned V-JEPA 2.1 latent world model.

Given a start frame and an SE(2) goal pose, the model predicts the latent
features (h_1..h_K) of the K future frames along the path to the goal.

Design (see README "Design decisions"):
  * encoder  = V-JEPA 2.1 ViT-L, frozen except LayerNorm; encodes every frame
    through the *image* path (T=1 clips) so each frame yields P = 256 patch
    tokens (one latent h_t per frame).
  * target_encoder = frozen copy of the pretrained encoder; its LayerNorm-ed
    features on the future frames are the regression targets.
  * predictor = V-JEPA 2.1 predictor, head re-initialised to 1024 dims
    (the encoder's embed dim).
  * goal = a dedicated GOAL token prepended to the predictor context; its grid
    index is one past the (K+1)*P grid so it never collides with visual tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ENC = dict(
    img_size=256,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    uniform_power=False,
    use_rope=True,
    img_temporal_dim_size=1,
    interpolate_rope=True,
    modality_embedding=True,
)


class GoalMLP(nn.Module):
    """(x, y, yaw) -> (B, D). Encodes the goal in the encoder feature space."""

    def __init__(self, embed_dim: int = 1024, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x, y, yaw = goal[:, 0:1], goal[:, 1:2], goal[:, 2]
        inp = torch.cat(
            [
                x / 10.0,
                y / 10.0,
                torch.cos(yaw).unsqueeze(1),
                torch.sin(yaw).unsqueeze(1),
            ],
            dim=1,
        )
        return self.net(inp)


class GoalVJEPA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        goal_mlp: nn.Module,
        target_encoder: nn.Module,
        num_future: int,
        img_size: int = 256,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.goal_mlp = goal_mlp
        self.target_encoder = target_encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.num_future = num_future
        self.patch_size = patch_size
        self.img_size = img_size
        self.p = (img_size // patch_size) ** 2  # tokens per frame
        # Goal token index: one past the (K+1) * P grid.
        self.goal_index = (num_future + 1) * self.p
        self.register_buffer("_arange_p", torch.arange(self.p), persistent=False)

    # ---- construction helpers --------------------------------------------- #

    @classmethod
    def build(
        cls,
        num_future: int,
        img_size: int = 256,
        patch_size: int = 16,
        device: torch.device | str = "cpu",
    ) -> "GoalVJEPA":
        from app.vjepa_2_1.models.predictor import vit_predictor
        from app.vjepa_2_1.models.vision_transformer import vit_large

        enc_kwargs = dict(DEFAULT_ENC)
        enc_kwargs["img_size"] = img_size
        enc_kwargs["patch_size"] = patch_size
        encoder = vit_large(**enc_kwargs)

        # Predictor with output head sized to the encoder embed dim (1024).
        # num_frames = K+2 so num_patches covers the goal token's index.
        predictor = vit_predictor(
            img_size=img_size,
            patch_size=patch_size,
            use_mask_tokens=True,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=384,
            teacher_embed_dim=encoder.embed_dim,
            num_frames=num_future + 2,
            tubelet_size=1,
            depth=12,
            num_heads=12,
            num_mask_tokens=8,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=1,
            return_all_tokens=True,
            img_temporal_dim_size=1,
        )

        target_encoder = type(encoder)(**enc_kwargs)
        goal_mlp = GoalMLP(embed_dim=encoder.embed_dim)
        model = cls(
            encoder, predictor, goal_mlp, target_encoder, num_future, img_size, patch_size
        )
        model.to(device)
        return model

    def freeze_encoder_except_ln(self) -> None:
        """Freeze the student encoder; keep only LayerNorm parameters trainable."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for m in self.encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    # ---- forward ----------------------------------------------------------- #

    def _encode_frames(self, clips: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """(B, C, 1, H, W) -> (B, P, D); route through the image path."""
        return net(clips, training=False)

    def forward_predict(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Goal-conditioned prediction of future latents -> (B, K*P, D)."""
        B = start.shape[0]
        P = self.p
        K = self.num_future

        z = self._encode_frames(start, self.encoder)  # (B, P, D)
        goal_emb = self.goal_mlp(goal)  # (B, D)
        x = torch.cat([goal_emb.unsqueeze(1), z], dim=1)  # (B, P+1, D)

        goal_idx = torch.full(
            (B, 1), self.goal_index, dtype=torch.long, device=start.device
        )
        masks_x = torch.cat([goal_idx, self._arange_p.unsqueeze(0).expand(B, -1)], dim=1)
        masks_y = torch.arange(
            P, (K + 1) * P, dtype=torch.long, device=start.device
        ).unsqueeze(0).expand(B, -1)

        z_pred, _ = self.predictor(x, masks_x, masks_y, mod="image", mask_index=0)
        return z_pred

    def encode_targets(self, future: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder latents of the future frames -> (B, K*P, D)."""
        B, K = future.shape[0], future.shape[1]
        with torch.no_grad():
            frames = future.reshape(B * K, 3, 1, future.shape[-2], future.shape[-1])
            h = self._encode_frames(frames, self.target_encoder)  # (B*K, P, D)
            h = F.layer_norm(h, (h.size(-1),))
            h = h.reshape(B, K * self.p, -1)
        return h

    @torch.no_grad()
    def encode_frames_latent(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) -> (B, P, D) layer-normed teacher latents (for the decoder)."""
        h = self._encode_frames(frames, self.target_encoder)
        return F.layer_norm(h, (h.size(-1),))

    def forward(
        self, start: torch.Tensor, future: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_pred = self.forward_predict(start, goal)
        h = self.encode_targets(future)
        return z_pred, h

    @torch.no_grad()
    def predict_latents(
        self, start: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        return self.forward_predict(start, goal)


def vjepa_loss(z_pred: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Dense predictive loss: L1 in latent space (V-JEPA 2.1, loss_exp=1)."""
    return torch.mean(torch.abs(z_pred - h))
