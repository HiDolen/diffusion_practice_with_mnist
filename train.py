# %% [markdown]
# ## 准备

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch import Tensor
from einops import rearrange

from pl_utils.nn import get_latent_2d_ids, RotaryEmbeddingMultiDimension, apply_rotary_emb

# %%
from pl_utils import init_before_training

init_before_training()

# %% [markdown]
# ## 数据集

# %% [markdown]
# ### 纯黑和纯白

# %%
from PIL import Image
import numpy as np

class BlackWhiteDataset(torch.utils.data.Dataset):
    def __init__(self, num_black, num_white, transform=None):
        self.num_black = num_black
        self.num_white = num_white
        self.transform = transform
        self.data = []
        self.targets = []

        # 创建纯黑图
        for _ in range(num_black):
            img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
            self.data.append(img)
            self.targets.append(10)  # 标签10表示纯黑图

        # 创建纯白图
        for _ in range(num_white):
            img = Image.fromarray(np.ones((28, 28), dtype=np.uint8) * 255)
            self.data.append(img)
            self.targets.append(11)  # 标签11表示纯白图

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


black_white_dataset = BlackWhiteDataset(1000, 1000, transform=transforms.ToTensor())

# %% [markdown]
# ### 创建 MINIST 数据集

# %%
from pl_utils.dataset import get_train_val_dataloader

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
dataset = ConcatDataset([train_dataset, test_dataset, black_white_dataset])
dataset = ConcatDataset([train_dataset, black_white_dataset])
train_loader, _ = get_train_val_dataloader(
    dataset,
    batch_size=256,
    test_size=0,
    num_workers=6,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# %% [markdown]
# ## 模型

# %% [markdown]
# ### AdaLayerNormZero

# %%
class AdaLayerNormZero(nn.Module):
    r"""
    改变自 diffusers 库的 AdaLayerNormZero 类。

    Norm layer adaptive layer norm zero (adaLN-Zero).

    """

    def __init__(
        self,
        embedding_dim: int,
        bias=True,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor = None,
    ):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

# %% [markdown]
# ### AdaLayerNormContinuous

# %%
class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        bias=True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor = None,
    ):
        emb = self.linear(self.silu(emb))
        shift, scale = emb.chunk(2, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states

# %% [markdown]
# ### Attention

# %%
from torch.nn import RMSNorm


class Attention(nn.Module):
    """只进行 attention，无 MLP 也无残差。直接返回 attention 结果。"""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        attn_dropout: float,
        attn_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attn_dropout = attn_dropout
        self.attn_bias = attn_bias

        hidden_size = head_dim * num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attn_bias)

        self.q_proj_add = nn.Linear(hidden_size, num_heads * head_dim, bias=attn_bias)
        self.k_proj_add = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.v_proj_add = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.q_norm_add = RMSNorm(head_dim)
        self.k_norm_add = RMSNorm(head_dim)
        self.o_proj_add = nn.Linear(num_heads * head_dim, hidden_size, bias=attn_bias)

    def forward(
        self,
        hidden_states,
        rotary_emb=None,
        attn_mask=None,
        add_hidden_states=None,
    ):
        """

        Args:
            hidden_states: [b_size, seq, channel]
        """
        B, S, C = hidden_states.size()
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if add_hidden_states is not None:
            add_q = self.q_proj_add(add_hidden_states)
            add_k = self.k_proj_add(add_hidden_states)
            add_v = self.v_proj_add(add_hidden_states)
            add_q = add_q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            add_k = add_k.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            add_v = add_v.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            add_q = self.q_norm_add(add_q)
            add_k = self.k_norm_add(add_k)
            q = torch.cat([add_q, q], dim=2)
            k = torch.cat([add_k, k], dim=2)
            v = torch.cat([add_v, v], dim=2)

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb)
            k = apply_rotary_emb(k, rotary_emb)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
            attn_mask=attn_mask,
            enable_gqa=True,
        )  # [B, num_heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, -1, C)  # [B, seq, hidden_size]

        if add_hidden_states is not None:
            add_hidden_states, hidden_states = (
                attn_output[:, : add_hidden_states.size(1)],
                attn_output[:, add_hidden_states.size(1) :],
            )
            add_hidden_states = self.o_proj_add(add_hidden_states)
            hidden_states = self.o_proj(hidden_states)

            return hidden_states, add_hidden_states

        hidden_states = self.o_proj(attn_output)
        return hidden_states, None


# %% [markdown]
# ### FeedForward

# %%
class FeedForward(nn.Module):
    """
    作为 Transformer 的 MLP 部分。
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size_factor: int = 4,
        mlp_dropout: float = 0.0,
        mlp_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_hidden_size_factor, bias=mlp_bias),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_size * mlp_hidden_size_factor, hidden_size, bias=mlp_bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)

# %% [markdown]
# ### TransformerBlock

# %%
class TransformerBlock(nn.Module):

    def __init__(
        self,
        transformer_dim=256,
        **kwargs,
    ):
        super().__init__()

        self.adanorm = AdaLayerNormZero(transformer_dim)
        self.adanorm_add = AdaLayerNormZero(transformer_dim)

        self.attn = Attention(**kwargs)

        self.mlp = FeedForward(hidden_size=transformer_dim, **kwargs)
        self.mlp_add = FeedForward(hidden_size=transformer_dim, **kwargs)

        self.attn_norm = nn.LayerNorm(transformer_dim, elementwise_affine=False, eps=1e-6)
        self.attn_norm_add = nn.LayerNorm(transformer_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(transformer_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm_add = nn.LayerNorm(transformer_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states,
        temb,
        rotary_emb=None,
        attn_mask=None,
        add_hidden_states=None,
    ):
        # AdaNorm
        hidden_states = self.attn_norm(hidden_states)
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adanorm(
            hidden_states, emb=temb
        )
        if add_hidden_states is not None:
            add_hidden_states = self.attn_norm_add(add_hidden_states)
            norm_add_hidden, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.adanorm_add(
                add_hidden_states, emb=temb
            )

        # attention
        attn_output, add_attn_output = self.attn(
            norm_hidden,
            rotary_emb,
            attn_mask=attn_mask,
            add_hidden_states=norm_add_hidden if add_hidden_states is not None else None,
        )
        # residual
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        # norm
        norm_hidden = self.mlp_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        # MLP
        mlp_output = self.mlp(norm_hidden)
        # residual
        hidden_states = hidden_states + mlp_output * gate_mlp.unsqueeze(1)

        if add_hidden_states is not None:
            # residual
            add_hidden_states = add_hidden_states + add_attn_output * c_gate_msa.unsqueeze(1)
            # norm
            norm_add_hidden = self.mlp_norm_add(add_hidden_states)
            norm_add_hidden = norm_add_hidden * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            # MLP
            add_mlp_output = self.mlp_add(norm_add_hidden)
            # residual
            add_hidden_states = add_hidden_states + add_mlp_output * c_gate_mlp.unsqueeze(1)

            return hidden_states, add_hidden_states

        return hidden_states, None


# %% [markdown]
# ### Transformer2DModel

# %%
from pl_utils.nn import TimeStepSinusoidalEmbbedding


class Transformer2DModel(nn.Module):

    def __init__(
        self,
        in_channels: int = 16,
        *,
        transformer_dim: int = 256,
        num_layers: int = 12,
        time_emb_dim=128,
        class_emb_dim=128,
        num_classes=10,
        head_dim: int = 64,
        rope_theta: int = 10000,
        **kwargs,
    ):
        super().__init__()

        self.transformer_dim = transformer_dim
        self.num_layers = num_layers

        self.in_proj = nn.Linear(in_channels, transformer_dim)

        self.time_embed = nn.Sequential(
            TimeStepSinusoidalEmbbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, transformer_dim),
        )
        self.class_embed = nn.Sequential(
            TimeStepSinusoidalEmbbedding(class_emb_dim, 10000),
            nn.Linear(class_emb_dim, class_emb_dim),
            nn.SiLU(),
            nn.Linear(class_emb_dim, transformer_dim),
        )

        self.pos_embed = RotaryEmbeddingMultiDimension(rope_theta, dim=[head_dim // 2, head_dim // 2])

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(transformer_dim=transformer_dim, head_dim=head_dim, **kwargs)
                for _ in range(num_layers)
            ]
        )

        self.out_adanorm = AdaLayerNormContinuous(transformer_dim)
        self.out_proj = nn.Linear(transformer_dim, in_channels)

    def forward(
        self,
        latents,
        t,
        class_idx,
        pos_ids,
        attn_mask=None,
    ):
        """

        Args:
            latents: [b, seq, c]
            t: [b]
            class_idx: [b]
            pos_ids: [seq, 2]
        """
        device = latents.device
        B, S, C = latents.size()

        hidden_states = self.in_proj(latents)

        # time & class embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(B)
        t_emb = self.time_embed(t.to(device))  # [b, transformer_dim]
        if class_idx.dim() == 0:
            class_idx = class_idx.unsqueeze(0).repeat(B)
        class_emb = self.class_embed(class_idx.to(device))  # [b, transformer_dim]

        # temb
        temb = t_emb
        # add_hidden_states
        add_hidden_states = class_emb.unsqueeze(1)

        # rotary emb
        if add_hidden_states is not None:
            add_pos_ids = torch.ones(add_hidden_states.size(1), 2).to(pos_ids.device)
            pos_ids = torch.cat([pos_ids, add_pos_ids], dim=0)  # [seq + n, 2]
        rotary_emb = self.pos_embed(pos_ids)

        for block in self.blocks:
            hidden_states, add_hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
                attn_mask=attn_mask,
                add_hidden_states=add_hidden_states,
            )

        hidden_states = self.out_adanorm(hidden_states, emb=temb)
        latents = self.out_proj(hidden_states)

        return latents


# %% [markdown]
# ### DiffusionModel

# %%
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """
    取自 diffusers 库的 calculate_shift 函数。

    用于动态决定时间步的偏移程度，使其与分辨率匹配。

    本质是线性函数。
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# %%
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from tqdm import tqdm
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment


class DiffusionModel(nn.Module):

    def __init__(
        self,
        img_channels: int = 1,
        patch_size: int = 2,
        num_train_timesteps: int = 1000,
        **kwargs,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.patch_size = patch_size
        self.num_train_timesteps = num_train_timesteps

        mu = calculate_shift((28 // patch_size) ** 2)
        self.train_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            use_dynamic_shifting=True,
        )
        self.train_scheduler.set_timesteps(num_train_timesteps, mu=mu)
        self.infer_scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
        )

        self.transformer_model = Transformer2DModel(
            in_channels=img_channels * patch_size**2,
            **kwargs,
        )

    def img_normalize(self, img):
        return (img - 0.5) / 0.3081

    def img_denormalize(self, img):
        return img * 0.3081 + 0.5

    def pack_latent(self, latents):
        return rearrange(
            latents,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def unpack_latent(self, latents, height, width):
        return rearrange(
            latents,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=height,
            w=width,
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def get_train_loss(self, samples, t, class_idx):
        device = samples.device
        B, C, H, W = samples.size()

        # prepare latent variables
        latents = self.img_normalize(samples)
        latents = self.pack_latent(latents)
        latents_pos_ids = get_latent_2d_ids(H // self.patch_size, W // self.patch_size)
        noise = torch.randn_like(latents).to(device)

        ### offset noise
        # noise += 0.1 * torch.randn(noise.size(0), 1, noise.size(2)).to(device)
        ### offset noise

        ### immiscible diffusion
        # a = latents.view(B, -1)
        # b = noise.view(B, -1)
        # dist = torch.cdist(a, b)
        # _, idx = linear_sum_assignment(dist.cpu())
        # noise = noise[torch.from_numpy(idx).to(device)]
        ### immiscible diffusion

        # timesteps to sigmas
        t = -t - 1
        sigmas = self.train_scheduler.sigmas[t.to("cpu")][:, None, None].to(device)

        # get noised latents
        noised_latents = (1.0 - sigmas) * latents + sigmas * noise
        # get pred latents
        pred_latents = self.transformer_model(
            noised_latents,
            t,
            class_idx,
            pos_ids=latents_pos_ids,
        )
        # get loss
        loss = F.mse_loss(noise - latents, pred_latents)
        return loss

    def forward(self, samples, t, class_idx):
        return self.get_train_loss(samples, t, class_idx)

    def sample_from_scratch(
        self,
        batch_size,
        height,
        width,
        class_idx,
        num_timesteps,
        latents=None,
    ):
        device = next(self.parameters()).device

        # prepare latent variables
        if latents is None:
            latents = torch.randn(batch_size, self.img_channels, height, width)
        latents = latents.to(device)
        latents = self.pack_latent(latents)
        latents_pos_ids = get_latent_2d_ids(height // self.patch_size, width // self.patch_size)

        # timesteps
        mu = calculate_shift(latents.size(1))
        self.infer_scheduler.set_timesteps(num_timesteps, mu=mu, device=device)
        timesteps_list = self.infer_scheduler.timesteps

        # denoising loop
        for timestep in tqdm(timesteps_list, leave=False):
            with torch.no_grad():
                noise_pred = self.transformer_model(
                    latents=latents,
                    t=timestep.to(latents.device),
                    class_idx=class_idx.to(latents.device),
                    pos_ids=latents_pos_ids,
                )
            latents = self.infer_scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        # unpack latents
        latents = self.unpack_latent(latents, height // self.patch_size, width // self.patch_size)

        # denormalize
        samples = self.img_denormalize(latents)

        return samples

# %% [markdown]
# ## Lightning

# %%
from pl_utils import BaseModule
from torchvision.utils import make_grid, save_image
import os


class LightningModel(BaseModule):

    def __init__(
        self,
        model: DiffusionModel,
        learning_rate_config,
        training_config,
        ema_update_callback=None,
    ):
        super().__init__(model, learning_rate_config, training_config)
        self.ema_update_callback = ema_update_callback

        self.model.apply(self._init_weight)

    def forward(self, batch_size=1, height=28, width=28, class_idx=0, num_timesteps=10):
        return self.model(batch_size, height, width, class_idx, num_timesteps)

    def training_step(self, batch, batch_idx):
        samples, class_idx = batch
        batch_size = samples.size(0)

        timesteps = torch.normal(0, 1, (batch_size,)).sigmoid()
        timesteps = (timesteps * (self.model.num_train_timesteps - 1)).long()

        loss = self.model(samples, timesteps, class_idx)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_update_callback:
            self.ema_update_callback()
        pass

    def on_train_epoch_end(self):
        label_indices = torch.arange(12)
        if not hasattr(self, "tmp_latents"):
            self.tmp_latents = torch.randn(
                1,
                self.model.img_channels,
                28,
                28,
            ).repeat(12, 1, 1, 1)
        self.model.eval()
        samples = self.model.sample_from_scratch(
            12, 28, 28, label_indices, 20, latents=self.tmp_latents
        )

        images = (samples.clamp(0, 1) * 255).to(torch.uint8)
        grid = make_grid(images, nrow=4, pad_value=255, value_range=(0, 255))
        os.makedirs('./training_images', exist_ok=True)
        save_image(grid / 255, f'./training_images/epoch_{self.current_epoch}.png')

    def _init_weight(self, module):
        std = 0.002
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)

# %% [markdown]
# ## 配置与实例化

# %% [markdown]
# ### 配置

# %%
from pl_utils import LearningRateConfig, TrainingConfig

learning_rate_config = LearningRateConfig(
    lr_warmup_steps=1000,
    lr_initial=1e-6,
    lr_max=2e-4,
    lr_end=2e-4,
)

training_config = TrainingConfig(
    optimizer='adamw',
    optimizer_args={
        # 'betas': (0.9, 0.999),
        # 'weight_decay': 1e-4,
        "fused": True,
    },
)

layer_config = {
    # DiffusionModel
    "img_channels": 1,
    "patch_size": 2,
    "num_train_timesteps": 1000,
    # Transformer2DModel
    "transformer_dim": 256,
    "num_layers": 3,
    "time_emb_dim": 256,
    "class_emb_dim": 256,
    "num_classes": 12,
    "head_dim": 64,
    "rope_theta": 100,
    # TransformerBlock
    # Attention
    "num_heads": 4,
    "num_kv_heads": 1,
    "attn_dropout": 0.0,
    "attn_bias": True,
    # FeedForward
    "mlp_hidden_size_factor": 4,
    "mlp_dropout": 0.0,
    "mlp_bias": True,
}

# %% [markdown]
# ### 实例化

# %%
model = DiffusionModel(**layer_config).to('cuda')

pl_model = LightningModel(
    model,
    learning_rate_config,
    training_config,
    ema_update_callback=None,
)

model.compile(
    fullgraph=False,
    dynamic=False,
    options={
        "shape_padding": True,
        "max_autotune": True,
        # "triton.cudagraphs": True,
        # "trace.graph_diagram": True,
    },
)

# %% [markdown]
# ## 正式训练

# %%
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint


checkpoint_callback = ModelCheckpoint(
    save_top_k=0,
    # monitor="val/reconstruct_loss",
    # mode="min",
    # dirpath=r"/mnt/e/vae_models",
    # save_weights_only=True,
)

trainer = L.Trainer(
    logger=False,
    accelerator='gpu',
    max_epochs=300,
    precision='bf16-mixed',
    # precision='32',
    log_every_n_steps=4,
    default_root_dir="./",
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0,
)

trainer.fit(pl_model, train_loader)
