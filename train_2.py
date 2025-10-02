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

from typing import Tuple, Union, List

# %%
from pl_utils import init_before_training

init_before_training(empty_cache=False)

# %%
def is_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell != None
    except NameError:
        return False

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
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
dataset = ConcatDataset([train_dataset, test_dataset, black_white_dataset])
dataset = ConcatDataset([train_dataset, black_white_dataset])

workers = 6 if not is_jupyter() else 0
train_loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True if workers > 0 else False,
)


# %%
if is_jupyter():
    print(f"Mean: {(train_dataset.data.float() / 255).mean().item()}")
    print(f"Std: {(train_dataset.data.float() / 255).std().item()}")

# %%
if is_jupyter():
    print(len(train_loader))
    for data, target in train_loader:
        print(data.shape)
        print(target.shape)
        print(data[0][0][10:20, 10:20])
        break


# %% [markdown]
# ## 模型

# %% [markdown]
# ### Config

# %%
from transformers.configuration_utils import PretrainedConfig


class Config(PretrainedConfig):

    model_type = "test_flow_match"

    def __init__(
        self,
        hidden_size=384,
        depth=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        attention_dropout=0.0,
        intermediate_size=384 * 4,
        latent_size=None,
        #
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        #
        time_embed_dim=128,
        time_embed_theta=10000.0,
        #
        initializer_range=0.02,
        register_token_num=4,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.latent_size = latent_size

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.time_embed_dim = time_embed_dim
        self.time_embed_theta = time_embed_theta

        self.initializer_range = initializer_range
        self.register_token_num = register_token_num

        super().__init__(**kwargs)

# %% [markdown]
# ### AdaLayerNormZero

# %%
class AdaLayerNormZero(nn.Module):
    r"""
    改变自 diffusers 库的 AdaLayerNormZero 类。

    用时间编码 emb 调整 hidden_states，并额外输出 gate_msa, shift_mlp, scale_mlp, gate_mlp。
    """

    def __init__(self, config: Config):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            emb (torch.Tensor): (batch_size, hidden_size)
        """
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

# %% [markdown]
# ### AdaLayerNormContinuous

# %%
class AdaLayerNormContinuous(nn.Module):
    r"""
    改变自 diffusers 库的 AdaLayerNormZero 类。

    用时间编码 emb 调整 hidden_states。
    """

    def __init__(self, config: Config):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            emb (torch.Tensor): (batch_size, hidden_size)
        """
        emb = self.linear(self.silu(emb))
        shift, scale = emb.chunk(2, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states

# %% [markdown]
# ### RotaryEmbedding

# %%
def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
):
    """施加旋转位置编码。改编自 diffusers 库的 apply_rotary_emb 函数。

    Args:
        x (`torch.Tensor`): [batch, head, seq, dim]
        freqs_cis (`Tuple[torch.Tensor]`): 预先准备好的旋转位置编码。([seq, dim], [seq, dim],)
    """
    cos, sin = freqs_cis  # [seq, dim]
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    # rotate half
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [batch, seq, head, dim//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out

# %%
class RotaryEmbeddingMultiDimension(nn.Module):
    """多维度旋转位置编码。

    能够返回多维度 RoPE 的 cos 和 sin，可用于生成像是 ViT 的二维位置编码。

    生成的 RoPE 编码需要在 attention 环节借助 `apply_rotary_emb()` 向 kv 施加。
    """

    def __init__(
        self,
        dims: List[int] = [64, 64],
        theta: int = 10000,
    ):
        """

        Args:
            dim (`List[int]`): 每个维度分别的 channel 数。sum(dim) 应等于单个注意力头的维度数。\
                会间接影响 RoPE 所表示维度的数量。
            theta (`int`): 设置 theta 值
        """
        super().__init__()
        self.dims = dims
        self.theta = theta

        for i, d in enumerate(dims):
            inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.register_buffer(f"inv_freq_{i}", inv_freq)

    def _get_1d_rotary_pos_embed(self, pos_ids, dim_idx):
        """改编自 diffusers 库的 get_1d_rotary_pos_embed 函数。"""
        inv_freq = getattr(self, f"inv_freq_{dim_idx}")
        freqs = torch.outer(pos_ids, inv_freq)  # type: ignore   # [S, D/2]

        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin

    def forward(self, x, pos_ids: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (`torch.Tensor`): 仅用来确定 device 和 dtype
            pos_ids (`torch.Tensor`): [seq_len, dim_count]
        """
        dim_count = len(self.dims)
        assert dim_count == pos_ids.size(-1), f"dim_count should be {dim_count}, but got {pos_ids.size(-1)}"

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            cos_out, sin_out = [], []
            pos_ids = pos_ids.float()
            for i in range(dim_count):
                cos, sin = self._get_1d_rotary_pos_embed(pos_ids[:, i], i)
                cos_out.append(cos)
                sin_out.append(sin)
            cos = torch.cat(cos_out, dim=-1).to(device=x.device)
            sin = torch.cat(sin_out, dim=-1).to(device=x.device)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# %% [markdown]
# ### Attention

# %%
from torch.nn import RMSNorm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class Attention(nn.Module):
    """只进行 attention，无 MLP 也无残差。直接返回 attention 结果。"""

    def __init__(
        self,
        # head_dim: int,
        # num_heads: int,
        # num_kv_heads: int,
        # attn_dropout: float,
        # attn_bias: bool = False,
        # **kwargs,
        config: Config,
    ):
        super().__init__()
        self.is_causal = False
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
    ):
        """

        Args:
            hidden_states: [b_size, seq, head_dim]
        """
        input_shape = hidden_states.size()[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [..., seq, head_dim]

        device = hidden_states.device

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)


        return attn_output, attn_weights

# %% [markdown]
# ### MLP

# %%
from transformers.activations import ACT2FN

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# %% [markdown]
# ### TransformerLayer

# %%
class TransformerLayer(nn.Module):

    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.adanorm = AdaLayerNormZero(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states,
        temb,
        position_embeddings=None,
        attention_mask=None,
    ):
        # 1. AdaNorm and attention
        residual = hidden_states
        hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adanorm(hidden_states, emb=temb)
        hidden_states, _ = self.attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states * gate_msa.unsqueeze(1)
        hidden_states = residual + hidden_states

        # 2. MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states * gate_mlp.unsqueeze(1)
        hidden_states = residual + hidden_states

        

        return hidden_states

# %% [markdown]
# ### TimeStepSinusoidalEmbbedding

# %%
class TimeStepSinusoidalEmbbedding(nn.Module):
    """
    将时间步转换为正余弦嵌入
    """

    def __init__(
        self,
        dim,
        base=10000,
    ):
        super().__init__()
        assert dim % 2 == 0, 'dim must be even'
        self.dim = dim
        
        half_dim = dim // 2
        exponent = (
            -torch.arange(half_dim)
            / (half_dim - 1)
            * torch.log(torch.tensor(base, dtype=torch.float32))
        )
        frequencies = torch.exp(exponent)
        self.register_buffer('frequencies', frequencies)


    def forward(self, timesteps):
        """
        Args:
            timesteps: Tensor: 整数张量，形状为 [batch]
        """
        angles = timesteps.unsqueeze(-1).float() * self.frequencies  # [batch, half_dim]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [batch, dim]
        return emb

# %% [markdown]
# ### TransformerModel

# %%
class TransformerModel(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = config

        self.time_embed = nn.Sequential(
            TimeStepSinusoidalEmbbedding(config.time_embed_dim, config.time_embed_theta),
            nn.Linear(config.time_embed_dim, config.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        )

        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.depth)])
        self.out_adanorm = AdaLayerNormContinuous(config)

        register_size = (config.register_token_num, config.hidden_size)
        self.register_tokens = nn.Parameter(torch.normal(0, 0.02, size=register_size))

    def forward(
        self,
        hidden_states,
        t,
        position_embeddings,
        attention_mask=None,
    ):
        """

        Args:
            hidden_states: [b, seq, c]
            t: [b]
            position_ids: [seq, 2]
        """
        B, S, C = hidden_states.size()

        # time embedding
        temb = self.time_embed(t)  # should be [b, hidden_size]
        if len(temb.shape) == 1:
            temb = temb.unsqueeze(0)

        # registers
        rn = self.config.register_token_num
        hidden_states = torch.cat([hidden_states, self.register_tokens.unsqueeze(0).expand(B, -1, -1)], dim=1)

        def pad_rope(cos, sin):
            cos_padded = F.pad(cos, (0, 0, 0, rn), value=1.0)
            sin_padded = F.pad(sin, (0, 0, 0, rn), value=0.0)
            return cos_padded, sin_padded
        position_embeddings = pad_rope(*position_embeddings)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                temb=temb,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # remove registers
        hidden_states = hidden_states[:, :-rn, :] if rn > 0 else hidden_states

        # out adanorm
        hidden_states = self.out_adanorm(hidden_states, emb=temb)

        return hidden_states

# %%
if is_jupyter():
    test_config = Config(
        hidden_size=128,
        depth=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_dropout=0.0,
        intermediate_size=128 * 2,
        #
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        #
        time_embed_dim=32,
        time_embed_theta=10000.0,
        #
        initializer_range=0.02,
        register_token_num=4,
    )

    test_model = TransformerModel(test_config)
    test_hidden_states = torch.randn(2, 16, test_config.hidden_size)
    test_t = torch.randint(0, 1000, (2,))
    # test_pos_ids = torch.arange(16).unsqueeze(0).repeat(2, 1)
    test_position_embeddings = RotaryEmbeddingMultiDimension(dims=[128 // 4])(torch.randn(10),torch.arange(16).unsqueeze(1))

    test_out = test_model(test_hidden_states, test_t, test_position_embeddings)
    print(test_out.shape)  # should be [2, 16, hidden_size]

# %% [markdown]
# ### DiffusionModel

# %%
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from tqdm.notebook import tqdm
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment


class DiffusionModel(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.transformer_model = TransformerModel(config)

        self.in_proj = nn.Linear(config.latent_size, config.hidden_size) if config.latent_size else nn.Identity()
        self.out_proj = nn.Linear(config.hidden_size, config.latent_size) if config.latent_size else nn.Identity()

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
        )
        self.register_buffer("sigmas", self.scheduler.sigmas, persistent=False)
        self.inference_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
        )

    def get_train_loss(
        self,
        latents,
        cond,
        timestep_idx,
        position_embeddings,
        attention_mask=None,
    ):
        """

        Args:
            latents (`torch.Tensor`): [b, n, c]，需要预测的数据
            cond (`torch.Tensor`): [b, n', c]，条件生成控制信息，会拼接在 latents 开头
            timestep_idx (`torch.Tensor`): [b]，时间步索引
            position_embeddings (`torch.Tensor`)，latents 的 RoPE 位置编码
            attention_mask (`torch.Tensor`, *optional*)
        """

        # get noised latents
        noise = torch.randn_like(latents).to(latents.device)
        sigmas = self.sigmas[timestep_idx][:, None, None]
        noised_latents = (1.0 - sigmas) * latents + sigmas * noise

        # get flow match timestep
        sigma = self.sigmas[timestep_idx]
        timestep = sigma * 1000

        # update position embeddings because of cond concat (prepend cond + time token)
        def pad_rope(cos, sin):
            pad = (0, 0, cond.size(1), 0)
            cos_padded = F.pad(cos, pad, value=1.0)
            sin_padded = F.pad(sin, pad, value=0.0)
            return cos_padded, sin_padded

        position_embeddings = pad_rope(*position_embeddings)

        # latents to hiddens
        hidden_states = self.in_proj(noised_latents)

        # concat cond
        hidden_states = torch.cat([cond, hidden_states], dim=1)

        # get pred velocity
        v_pred = self.transformer_model(
            hidden_states,
            timestep,
            position_embeddings,
            attention_mask=attention_mask,
        )[:, cond.size(1) :, :]
        v_pred = self.out_proj(v_pred)

        # get loss
        v_target = noise - latents
        loss = F.mse_loss(v_target, v_pred)

        return loss

    @torch.inference_mode()
    def sample_from_scratch(
        self,
        latents,
        cond,
        position_embeddings,
        num_timesteps,
        progress_bar=True,
    ):
        """

        Args:
            latents (`torch.Tensor`): [b, n, c]，纯噪声 latent
            cond (`torch.Tensor`): [b, n', c]，条件生成控制信息，会拼接在 latents 开头
            position_embeddings (`torch.Tensor`)，latents 的 RoPE 位置编码
            num_timesteps (`int`): 采样的时间步数
            progress_bar (`bool`, *optional*): 是否显示进度条。默认为 True
        """
        device = next(self.parameters()).device

        # update position embeddings because of cond concat (prepend cond + time token)
        def pad_rope(cos, sin):
            pad = (0, 0, cond.size(1), 0)  # top-pad for prepended cond + time token
            cos_padded = F.pad(cos, pad, value=1.0)
            sin_padded = F.pad(sin, pad, value=0.0)
            return cos_padded, sin_padded

        position_embeddings = pad_rope(*position_embeddings)

        # prepare schedule
        self.inference_scheduler.set_timesteps(
            num_timesteps,
            mu=2,
            device=device,
        )
        sigma_list = self.inference_scheduler.sigmas
        
        # denoising loop
        for i in tqdm(range(num_timesteps), leave=False) if progress_bar else range(num_timesteps):
            timestep = sigma_list[i] * 1000

            hidden_states = self.in_proj(latents)
            hidden_states = torch.cat([cond, hidden_states], dim=1)
            velocity = self.transformer_model(
                hidden_states,
                t=timestep.to(latents.device),
                position_embeddings=position_embeddings,
            )[:, cond.size(1) :, :]
            velocity = self.out_proj(velocity)

            dt = sigma_list[i + 1] - sigma_list[i]
            latents = latents + velocity * dt

        return latents

# %% [markdown]
# ### Diffusion2DModelForPicGenerating

# %%
class Diffusion2DModelForPicGenerating(nn.Module):
    def __init__(self, config: Config, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size

        config.latent_size = self.latent_size

        self.diffusion_model = DiffusionModel(config)

        self.label_indices_embedding = nn.Embedding(12, config.hidden_size)

        self.rope_emb = RotaryEmbeddingMultiDimension(
            dims=[config.hidden_size // config.num_attention_heads // 2] * 2,
            theta=config.rope_theta,
        )

    def _img_normalize(self, img):
        # return (img - 0.1307) / 0.3081
        return 2 * img - 1.0

    def _img_denormalize(self, img):
        # return img * 0.3081 + 0.1307
        return (img + 1.0) / 2.0

    def _pack_latents(self, x):
        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def _unpack_latent(self, x, h, w):
        return rearrange(
            x,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=h,
            w=w,
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def get_train_loss(
        self,
        imgs,
        label_idx,
        timestep_idx,
    ):
        B, C, height, width = imgs.size()

        # prepare latents
        imgs = self._img_normalize(imgs)
        origin_latents = self._pack_latents(imgs)

        # prepare position ids，[N, 2]
        position_ids = torch.stack(
            torch.meshgrid(
                torch.arange(height // self.patch_size, device=imgs.device),
                torch.arange(width // self.patch_size, device=imgs.device),
                indexing="ij",
            ),
            dim=-1,  # [H, W, 2]
        ).view(-1, 2)
        # get position embeddings
        position_embeddings = self.rope_emb(imgs, position_ids)
        # label embedding
        label_emb = self.label_indices_embedding(label_idx).unsqueeze(1)  # [B, 1, hidden_size]

        # get loss
        loss = self.diffusion_model.get_train_loss(
            origin_latents,
            cond=label_emb,
            timestep_idx=timestep_idx,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )

        return loss

    @torch.inference_mode()
    def sample_from_scratch(
        self,
        label_idx,
        height,
        width,
        latents,
        num_timesteps=50,
        progress_bar=True,
    ):
        device = next(self.parameters()).device

        # prepare latents
        latents = self._pack_latents(latents)

        # prepare position ids，[N, 2]
        position_ids = torch.stack(
            torch.meshgrid(
                torch.arange(height // self.patch_size, device=device),
                torch.arange(width // self.patch_size, device=device),
                indexing="ij",
            ),
            dim=-1,  # [H, W, 2]
        ).view(-1, 2)
        # get position embeddings
        position_embeddings = self.rope_emb(latents, position_ids)

        # label embedding
        label_emb = self.label_indices_embedding(label_idx).unsqueeze(1)  # [B, 1, hidden_size]

        # inference
        latents = self.diffusion_model.sample_from_scratch(
            latents,
            cond=label_emb,
            position_embeddings=position_embeddings,
            num_timesteps=num_timesteps,
            progress_bar=progress_bar,
        )

        # output imgs
        imgs = self._unpack_latent(latents, height // self.patch_size, width // self.patch_size)
        imgs = self._img_denormalize(imgs)
        imgs = torch.clamp(imgs, 0.0, 1.0)

        return imgs

# %%
if is_jupyter():
    test_config = Config(
        hidden_size=128,
        depth=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_dropout=0.0,
        intermediate_size=128 * 2,
        latent_size=1 * 2 * 2,
        #
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        #
        time_embed_dim=32,
        time_embed_theta=10000.0,
        #
        initializer_range=0.02,
        register_token_num=4,
    )

    test_model = Diffusion2DModelForPicGenerating(test_config, patch_size=2)
    test_imgs, label_idx = next(iter(train_loader))
    batch_size = test_imgs.size(0)
    
    test_out = test_model.get_train_loss(
        test_imgs,
        label_idx,
        torch.randint(0, 1000, (batch_size,)),
    )
    print(test_out)

    test_latents = torch.randn(12, 1, 28, 28)
    test_out = test_model.sample_from_scratch(
        label_idx=torch.arange(12),
        height=28,
        width=28,
        latents=test_latents,
        num_timesteps=10,
    )
    print(test_out.shape)

# %% [markdown]
# ## Lightning

# %%
from pl_utils import BaseModule
from torchvision.utils import make_grid, save_image
import os


class LightningModel(BaseModule):

    def __init__(
        self,
        model: Diffusion2DModelForPicGenerating,
        learning_rate_config,
        training_config,
    ):
        super().__init__(model, learning_rate_config, training_config)

        self.model.apply(self._init_weight)

    # def forward(self, batch_size=1, height=28, width=28, class_idx=0, num_timesteps=10):
    #     return self.model(batch_size, height, width, class_idx, num_timesteps)

    def training_step(self, batch, batch_idx):
        samples, class_idx = batch
        batch_size = samples.size(0)

        timesteps_idx = torch.normal(0, 1, (batch_size,), device=samples.device).sigmoid()
        timesteps_idx = (timesteps_idx * (1000 - 1)).long()
        # TODO 上面这个采样使用的是 logit_normal 方案
        # https://github.com/huggingface/diffusers/blob/896fb6d8d7c10001eb2a92568be7b4bd3d5ddea3/examples/dreambooth/train_dreambooth_lora_sd3.py#L1490
        # 可以试试其他的。例如 seed-vc 使用的是均匀采样（来源于 CFM 类）

        loss = self.model.get_train_loss(
            samples,
            label_idx=class_idx,
            timestep_idx=timesteps_idx,
        )
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
        self.log('train/grad_norm', grad_norm.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        if self.logger.log_dir is None:
            return

        label_idx = torch.arange(12, device=self.device)
        if not hasattr(self, "tmp_latents"):
            self.tmp_latents = torch.randn(1, 1, 28, 28).repeat(12, 1, 1, 1)

        with torch.inference_mode():
            samples = self.model.sample_from_scratch(
                label_idx=label_idx,
                height=28,
                width=28,
                latents=self.tmp_latents.to(device=self.device),
                num_timesteps=50,
                progress_bar=False,
            )

        images = (samples.clamp(0, 1) * 255).to(torch.uint8)
        grid = make_grid(images, nrow=4, pad_value=255, value_range=(0, 255))

        log_dir = self.logger.log_dir
        os.makedirs(f"{log_dir}/images", exist_ok=True)
        save_image(grid / 255, f'{log_dir}/images/epoch_{self.current_epoch}.png')

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
    # lr_warmup_steps=0.,
    lr_initial=4e-4,
    lr_max=4e-4,
    lr_end=4e-4,
)

training_config = TrainingConfig(
    optimizer='adamw',
    optimizer_args={
        'betas': (0.9, 0.95),
        'weight_decay': 1e-2,
        "fused": True,
    },
    excluded_from_weight_decay=["bias", "norm", "embed", "scale"],
)

config = Config(
    hidden_size=128,
    depth=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    attention_dropout=0.0,
    intermediate_size=128 * 3,
    latent_size=1 * 4 * 4,
    #
    rms_norm_eps=1e-6,
    rope_theta=1000.0,
    #
    time_embed_dim=256,
    time_embed_theta=10000.0,
    #
    initializer_range=0.02,
    register_token_num=4,
)

# %% [markdown]
# ### 实例化

# %%
model = Diffusion2DModelForPicGenerating(config, patch_size=4)

pl_model = LightningModel(
    model,
    learning_rate_config,
    training_config,
)

model.compile()

# %% [markdown]
# ## 正式训练

# %%
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pl_utils.lightning import format_next_version_name
from lightning.pytorch.strategies import DDPStrategy

name = "最终收尾"
logger = TensorBoardLogger(save_dir="./", version=format_next_version_name(name))

checkpoint_callback = ModelCheckpoint(
    save_top_k=0,
    # monitor="val/reconstruct_loss",
    # mode="min",
    # dirpath=r"/mnt/e/vae_models",
    save_weights_only=True,
    # save_last=True,
)

trainer = L.Trainer(
    logger=logger,
    accelerator='gpu',
    max_epochs=100,
    # strategy=DDPStrategy(find_unused_parameters=False),
    precision='bf16-mixed',
    # precision='32',
    #
    log_every_n_steps=4,
    default_root_dir="./",
    #
    # check_val_every_n_epoch=4,
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0,
    # fast_dev_run=True,
)

trainer.fit(pl_model, train_loader)

# %% [markdown]
# ## 提前退出

# %%
import sys
from IPython import get_ipython


# 如果是脚本而不是jupyter notebook，此时就该退出了
try:
    shell = get_ipython()
    if shell is None:
        sys.exit()
except:
    sys.exit()


