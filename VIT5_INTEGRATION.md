# ViT-5 集成到 DINOv3（方案 C：最小开关式改动）

把 [ViT-5 论文](https://arxiv.org/abs/2602.08071) (`models_v2.py`) 的两项 DINOv3 原生不支持的特性——**qk_norm** 与 **register tokens 上的独立 RoPE**——以**开关形式**合入 DINOv3 的 ViT，不新增模型文件、不改动 FSDP/meta-device/iBOT mask 路径。

其它 ViT-5 改动（RMSNorm、SwiGLU、RoPE、无 bias、无 layer_scale）DINOv3 本就支持，通过 config 覆盖即可。

## 改动清单（6 个文件）

### 1. [dinov3/layers/attention.py](dinov3/layers/attention.py)

- 新增 `from dinov3.layers.rms_norm import RMSNorm`。
- `SelfAttention.__init__` 新增 `qk_norm: bool = False`；启用时构建 `self.q_norm = RMSNorm(head_dim)` / `self.k_norm = RMSNorm(head_dim)`。
- `compute_attention` 中，在 `apply_rope` **之前**对 q、k 各过一次 RMSNorm（dtype 回转为原始 q/k dtype），与 `models_v2.py` 顺序一致。

### 2. [dinov3/layers/block.py](dinov3/layers/block.py)

- `SelfAttentionBlock.__init__` 新增 `qk_norm: bool = False`，透传给 `attn_class(...)`。

### 3. [dinov3/models/vision_transformer.py](dinov3/models/vision_transformer.py)

- `DinoVisionTransformer.__init__` 新增三个 kwargs：
  - `qk_norm: bool = False`
  - `register_rope_enabled: bool = False`
  - `register_rope_theta: float = 100.0`
- 启用 `register_rope_enabled` 且 `n_storage_tokens > 0` 时：
  - 计算 `reg_side = int(round(sqrt(n_storage_tokens)))`，`assert reg_side**2 == n_storage_tokens`（必须是平方数）。
  - 构建第二个 `RopePositionEmbedding`，`base=register_rope_theta`（默认 100），其它 rope 参数复用 patch rope 的 `normalize_coords` 和 `dtype`。
- 新 helper `_build_rope(H, W)`：
  - 不启用 register rope → 返回 patch rope `(sin_p, cos_p)`（与旧行为一致）。
  - 启用 → 拼接 `sin = cat(sin_r, sin_p, dim=-2)`、`cos = cat(cos_r, cos_p, dim=-2)`。
- `forward_features_list` 与 `_get_intermediate_layers_not_chunked` 把 `self.rope_embed(H=H, W=W)` 替换为 `self._build_rope(H=H, W=W)`。
- Token 顺序保持 DINOv3 约定 `[cls, storage, patch]`；`apply_rope` 使用 `prefix = N - sin.shape[-2]`，因此拼接后 sin 覆盖 storage+patch，CLS 自动不旋转。
- `init_weights()` 新增 `register_rope_embed._init_weights()`（若存在）。
- blocks 构建时透传 `qk_norm=qk_norm`。

### 4. [dinov3/models/__init__.py](dinov3/models/__init__.py)

- `build_model` 的 `vit_kwargs` 新增：
  ```python
  qk_norm=args.qk_norm,
  register_rope_enabled=args.register_rope_enabled,
  register_rope_theta=args.register_rope_theta,
  ```

### 5. [dinov3/configs/ssl_default_config.yaml](dinov3/configs/ssl_default_config.yaml)

在 `student:` 块下新增默认值（全 false，保持现有行为）：

```yaml
qk_norm: false                    # ViT-5: RMSNorm on q,k (Llama-style)
register_rope_enabled: false      # ViT-5: apply 2D RoPE to register tokens (requires square n_storage_tokens)
register_rope_theta: 100.0        # ViT-5: separate RoPE base for register tokens
```

### 6. 新增两个训练预设

`models_v2.py` 里 ViT-5 有两种并列变体，分别对应两个预设：

**[dinov3/configs/train/vit5_base_im1k.yaml](dinov3/configs/train/vit5_base_im1k.yaml)** — 对应 `vit5_base_swi`
`ffn_layer=swiglu / ffn_ratio=4.0 / layerscale=null`，其余 ViT-5 开关全开。

> ⚠️ **SwiGLU 的 `ffn_ratio` 取值陷阱**：DINOv3 的 `SwiGLUFFN` 内部会做 `hidden×2/3` 收缩（Llama 惯例）。设 `ffn_ratio=4.0` 对应实际隐层 = int(embed·4·2/3) = `8/3·embed`，与 `models_v2.py::vit5_base_swi` 的 `mlp_ratio=2.667` 等价。**不能直接写 2.667**，否则实际隐层只剩 `16/9·embed`，约为 ViT-5 原配的 67%。

**[dinov3/configs/train/vit5_base_mlp_im1k.yaml](dinov3/configs/train/vit5_base_mlp_im1k.yaml)** — 对应 `vit5_base`
`ffn_layer=mlp / ffn_ratio=4.0 / layerscale=1e-4`（LayerScale 启用、初值与 `models_v2.py` 对齐），其余 ViT-5 开关（RMSNorm、无 bias、4 registers + RoPE、qk_norm）同样开启。

共同设置：`arch=vit_base / drop_path_rate=0.1 / qkv_bias=ffn_bias=proj_bias=false / n_storage_tokens=4 / qk_norm=true / register_rope_enabled=true / register_rope_theta=10.0`。
其它训练/优化/评估超参与官方 vitl 预设保持一致（`sqrt_wrt_1024` 自动按总 batch 缩放 lr）。

**关于 `register_rope_theta` 取值（⚠️ 需要 ablation）**：

论文 Table 3 要求 register 的 RoPE 频率**显著高于** patch token。查 [rope_position_encoding.py:112](dinov3/layers/rope_position_encoding.py#L112)：`periods = base ** (2·i/D)`，`base` 越大周期越长、频率越低。DINOv3 patch 默认 `base=100`。

两个关键约束：
- **不能用 1.0**：`1.0^x = 1.0`，所有维度坍缩为单频率，破坏 RoPE 的多频结构（这是一个常见陷阱）。
- **不能用 100.0**：等于 patch base → 论文 Table 3 中的 "same freq." 情形，次优。

**不能从 `models_v2.py` 直接换算**：`models_v2.py` 用绝对位置坐标（0…13），DINOv3 用归一化坐标（[-1,+1]），同一 base 数值在两套实现里物理含义不可比，按 1/100 比例外推无数学依据。

预设默认 `register_rope_theta=10.0` 仅作为**经验初猜**（非退化、周期范围 [1, 10] 相对 patch [1, 100] 明显更高频）。**正式训练前建议在 {0.1, 10, 30} 上做短 ablation**，在你的数据规模下找到真正对应论文 "high freq." 的取值。

**关于变体选择**：论文 Table 2 / Table 9 列出的 ViT-5 **default setup** 是 **LayerScale + GeLU MLP**（对应 `vit5_base_mlp_im1k.yaml` / `models_v2.py::vit5_base`），而非 SwiGLU 变体。SwiGLU 版（`vit5_base_im1k.yaml`）是论文探讨的并列选项，与 LayerScale 共用时存在 over-gating 风险。若你要严格复现论文报告的 84.2% ImageNet 数字，优先用 `vit5_base_mlp_im1k.yaml`。

**RMSNorm eps = 1e-6（已对齐）**：`models_v2.py` 里 block RMSNorm 和 q/k RMSNorm 均用 `eps=1e-6`；DINOv3 `RMSNorm` 默认 `eps=1e-5`，不改会有数值偏差。修正：
- `norm_layer_dict["rmsnorm"] = partial(RMSNorm, eps=1e-6)`（vision_transformer.py）。
- `attention.py` 的 `q_norm/k_norm = RMSNorm(head_dim, eps=1e-6)`。

**APE（additive absolute pos embed，新增开关 `student.use_ape`）**：ViT-5 在 RoPE 之外还对 patch tokens 额外加一份可学习 APE（论文 3.4，`models_v2.py` 的 `ape=True` 默认）。DINOv3 原版没有。新增：
- `DinoVisionTransformer` 新增 `use_ape: bool = False`，启用时构建 `pos_embed = nn.Parameter((1, grid², embed_dim))`，`init_weights` 里 `std=0.02` 初始化。
- `prepare_tokens_with_masks` 在 cat 之前对 patch tokens 执行 `x += pos_embed`。
- 多尺度裁剪（global 224 / local 96 → 不同 patch grid）下用 `F.interpolate(..., mode="bicubic")` 将 `pos_embed` 插值到当前 (H,W)，避免 shape 不匹配。
- 两个 ViT-5 预设均开启 `use_ape: true`；`ssl_default_config.yaml` 默认 false，保持原 DINOv3 行为不受影响。

## 使用方式

### 推荐：用预设 yaml（最简洁）

在 `torchrun` 里把 `--config-file` 指向预设，无需再写一堆 `student.*` 覆盖：

```
# SwiGLU 变体（对应 vit5_base_swi）
--config-file dinov3/configs/train/vit5_base_im1k.yaml

# GeLU MLP + LayerScale 变体（对应 vit5_base）
--config-file dinov3/configs/train/vit5_base_mlp_im1k.yaml
```

新增其它 size 变体时拷贝对应 yaml，改 `arch / drop_path_rate` 即可：
- `vit5_small*`：`student.arch=vit_small`。
- `vit5_large*`：`student.arch=vit_large`，`drop_path_rate=0.3`。

### 备选：手写覆盖参数

若不用预设，也可沿用旧 yaml（如 `vitl_im1k_lin834.yaml`）+ 命令行覆盖：

```
student.arch=vit_base student.patch_size=16
student.norm_layer=rmsnorm student.ffn_layer=swiglu student.ffn_ratio=4.0
student.qkv_bias=false student.ffn_bias=false student.proj_bias=false
student.layerscale=null student.n_storage_tokens=4
student.qk_norm=true student.register_rope_enabled=true student.register_rope_theta=10
student.pos_embed_type=rope
```

## 与 ViT-5 特性对照

| ViT-5 特性 | 实现方式 |
|---|---|
| RMSNorm | `student.norm_layer=rmsnorm`（DINOv3 原生） |
| SwiGLU FFN | `student.ffn_layer=swiglu` + `ffn_ratio=4.0`（原生） |
| 2D RoPE on patches | `student.pos_embed_type=rope`（原生） |
| 无 qkv/ffn/proj bias | `qkv_bias/ffn_bias/proj_bias=false`（原生） |
| 无 layer_scale | `student.layerscale=null`（原生） |
| 4 register tokens | `student.n_storage_tokens=4`（原生，token 顺序 `[cls, storage, patch]`） |
| **qk_norm (RMSNorm on q,k)** | **新增 `student.qk_norm=true`** |
| **register tokens 独立高频 RoPE** | **新增 `student.register_rope_enabled=true` + `register_rope_theta=10.0`**（经验初猜，需 ablation；不能用 1.0 会单频退化，不能用 100 与 patch 同频） |

## 兼容性 / 不变性

- 三个新字段默认 false，**未启用时行为与修改前完全一致**，不会影响已在跑的任务或已有 checkpoint 的加载。
- FSDP、meta-device、`init_weights()`、iBOT mask、`get_intermediate_layers`、`compile`、`fp8` 路径均未触及。
- 已通过 `python -m py_compile` 语法检查；运行时验证建议在训练机上跑 1–2 iter。

## 为什么没有直接把 `models_v2.py` 套进来

`models_v2.py` 的 `vit_models` 为 timm 风格接口，缺少 DINOv3 训练器硬依赖的 `prepare_tokens_with_masks` / `forward_features_list` 返回 dict / `mask_token` 参数 / meta-device 初始化等；且其 token 顺序为 `[cls, patch, registers]`，与 DINOv3 的 `[cls, storage, patch]` 不一致。完整包装需要重写 ~300–400 行并重跑 FSDP 兼容性验证，隐性 bug 面大（mask 位置、token 切片、FSDP 分片）。

方案 C 只加两个开关、~80 行改动即可复刻 ViT-5 的两个关键增益（qk_norm、register-RoPE），同时复用 DINOv3 的整套训练基础设施，实为最小且最安全的路径。
