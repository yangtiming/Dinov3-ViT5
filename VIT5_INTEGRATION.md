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
`ffn_layer=swiglu / ffn_ratio=2.667 / layerscale=null`，其余 ViT-5 开关全开。

**[dinov3/configs/train/vit5_base_mlp_im1k.yaml](dinov3/configs/train/vit5_base_mlp_im1k.yaml)** — 对应 `vit5_base`
`ffn_layer=mlp / ffn_ratio=4.0 / layerscale=1e-4`（LayerScale 启用、初值与 `models_v2.py` 对齐），其余 ViT-5 开关（RMSNorm、无 bias、4 registers + RoPE、qk_norm）同样开启。

共同设置：`arch=vit_base / drop_path_rate=0.1 / qkv_bias=ffn_bias=proj_bias=false / n_storage_tokens=4 / qk_norm=true / register_rope_enabled=true / register_rope_theta=100`。
其它训练/优化/评估超参与官方 vitl 预设保持一致（`sqrt_wrt_1024` 自动按总 batch 缩放 lr）。

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
student.norm_layer=rmsnorm student.ffn_layer=swiglu student.ffn_ratio=2.667
student.qkv_bias=false student.ffn_bias=false student.proj_bias=false
student.layerscale=null student.n_storage_tokens=4
student.qk_norm=true student.register_rope_enabled=true student.register_rope_theta=100
student.pos_embed_type=rope
```

## 与 ViT-5 特性对照

| ViT-5 特性 | 实现方式 |
|---|---|
| RMSNorm | `student.norm_layer=rmsnorm`（DINOv3 原生） |
| SwiGLU FFN | `student.ffn_layer=swiglu` + `ffn_ratio=2.667`（原生） |
| 2D RoPE on patches | `student.pos_embed_type=rope`（原生） |
| 无 qkv/ffn/proj bias | `qkv_bias/ffn_bias/proj_bias=false`（原生） |
| 无 layer_scale | `student.layerscale=null`（原生） |
| 4 register tokens | `student.n_storage_tokens=4`（原生，token 顺序 `[cls, storage, patch]`） |
| **qk_norm (RMSNorm on q,k)** | **新增 `student.qk_norm=true`** |
| **register tokens 独立 RoPE (θ=100)** | **新增 `student.register_rope_enabled=true` + `register_rope_theta=100`** |

## 兼容性 / 不变性

- 三个新字段默认 false，**未启用时行为与修改前完全一致**，不会影响已在跑的任务或已有 checkpoint 的加载。
- FSDP、meta-device、`init_weights()`、iBOT mask、`get_intermediate_layers`、`compile`、`fp8` 路径均未触及。
- 已通过 `python -m py_compile` 语法检查；运行时验证建议在训练机上跑 1–2 iter。

## 为什么没有直接把 `models_v2.py` 套进来

`models_v2.py` 的 `vit_models` 为 timm 风格接口，缺少 DINOv3 训练器硬依赖的 `prepare_tokens_with_masks` / `forward_features_list` 返回 dict / `mask_token` 参数 / meta-device 初始化等；且其 token 顺序为 `[cls, patch, registers]`，与 DINOv3 的 `[cls, storage, patch]` 不一致。完整包装需要重写 ~300–400 行并重跑 FSDP 兼容性验证，隐性 bug 面大（mask 位置、token 切片、FSDP 分片）。

方案 C 只加两个开关、~80 行改动即可复刻 ViT-5 的两个关键增益（qk_norm、register-RoPE），同时复用 DINOv3 的整套训练基础设施，实为最小且最安全的路径。
