# Fine-tuning Plan for Small Real-ESRGAN Model

**Model:** `small-realesr-animevideox2v3` (32-feat, 32-conv)

**Objective:** Fine-tune the compressed model to recover quality lost during weight transfer from the original 64-feat/16-conv model.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Configuration](#3-configuration)
4. [Training Execution](#4-training-execution)
5. [Validation & Monitoring](#5-validation--monitoring)
6. [Timeline & Milestones](#6-timeline--milestones)
7. [Success Criteria](#7-success-criteria)

---

## 1. Overview

### 1.1 Training Strategy

The fine-tuning process follows a two-phase approach:

**Phase 1: Quality Recovery & Generic SR Learning**
- Train on diverse dataset (DF2K + subset of Flickr2K/OST)
- Adapt transferred weights to compensate for capacity reduction (64→32 features)
- Learn robust generic super-resolution features
- Create a strong foundation model for future tasks

**Phase 2: Domain Specialization for IR (Planned)**
- Fine-tune Phase 1 checkpoint on IR-specific dataset
- Adapt to infrared image characteristics
- Optimize for real-world IR deployment

**Important Note on Dataset Size:**
Given the small model capacity (0.30M parameters), training on the full 13,774 images risks overfitting or underfitting. We start with a **curated subset of 3,000-5,000 images** and monitor validation metrics closely.

### 1.2 Key Differences from Original Training

- **Starting point:** Pre-transferred weights from original model (not random initialization)
- **Reduced iterations:** 200k instead of 400k (due to pre-trained initialization)
- **Larger batch size:** 16 instead of 12 (smaller model allows more samples per batch)
- **Architecture:** SRVGGNetCompact (32-feat, 32-conv) vs RRDBNet (64-feat, 23-block)

---

## 2. Prerequisites

### 2.1 Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- RAM: 32GB
- Storage: 100GB free space (for datasets and checkpoints)

**Recommended:**
- GPU: 4× RTX 3090 or 4090
- RAM: 64GB
- Storage: 200GB NVMe SSD

### 2.2 Software Dependencies

```bash
# Already installed in environment
- Python 3.8+
- PyTorch 1.12+
- basicsr
- Real-ESRGAN repository
```

### 2.3 Dataset Preparation

#### Download Datasets

```bash
# Create dataset directory
mkdir -p datasets/DF2K

# Download DIV2K (800 images, ~4.5GB)
cd datasets/DF2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

# Download Flickr2K (2650 images, ~15GB)
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
tar -xvf Flickr2K.tar
mv Flickr2K/Flickr2K_HR ./
rm -rf Flickr2K

# Download OST dataset (10,324 images, OutdoorSceneTrain)
# Manual download required from Google Drive:
# https://drive.google.com/drive/folders/16PIViLkv4WsXk4fV1gDHvEtQxdMq6nfY
# Download 3 zip files to datasets directory, then extract:
cd datasets
bash extract_ost.sh
```

**Expected result:** `DF2K/OST/` containing ~10,324 images from 7 categories (animal, building, grass, mountain, plant, sky, water).

#### Dataset Subset Selection

To avoid overfitting with the small model (0.30M params), we use a curated subset:

```bash
# Run automated subset creation script
cd datasets
bash create_subset.sh

# This will create:
# - DF2K_subset/DIV2K_train_HR: 800 images (all)
# - DF2K_subset/Flickr2K_HR: 1,500 images (random sample)
# - DF2K_subset/OST: 1,000 images (random sample)
# Total: ~3,300 images
```

**Note:** If you need to recreate the subset with different random samples:
```bash
rm -rf DF2K_subset
bash create_subset.sh
```

**Alternative:** If validation shows good convergence, gradually increase dataset size in subsequent training iterations.

#### Generate Meta Info

```bash
# Generate meta_info.txt for subset
cd ..  # Back to project root
python scripts/generate_meta_info.py \
  --input datasets/DF2K_subset/DIV2K_train_HR datasets/DF2K_subset/Flickr2K_HR datasets/DF2K_subset/OST \
  --root datasets/DF2K_subset datasets/DF2K_subset datasets/DF2K_subset \
  --meta_info datasets/DF2K_subset/meta_info/meta_info_DF2K_subset.txt
```

### 2.4 Pretrained Model

The starting point is the weight-transferred model:

```bash
# Model already prepared
weights/small-realesr-animevideox2v3.pth
```

This model was created by:
1. Selecting top-32 channels from original 64-feat model based on L2 norm
2. Duplicating 16 convolutional blocks to create 32 blocks
3. Maintaining sorted channel indices to prevent color artifacts

---

## 3. Configuration

### 3.1 Create Training Config

Create file: `options/finetune_small_realesr_animevideox2v3.yml`

```yaml
# general settings
name: finetune_small_realesr_animevideox2v3_200k
model_type: RealESRGANModel
scale: 2
num_gpu: auto
manual_seed: 0

# USM the ground-truth
l1_gt_usm: True
percep_gt_usm: True
gan_gt_usm: False

# first degradation process
resize_prob: [0.2, 0.7, 0.1]  # up, down, keep
resize_range: [0.15, 1.5]
gaussian_noise_prob: 0.5
noise_range: [1, 30]
poisson_scale_range: [0.05, 3]
gray_noise_prob: 0.4
jpeg_range: [30, 95]

# second degradation process
second_blur_prob: 0.8
resize_prob2: [0.3, 0.4, 0.3]  # up, down, keep
resize_range2: [0.3, 1.2]
gaussian_noise_prob2: 0.5
noise_range2: [1, 25]
poisson_scale_range2: [0.05, 2.5]
gray_noise_prob2: 0.4
jpeg_range2: [30, 95]

gt_size: 256
queue_size: 180

# dataset and data loader settings
datasets:
  train:
    name: DF2K
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K
    meta_info: datasets/DF2K/meta_info/meta_info_DF2K.txt
    io_backend:
      type: disk

    blur_kernel_size: 21
    kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    betag_range: [0.5, 4]
    betap_range: [1, 2]

    blur_kernel_size2: 21
    kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob2: 0.1
    blur_sigma2: [0.2, 1.5]
    betag_range2: [0.5, 4]
    betap_range2: [1, 2]

    final_sinc_prob: 0.8

    gt_size: 256
    use_hflip: True
    use_rot: False

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 5
    batch_size_per_gpu: 16  # Increased from 12
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  # Validation settings
  val:
    name: validation
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/GTmod12
    dataroot_lq: datasets/Set5/LRbicx2
    io_backend:
      type: disk

# network structures
network_g:
  type: SRVGGNetCompact
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 32
  num_conv: 32
  upscale: 2
  act_type: 'prelu'

network_d:
  type: UNetDiscriminatorSN
  num_in_ch: 3
  num_feat: 64
  skip_connection: True

# path
path:
  pretrain_network_g: weights/small-realesr-animevideox2v3.pth
  param_key_g: params_ema
  strict_load_g: false  # Set to false due to architecture difference
  pretrain_network_d: ~  # Start discriminator from scratch
  param_key_d: params
  strict_load_d: false
  resume_state: ~

# training settings
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  optim_d:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [200000]
    gamma: 0.5

  total_iter: 200000  # Reduced from 400k
  warmup_iter: -1  # no warm up

  # losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

  # perceptual loss (content and style losses)
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1
      'conv4_4': 1
      'conv5_4': 1
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: !!float 1.0
    style_weight: 0
    range_norm: false
    criterion: l1

  # gan loss
  gan_opt:
    type: GANLoss
    gan_type: vanilla
    real_label_val: 1.0
    fake_label_val: 0.0
    loss_weight: !!float 1e-1

  net_d_iters: 1
  net_d_init_iters: 0

# validation settings
val:
  val_freq: !!float 5e3
  save_img: True

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false

# logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500
```

### 3.2 Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `total_iter` | 200,000 | Half of original due to pre-trained initialization |
| `batch_size_per_gpu` | 16 | Larger batch possible with smaller model |
| `learning_rate` | 1e-4 | Standard rate for fine-tuning |
| `scale` | 2 | 2× upsampling (not 4×) |
| `num_feat` | 32 | Compressed from 64 |
| `num_conv` | 32 | Extended from 16 |
| `val_freq` | 5,000 | Validate every 5k iterations |

---

## 4. Training Execution

### 4.1 Debug Mode (Test First)

Always test in debug mode before full training:

```bash
python realesrgan/train.py \
  -opt options/finetune_small_realesr_animevideox2v3.yml \
  --debug
```

This will:
- Run for a few iterations only
- Verify data loading works
- Check GPU memory usage
- Validate config file syntax

### 4.2 Single GPU Training

```bash
python realesrgan/train.py \
  -opt options/finetune_small_realesr_animevideox2v3.yml \
  --auto_resume
```

**Estimated time:** 24-30 hours (RTX 3090)

### 4.3 Multi-GPU Training (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=4321 \
  realesrgan/train.py \
  -opt options/finetune_small_realesr_animevideox2v3.yml \
  --launcher pytorch \
  --auto_resume
```

**Estimated time:** 6-8 hours (4× RTX 3090)

### 4.4 Resume from Checkpoint

Training automatically resumes if interrupted (due to `--auto_resume` flag). Checkpoints are saved in:

```
experiments/finetune_small_realesr_animevideox2v3_200k/models/
```

Manual resume:

```yaml
# In config file, modify:
path:
  resume_state: experiments/finetune_small_realesr_animevideox2v3_200k/training_states/50000.state
```

---

## 5. Validation & Monitoring

### 5.1 Real-time Monitoring

**TensorBoard:**
```bash
tensorboard --logdir experiments/finetune_small_realesr_animevideox2v3_200k/tb_logger
```

Monitor:
- Loss curves (L1, perceptual, GAN, discriminator)
- PSNR/SSIM trends
- Learning rate schedule

### 5.2 Validation Metrics

Validation runs every 5,000 iterations on Set5 dataset:

| Metric | Description | Target |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | ≥ 28 dB |
| **SSIM** | Structural Similarity Index | ≥ 0.88 |

### 5.3 Visual Quality Inspection

Check saved validation images:
```
experiments/finetune_small_realesr_animevideox2v3_200k/visualization/
```

Look for:
- ✅ Sharp edges and textures
- ✅ Natural colors (no rainbow artifacts)
- ✅ Consistent brightness
- ❌ Blocking artifacts
- ❌ Over-smoothing

### 5.4 Checkpoint Selection

Checkpoints saved every 5,000 iterations:
```
experiments/finetune_small_realesr_animevideox2v3_200k/models/
├── net_g_5000.pth
├── net_g_10000.pth
├── ...
├── net_g_200000.pth
└── net_g_latest.pth
```

Best checkpoint selection criteria:
1. Highest PSNR on validation set
2. Visual quality inspection (no artifacts)
3. Typically around 150k-200k iterations

---

## 6. Timeline & Milestones

### 6.1 Preparation Phase (Day 0)

```
├─ Dataset download:           2-3 hours
├─ Meta info generation:       15 minutes
├─ Config file creation:       30 minutes
├─ Debug mode testing:         1 hour
└─ Total:                      4-5 hours
```

### 6.2 Training Phase (Day 1-2)

**Single GPU (RTX 3090):**
```
├─ 0-50k iterations:           6 hours    (Initial convergence)
├─ 50k-100k iterations:        6 hours    (Quality improvement)
├─ 100k-150k iterations:       6 hours    (Fine adjustments)
├─ 150k-200k iterations:       6 hours    (Final polish)
└─ Total:                      24 hours
```

**4× GPU (RTX 3090):**
```
├─ 0-50k iterations:           1.5 hours
├─ 50k-100k iterations:        1.5 hours
├─ 100k-150k iterations:       1.5 hours
├─ 150k-200k iterations:       1.5 hours
└─ Total:                      6-7 hours
```

### 6.3 Evaluation Phase (Day 3)

```
├─ Full validation (Set5, Set14, DIV2K val):  2 hours
├─ Visual quality inspection:                 2 hours
├─ Quantitative comparison with original:     1 hour
├─ Export best checkpoint:                    30 minutes
└─ Total:                                     5-6 hours
```

### 6.4 Expected Training Dynamics

| Iteration Range | Expected Behavior |
|----------------|-------------------|
| 0-20k | Rapid PSNR increase, loss decrease |
| 20k-50k | Steady improvement, GAN loss stabilizes |
| 50k-100k | Gradual quality refinement |
| 100k-150k | Fine-tuning, minor improvements |
| 150k-200k | Convergence, minimal changes |

---

## 7. Success Criteria

### 7.1 Quantitative Metrics

| Metric | Target | Baseline (Original) | Status |
|--------|--------|---------------------|--------|
| **PSNR (Set5)** | ≥ Original - 2.0 dB | TBD | Pending |
| **SSIM (Set5)** | ≥ 0.88 | TBD | Pending |
| **PSNR (DIV2K val)** | ≥ Original - 2.0 dB | TBD | Pending |
| **Training convergence** | Stable losses | N/A | Pending |

### 7.2 Qualitative Assessment

**Visual Quality Checklist:**
- ✅ No color artifacts (rainbow patterns)
- ✅ Sharp edges without over-sharpening
- ✅ Natural texture preservation
- ✅ Consistent across different image types
- ✅ No blocking or ringing artifacts

### 7.3 Performance Verification

| Metric | Target | Status |
|--------|--------|--------|
| **Inference time (QCS8550)** | ≤ 33ms | ✅ Already verified (32ms) |
| **Model size** | ~0.30M params | ✅ Already verified |
| **Memory footprint** | < 50 MB | Pending |

### 7.4 Failure Scenarios & Mitigations

| Issue | Detection | Mitigation |
|-------|-----------|------------|
| **Non-converging losses** | Losses oscillate or increase | Reduce learning rate, check data |
| **Color artifacts** | Rainbow patterns in output | Verify channel indices sorting, reduce GAN weight |
| **Mode collapse (GAN)** | Discriminator loss → 0 | Adjust GAN loss weight, increase D training frequency |
| **Over-smoothing** | Low perceptual quality despite high PSNR | Increase perceptual loss weight |
| **Training too slow** | < 2 iter/s | Reduce batch size, check I/O bottleneck |

---

## 8. Post-Training

### 8.1 Model Export

```bash
# Copy best checkpoint to weights directory
cp experiments/finetune_small_realesr_animevideox2v3_200k/models/net_g_150000.pth \
   weights/small-realesr-animevideox2v3-finetuned.pth
```

### 8.2 Comprehensive Evaluation

Run full benchmark:
```bash
python inference_realesrgan.py \
  -n small-realesr-animevideox2v3-finetuned \
  -i datasets/Set5/LRbicx2 \
  -o results/Set5 \
  --outscale 2
```

### 8.3 Comparison Report

Generate comparison:
- Before fine-tuning (transferred weights only)
- After fine-tuning
- Original model (64-feat, 16-conv)

Metrics:
- PSNR, SSIM, LPIPS
- Inference time
- Visual samples

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: CUDA out of memory**
```bash
# Solution: Reduce batch size
batch_size_per_gpu: 8  # or 4
```

**Issue: "strict_load_g" error**
```yaml
# Solution: Set to false when architecture differs
strict_load_g: false
```

**Issue: Validation dataset not found**
```bash
# Download Set5 benchmark
mkdir -p datasets/Set5
wget http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html
# Extract to datasets/Set5/{GTmod12, LRbicx2}
```

**Issue: Slow training (< 1 iter/s)**
```yaml
# Check I/O bottleneck
num_worker_per_gpu: 8  # Increase workers
prefetch_mode: cuda    # Enable CUDA prefetch
```

### 9.2 Debugging Commands

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor training log
tail -f experiments/finetune_small_realesr_animevideox2v3_200k/train_*.log

# Test single batch
python realesrgan/train.py -opt options/finetune_small_realesr_animevideox2v3.yml --debug
```

---

## 10. Next Steps

After successful Phase 1 fine-tuning:

1. **Evaluate on generic images** - Verify quality recovery
2. **Plan Phase 2 (IR specialization)** - Detailed dataset collection and training strategy
3. **Optimize for deployment** - ONNX export, QNN compilation
4. **Final integration** - Deploy to QCS8550 device

---

**Last Updated:** 2025-10-07
**Author:** Training Team
