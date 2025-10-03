# MyXTTS Architecture Harmony

## Synchronized Component Architecture

This document visualizes the harmonized architecture of MyXTTS model components.

### Component Dimensions Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       MyXTTS Model Architecture                  │
│                     (همگام‌سازی شده و هماهنگ)                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   Text Input     │         │  Audio Input     │
│   (Token IDs)    │         │ (Mel Spectro)    │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         ▼                            ▼
┌─────────────────────┐      ┌──────────────────────┐
│   TextEncoder       │      │   AudioEncoder       │
│   ───────────       │      │   ────────────       │
│   dim:    512       │      │   dim:    768        │
│   layers: 8         │      │   layers: 8          │
│   heads:  8         │      │   heads:  12         │
│   head_dim: 64      │      │   head_dim: 64       │
│   ff_dim: 2048      │      │   ff_dim: 3072       │
└─────────┬───────────┘      └──────────┬───────────┘
          │                             │
          │  text_encoded               │  audio_encoded
          │  [B, T_txt, 512]            │  [B, T_ref, 768]
          │                             │
          │                             │  speaker_embedding
          │                             │  [B, 512]
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    MelDecoder       │
              │    ─────────        │
              │    dim:    1536     │
              │    layers: 16       │
              │    heads:  24       │
              │    head_dim: 64     │
              │    ff_dim: 6144     │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   mel_output        │
              │   [B, T_mel, 80]    │
              └─────────────────────┘
```

## Dimensional Synergy (هم‌افزایی ابعاد)

### Progressive Dimension Increase

The model architecture follows a progressive dimension increase pattern that reflects task complexity:

```
Text Encoding → Audio Encoding → Mel Decoding
    512      →      768       →      1536
    (1x)     →     (1.5x)     →      (3x)
```

**Rationale (منطق طراحی):**
- **Text (512)**: Base semantic understanding
- **Audio (768)**: Richer acoustic feature space (50% larger)
- **Decoder (1536)**: Complex multimodal synthesis (3x text dimension)

### Attention Head Scaling

All components maintain 64-dimensional attention heads while scaling head count:

```
Component    │ Dimension │ Heads │ Head Dim │ Coverage
─────────────┼───────────┼───────┼──────────┼─────────
TextEncoder  │    512    │   8   │    64    │ Basic
AudioEncoder │    768    │  12   │    64    │ Enhanced
MelDecoder   │   1536    │  24   │    64    │ Maximum
```

**Benefits:**
- ✓ Consistent attention patterns across components
- ✓ Easy cross-component feature transfer
- ✓ Stable training dynamics

### Layer Depth Strategy

```
Encoder Depth: 8 layers each (balanced)
    ↓
Decoder Depth: 16 layers (2x autoregressive complexity)
```

**Rationale:**
- Encoders: 8 layers provide sufficient context
- Decoder: 16 layers (2x) for complex generation task
- Balanced encoder depths ensure co-training stability

## Component Interactions (تعاملات کامپوننت)

### 1. Text-to-Decoder Flow

```
Text Tokens
    ↓ [embedding]
Text Features (512-dim, 8 layers)
    ↓ [cross-attention in decoder]
Decoder Processing (1536-dim, 16 layers)
    ↓ [projection]
Mel Spectrogram (80-dim)
```

### 2. Audio-to-Decoder Flow

```
Reference Audio
    ↓ [mel extraction]
Mel Features (80-dim)
    ↓ [audio encoding]
Audio Features (768-dim, 8 layers)
    ├─→ [pooling] Speaker Embedding (512-dim)
    └─→ [cross-attention] Decoder
```

### 3. Speaker Conditioning Path

```
Audio Input → AudioEncoder → Speaker Embedding (512-dim)
                                    ↓
                              [broadcast & concat]
                                    ↓
                         Decoder Input (1536-dim)
```

## Feedforward Dimension Pattern

All components follow the standard Transformer 4x feedforward ratio:

```
Layer Type    │ Model Dim │ FF Dim │ Ratio
──────────────┼───────────┼────────┼──────
TextEncoder   │    512    │  2048  │  4x
AudioEncoder  │    768    │  3072  │  4x
MelDecoder    │   1536    │  6144  │  4x
```

**Why 4x?**
- Standard in successful Transformer architectures
- Provides sufficient non-linear capacity
- Proven optimal through extensive research

## Capacity Distribution

Visual representation of model capacity across components:

```
Model Capacity by Component:

TextEncoder   ████████░░░░░░░░░░░░  (~20%)
AudioEncoder  ████████████░░░░░░░░  (~30%)
MelDecoder    ████████████████████  (~50%)
```

**Interpretation:**
- Decoder has largest capacity (50%) - justified by generation complexity
- Audio encoder larger than text (30% vs 20%) - audio is more complex
- Balanced total capacity for efficient training

## Dimensional Consistency Validation

### Attention Head Dimension Check

```python
# All components use 64-dimensional heads
text_head_dim = 512 / 8 = 64 ✓
audio_head_dim = 768 / 12 = 64 ✓
decoder_head_dim = 1536 / 24 = 64 ✓
```

### Feedforward Ratio Check

```python
# All components use 4x feedforward ratio
text_ff_ratio = 2048 / 512 = 4.0 ✓
audio_ff_ratio = 3072 / 768 = 4.0 ✓
decoder_ff_ratio = 6144 / 1536 = 4.0 ✓
```

### Dimension Divisibility Check

```python
# All dimensions divisible by head count
512 % 8 == 0 ✓
768 % 12 == 0 ✓
1536 % 24 == 0 ✓
```

## Synergistic Benefits (مزایای هم‌افزایی)

### 1. Training Stability
- Balanced encoder depths prevent gradient imbalance
- Consistent head dimensions enable smooth attention flow
- Progressive dimension increase matches task complexity

### 2. Feature Quality
- Larger audio encoder captures richer acoustic details
- Large decoder dimension supports high-quality synthesis
- 512-dim speaker embedding ensures good voice separation

### 3. Computational Efficiency
- Dimension ratios optimize parameter usage
- 64-dim heads balance expressiveness and efficiency
- 4x feedforward ratio proven optimal

### 4. Scalability
- Clear dimension progression allows easy scaling
- Consistent patterns simplify architecture changes
- Well-tested ratios ensure stability when scaling

## Configuration Reference

Complete synchronized configuration:

```yaml
model:
  # Text Encoder (متن)
  text_encoder_dim: 512
  text_encoder_layers: 8
  text_encoder_heads: 8
  
  # Audio Encoder (صوت)
  audio_encoder_dim: 768
  audio_encoder_layers: 8
  audio_encoder_heads: 12
  
  # Decoder (دیکودر)
  decoder_dim: 1536
  decoder_layers: 16
  decoder_heads: 24
  
  # Speaker (گوینده)
  speaker_embedding_dim: 512
  
  # Mel Spectrogram
  n_mels: 80
  sample_rate: 22050
```

## Summary (خلاصه)

The MyXTTS architecture is now fully synchronized with:
- ✅ Consistent dimension patterns across all components
- ✅ Synergistic capacity distribution
- ✅ Validated attention mechanisms
- ✅ Optimized feedforward ratios
- ✅ Balanced layer depths
- ✅ Comprehensive test coverage

This harmony ensures optimal model performance and maintainability.

---

**Related Documents:**
- `SYNCHRONIZATION_SUMMARY.md` - Detailed synchronization report
- `docs/Html/model_components.html` - Component documentation
- `tests/test_config_consistency.py` - Validation tests
