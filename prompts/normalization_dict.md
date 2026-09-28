* # 📖 Neural Architecture Normalization Dictionary

  > **THE GOLDEN RULE:**
  > NEVER change the `raw_text` field. Capture exactly what the author wrote (including preserving LaTeX). Only use these normalized terms inside the `properties` dictionary.

  ---

  ### 1. MODIFIER Class (Stateless Operations)

  | Subtype         | Target Property | Normalized Value (ONNX) | Visual Shorthands (What the paper says) | Notes                                       |
  | :-------------- | :-------------- | :---------------------- | :-------------------------------------- | :------------------------------------------ |
  | `activation`    | `algorithm`     | `Relu`                  | relu, ReLu, RELU                        |                                             |
  | `activation`    | `algorithm`     | `LeakyRelu`             | lrelu, leaky relu, leaky_relu                       |                                             |
  | `activation`    | `algorithm`     | `PRelu`                 | prelu, parametric relu                  |                                             |
  | `activation`    | `algorithm`     | `Gelu`                  | gelu, GELU                              |                                             |
  | `activation`    | `algorithm`     | `Sigmoid`               | sigm, sigmoid, $\sigma$                 |                                             |
  | `activation`    | `algorithm`     | `Tanh`                  | tanh, TanH                              |                                             |
  | `activation`    | `algorithm`     | `Swish`                 | swish, SiLU                             | SiLU is mathematically equivalent to Swish. |
  | `activation`    | `algorithm`     | `HardSwish`             | h-swish, hard swish                     |                                             |
  | `activation`    | `algorithm`     | `Elu`                   | elu, ELU                                |                                             |
  | `activation`    | `algorithm`     | `Selu`                  | selu, SELU                              |                                             |
  | `activation`    | `algorithm`     | `Mish`                  | mish, Mish                              |                                             |
  | `activation`    | `algorithm`     | `Softmax`               | softmax, SM                             |                                             |
  | `activation`    | `algorithm`     | `LogSoftmax`            | log softmax                             |                                             |
  | `normalization` | `algorithm`     | `BatchNormalization`    | bn, batch norm, Batchnorm, batch_norm               |                                             |
  | `normalization` | `algorithm`     | `LayerNormalization`    | ln, layer norm, Layernorm, layer_norm               |                                             |
  | `normalization` | `algorithm`     | `InstanceNormalization` | in, instance norm                       |                                             |
  | `normalization` | `algorithm`     | `GroupNormalization`    | gn, group norm                          |                                             |
  | `normalization` | `algorithm`     | `LRN`                   | lrn, local response norm, local_response_norm                |                                             |
  | `normalization` | `algorithm`     | `RMSNormalization`      | rms norm, RMSNorm                       |                                             |
  | `normalization` | `algorithm`     | `LpNormalization`       | lp norm, L2 norm                        |                                             |
  | `normalization` | `algorithm`     | `AdaIN`                 | adain                                   | Adaptive Instance Norm.                     |
  | `downsample`    | `algorithm`     | `MaxPool`               | max, maxpool, max-pooling               |                                             |
  | `downsample`    | `algorithm`     | `AveragePool`           | avg, average, avgpool                   |                                             |
  | `downsample`    | `algorithm`     | `GlobalAveragePool`     | gap, global avg pool                    |                                             |
  | `downsample`    | `algorithm`     | `GlobalMaxPool`         | gmp, global max pool                    |                                             |
  | `downsample`    | `algorithm`     | `LpPool`                | lp pool, L2 pool                        |                                             |
  | `downsample`    | `algorithm`     | `MaxRoiPool`            | roi pool, roipooling                    |                                             |
  | `downsample`    | `algorithm`     | `PatchMerge`            | patch merge                             | Swin Transformer specific.                  |
  | `upsample`      | `algorithm`     | `ResizeLinear`          | bilinear, bi-linear, resize             |                                             |
  | `upsample`      | `algorithm`     | `ResizeNearest`         | nearest, nn-interpolate                 |                                             |
  | `upsample`      | `algorithm`     | `ResizeCubic`           | bicubic                                 |                                             |
  | `upsample`      | `algorithm`     | `MaxUnpool`             | unpool, max-unpooling                   |                                             |
  | `upsample`      | `algorithm`     | `GridSample`            | grid sample                             |                                             |
  | `upsample`      | `algorithm`     | `PatchExpand`           | patch expand                            | Transformer specific.                       |
  | `reshape`       | `algorithm`     | `Flatten`               | flatten, vectorization                  |                                             |
  | `reshape`       | `algorithm`     | `Slice`                 | crop, clipping, slice                   |                                             |
  | `reshape`       | `algorithm`     | `Split`                 | split, chunk                            |                                             |
  | `reshape`       | `algorithm`     | `Squeeze`               | squeeze                                 | Removes dimensions of size 1.               |
  | `reshape`       | `algorithm`     | `Unsqueeze`             | unsqueeze, expand_dims                  | Adds dimensions of size 1.                  |
  | `reshape`       | `algorithm`     | `Transpose`             | permute, transpose                      |                                             |
  | `reshape`       | `algorithm`     | `Pad`                   | pad, padding                            |                                             |
  | `reshape`       | `algorithm`     | `Tile`                  | tile, repeat                            |                                             |
  | `reshape`       | `algorithm`     | `Expand`                | expand, broadcast                       |                                             |
  | `reshape`       | `algorithm`     | `Gather`                | gather, index                           |                                             |

  ---

  ### 2. JUNCTION Class (Routing Logic)

  | Subtype    | Target Property | Normalized Value (ONNX) | Visual Shorthands (What the paper says) | Notes                             |
  | :--------- | :-------------- | :---------------------- | :-------------------------------------- | :-------------------------------- |
  | `multiply` | `mechanism`     | `Mul`                   | element-wise, $\odot$, $\otimes$        | Element-wise multiplication.      |
  | `multiply` | `mechanism`     | `MatMul`                | dot product, matmul, mm                 | Matrix multiplication.            |
  | `concat`   | `dim`           | `channel`               | channel-wise, depth-wise                | Concatenating along channel axis. |
  | `concat`   | `dim`           | `spatial`               | spatial, width-wise                     | Concatenating along spatial axes. |

  ---

### 3. LAYER & MACRO Classes (Trainable Layers & Blocks)

  | Class   | Subtype     | Target Property     | Normalized Value       | Visual Shorthands               | Notes                                            |
  | :------ | :---------- | :------------------ | :--------------------- | :------------------------------ | :----------------------------------------------- |
  | `LAYER` | `recurrent` | `name` (Global key) | `RNN`                  | rnn, vanilla rnn                |                                                  |
  | `LAYER` | `recurrent` | `name` (Global key) | `LSTM`                 | lstm                            |                                                  |
  | `LAYER` | `recurrent` | `name` (Global key) | `GRU`                  | gru                             |                                                  |
  | `LAYER` | `attention` | `name` (Global key) | `Attention`            | self-attention, cross-attention |                                                  |
  | `LAYER` | `custom`    | `algorithm`         | `DeformConv`           | deformable conv, DCN            | Treated as custom due to dynamic weights.        |
  | `MACRO` | `block`     | `name` (Global key) | `SqueezeAndExcitation` | se block                        |                                                  |
  | `MACRO` | `block`     | `name` (Global key) | `FeaturePyramid`       | fpn                             |                                                  |
  | `MACRO` | `block`     | `name` (Global key) | `CBAM`                 | cbam                            |                                                  |
