> ## [MIT License](https://opensource.org/license/mit)
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the "Software"),
> to deal in the Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, sublicense,
> and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
> THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

# **nViT: Normalized Vision Transformer**
Highly based on [original nGPT implementation](https://github.com/NVIDIA/ngpt)

**Author**: Tibor Sloboda

## **Project Overview**

This repository provides code for **nViT**, a normalized Vision Transformer implementation. The key modifications focus on applying normalization techniques to the standard ViT architecture.

The main components are:

1. **Modifications**: 
   - `model.py` includes both the **original** and **normalized Vision Transformer** models
   - `train.py` contains the **normalization procedure** for training
   - The architecture follows standard ViT design with added normalization layers

2. **Dependencies**:
   - **PyTorch**
   - Standard Python ML libraries (numpy, etc.)

## **Getting Started**

### **Running the Code**

Install dependencies with `poetry install`

To start the training process with defined hyperparameters, execute `launcher.sh`.
You can specify the number of GPUs to use as an argument: `./launcher.sh 8`

If you are using Windows, you can use `docker_launcher.ps1` instead, in which case you need to build the Docker image first with `build.sh` or `build.ps1` in the `docker` directory. The docker launcher has both Linux (bash) and Windows (PowerShell) variants.

### **Implementation Details**

This implementation focuses on demonstrating how normalization can be applied to Vision Transformers. The code is designed to be clear and educational rather than optimized for production use.

Key features:
- Standard PyTorch attention mechanisms
- Support for various image datasets (ImageNet, CIFAR)
- Configurable model size and training parameters
- Distributed training support

The code is kept simple and readable to serve as a reference implementation for normalized Vision Transformers.

### **Theoretical Background**

The nViT (normalized Vision Transformer) builds upon the principles of nGPT, extending the concept of normalized representations to vision transformers. The key insight is mapping all representations to a unit hypersphere, facilitating meaningful distance computations and token comparisons.

#### **Core Concepts**

1. **Normalized Representations**
   - All feature vectors are normalized to unit length: $\|x\|_2 = 1$
   - Enables direct comparison using dot products as a similarity measure
   - Distances on the hypersphere correspond to semantic relationships

2. **Kohonen Maps as Visual Vocabulary**
   - Unlike language models, vision transformers lack a discrete token vocabulary
   - Kohonen Self-Organizing Maps (SOMs) create a continuous, trainable visual vocabulary
   - Each map node $M_i$ represents a prototype patch on the unit hypersphere
   - Topological preservation ensures similar features are mapped to nearby nodes

3. **Dual-Path Processing**
   - Local patches (8×8): $P_l$ captures fine-grained details
   - Global patches (16×16): $P_g$ captures broader context
   - Cross-attention mechanism: $C(P_l, P_g)$ integrates multi-scale information

#### **Loss Functions**

The training objective combines several losses with adjustable weights:

1. **Main Classification Loss**
   
   $\mathcal{L}_{cls} = \text{CrossEntropy}(f(x), y)$

2. **Quantization Loss**
   - Ensures mapped representations stay close to original patches

   $\mathcal{L}_{quant} = \lambda_{lq} \|K_l(P_l) - P_l\|_2^2 + \lambda_{gq} \|K_g(P_g) - P_g\|_2^2$
   
   where $K_l$ and $K_g$ are local and global Kohonen maps
   $\lambda_{lq} = \lambda_{gq} = 0.1$ (default weights)

3. **Consistency Loss**
   - Aligns local and global representations
   
   $\mathcal{L}_{cons} = \lambda_c (1 - \cos(K_l(P_l), K_g(P_g)))$
   
   where $\cos(a,b)$ is cosine similarity
   $\lambda_c = 0.5$ (default weight)

4. **Smoothness Loss**
   - Maintains topological organization of Kohonen maps
   
   $\mathcal{L}_{smooth} = \lambda_s (\|\nabla K_l\|_2^2 + \|\nabla K_g\|_2^2)$
   
   where $\nabla K$ represents neighborhood differences
   $\lambda_s = 0.1$ (default weight)

5. **Reconstruction Loss**
   - Ensures preservation of spatial information
   
   $\mathcal{L}_{rec} = \lambda_r \|\text{Dec}(\text{Enc}(x)) - x\|_2^2$
   
   $\lambda_r = 0.1$ (default weight)

#### **Combined Training Objective**

The final loss combines all components:

$\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{quant} + \mathcal{L}_{cons} + \mathcal{L}_{smooth} + \mathcal{L}_{rec}$

#### **Training Dynamics**

1. **Kohonen Map Learning**
   - Learning rate follows a cosine schedule with warmup
   - $\alpha(t) = \alpha_{max} \cos(\pi t/T)$ after warmup
   - Nodes are updated using neighborhood function:
     
     $M_i \leftarrow M_i + \alpha(t)h(i,j)(x - M_i)$
     
     where $h(i,j)$ is the neighborhood function

2. **Cross-Attention Integration**
   - Local and global features are combined through cross-attention
   - Attention weights are computed on normalized representations:
     
     $A = \text{softmax}(Q_l K_g^T / \sqrt{d})$
   - Output preserves unit norm through final normalization

3. **Representation Flow**
   ```
   Input Image → Local/Global Patches → Kohonen Mapping →
   Cross-Attention → Transformer Blocks → Classification
   ```

This framework effectively combines the benefits of:
- Normalized representations (from nGPT)
- Continuous visual vocabulary (via Kohonen maps)
- Multi-scale feature processing (through dual-path architecture)
- Self-supervised learning (via reconstruction)

The result is a vision transformer that maintains interpretable, normalized representations throughout its processing pipeline while capturing both local and global visual patterns.

---
