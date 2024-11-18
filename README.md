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

To start the training process with defined hyperparameters, execute `launcher.sh`.
You can specify the number of GPUs to use as an argument: `./launcher.sh 8`

If you are using Windows, you can use `docker_launcher.ps1` instead, in which case Dockerfile

### **Implementation Details**

This implementation focuses on demonstrating how normalization can be applied to Vision Transformers. The code is designed to be clear and educational rather than optimized for production use.

Key features:
- Standard PyTorch attention mechanisms
- Support for various image datasets (ImageNet, CIFAR)
- Configurable model size and training parameters
- Distributed training support

The code is kept simple and readable to serve as a reference implementation for normalized Vision Transformers.

---
