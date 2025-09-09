# Transformer Experiments

This repository contains experiments with transformer language models, implementing a decoder-only architecture from scratch using PyTorch. The project focuses on training efficiency, performance optimization, and model evaluation.

## Project Overview

The main experiment implements a GPT-style decoder-only transformer model trained on Wikipedia data. The implementation includes modern training techniques and optimizations for efficient learning.

## Architecture

### Model Configuration
- **Architecture**: Decoder-only transformer (GPT-style)
- **Model dimension**: 768 (configurable, tested up to 1024)
- **Layers**: 12 (tested with 24-layer configurations)
- **Attention heads**: 8 (tested with 16 heads)
- **MLP dimension**: 4x model dimension
- **Context length**: 512 tokens (configurable up to 1024)
- **Vocabulary**: ~50K tokens (GPT-2 tokenizer)

### Key Features
- **Weight tying**: Shared weights between token embedding and output projection
- **Activation checkpointing**: Memory-efficient gradient computation
- **Scaled Dot-Product Attention**: Using PyTorch's efficient SDPA implementation
- **Layer normalization**: Pre-norm configuration
- **Dropout**: Configurable dropout in attention and MLP layers

## Training Implementation

### Optimization Features
- **Fused AdamW optimizer**: Hardware-optimized for CUDA
- **Gradient accumulation**: Efficient large batch training
- **Warmup cosine scheduler**: Learning rate scheduling with warmup
- **Mixed precision training**: bfloat16 for memory efficiency
- **Optional torch.compile**: PyTorch 2.x compilation for speed

### Training Configuration
- **Token budget**: 200-300M tokens
- **Effective batch size**: 512 tokens
- **Micro batch size**: 32-64 (memory dependent)
- **Learning rate**: Peak LR with warmup and cosine decay
- **Weight decay**: Applied to parameters excluding embeddings and norms

## Data Pipeline

### Dataset
- **Source**: Wikipedia streaming dataset
- **Processing**: Packed sequences with document boundaries marked by EOS tokens
- **Tokenization**: HuggingFace GPT-2 tokenizer adapter
- **Streaming**: Memory-efficient streaming with worker sharding

### Evaluation Metrics
- **Bits per token**: Primary training metric
- **Bits per byte**: More interpretable evaluation metric
- **Next-token accuracy**: Validation accuracy
- **Attention distance histograms**: Analysis of attention patterns

## Performance Results

Training runs achieved the following results on 200-300M token budgets:

1. **Initial baseline**: ~14 bits per byte
2. **Optimized training**: ~12.9 bits per byte (with proper weight decay and warmup)
3. **Improved hyperparameters**: ~12.7 bits per byte (reduced dropout, adjusted weight decay)
4. **Learning rate scheduling**: ~8.7 bits per byte (hold phase in LR schedule)
5. **Latest optimized run**: ~11 bits per byte

### Training Speed Optimizations
- Fused AdamW: ~25% speed improvement (55min → 40min for 300M tokens)
- Activation checkpointing: Memory savings with minimal speed impact
- Gradient accumulation: Enables larger effective batch sizes

## Files

- `transformer_experiments.ipynb`: Main implementation and training notebook
- `rlvr.ipynb`: Placeholder for future RL with verifiable rewards experiments
- `CLAUDE.md`: AI assistant guidelines and coding standards

## Usage

The notebook contains a complete implementation including:

1. **Model Definition**: Full transformer architecture implementation
2. **Training Loop**: Optimized training with logging and validation
3. **Data Loading**: Wikipedia dataset streaming and preprocessing
4. **Evaluation**: Comprehensive metrics and attention analysis
5. **Visualization**: Training curves and attention pattern analysis

## Key Implementation Details

### Memory Optimization
- Activation checkpointing for reduced memory usage
- Gradient accumulation for large effective batch sizes
- Mixed precision training with bfloat16

### Training Stability
- Proper parameter initialization (GPT-2 style)
- Gradient clipping for training stability
- Layer norm for improved convergence

### Evaluation
- Separate validation data stream
- Comprehensive metrics including bits per byte
- Attention pattern analysis for model interpretability

## Future Work

The `rlvr.ipynb` notebook is planned to contain experiments with reinforcement learning using verifiable rewards, building on the foundation established in the main transformer experiments.

## Requirements

- PyTorch 2.x+ (for torch.compile and SDPA)
- transformers (HuggingFace)
- datasets
- matplotlib
- tqdm
- CUDA-capable GPU (recommended)

This implementation serves as both a learning exercise and a foundation for more advanced transformer experiments, with a focus on training efficiency and modern optimization techniques.