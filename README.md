# Triton Decode Lab

A hands-on project for building and optimizing LLM decode kernels with Triton, from basic operators to paged attention and model integration.

## Project Goals

### Basic Operators

* RMSNorm
* Fused Residual Add + RMSNorm
* SiLU × Mul

### Decode Attention

* Grouped-Query Attention (GQA) decode with a contiguous KV cache
* Online Softmax
* Split-KV parallelism
* Paged Decode Attention

### System Integration

* KV cache page allocation and reclamation
* KV cache appending
* Batching with variable sequence lengths
* Multi-step decoding
* Integration with one selected model

### Evaluation

Evaluate each optimization across four stages:

**Correctness → Kernel Performance → Module Performance → Model Performance**

Document both performance gains and regressions, including the workloads and configurations where each optimization helps or hurts.

## Progress

* [x] RMSNorm
* [x] Fused Residual Add + RMSNorm
* [ ] SiLU × Mul
* [ ] Contiguous KV GQA Decode
* [ ] Online Softmax
* [ ] Split-KV
* [ ] Paged Decode Attention
* [ ] KV Cache Management
* [ ] Variable-Length Batching
* [ ] Multi-Step Decoding
* [ ] Model Integration
* [ ] End-to-End Evaluation

