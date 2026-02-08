<div align="center">
## 🦙 TinyLlama CPU Inference Engine

> A high-performance local intelligence module for private financial automation


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>
---

## 📖 Overview

This project implements a **1.1 Billion parameter Large Language Model** (TinyLlama) optimized for real-time execution on low-resource consumer hardware. It serves as the **Local Privacy Engine** for the Loan Buddy project, enabling offline analysis of sensitive financial documents like bank statements and KYC data without cloud dependency.

---

## 🎯 Key Features

- **🚀 Edge-Optimized Inference** — Custom KV-Caching logic to prevent CPU bottlenecks during long generations
- **📊 MLOps Dashboard** — Live monitoring of RAM consumption and inference throughput via integrated `psutil` dashboard
- **🤖 Agentic Personalities** — Specialized "Loan Buddy Mode" that shifts from general assistant to financial auditor
- **💾 Memory-Mapped Loading** — Uses `session.use_mmap` to handle large models on limited RAM devices without crashes

---

## 🛠️ Technical Highlights

**The Interview Question:** *"Why did you use ONNX instead of just standard PyTorch?"*

**The Answer:** Standard PyTorch was too resource-intensive for my 8GB RAM and Ryzen 3 environment. By migrating to **ONNX Runtime with 4-bit quantization**, I reduced the memory footprint by roughly **75%**. This allowed me to achieve stable inference speeds of **10-12 tokens/sec**, ensuring the application remains responsive on consumer-grade hardware while maintaining **100% data privacy**.

---

## 🏗️ Architecture

This repository is designed to be used as a **Git Submodule** within the Loan Buddy ecosystem.

```
TINYLLAMA.../
├── models/
│   ├── .cache/huggingface/download/
│   │   └── onnx/                    # Cached model files
│   ├── onnx/
│   │   ├── model_fp16.onnx          # FP16 precision model
│   │   ├── model_q4f16.onnx         # 4-bit quantized model
│   │   └── model.onnx               # Base ONNX model
│   ├── config.json
│   ├── generation_config.json
│   ├── quantize_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── src/
│   ├── app.py                       # Streamlit MLOps Dashboard
│   ├── benchmark.py                 # Performance testing
│   ├── download_model.py            # Model download utility
│   └── engine.py                    # Core inference module
├── venv/                            # Virtual environment
├── requirements.txt                 # Dependency list
└── .gitignore
```

---

## 📊 Performance Benchmarks

**Hardware:** AMD Ryzen 3 3250U | 8GB RAM

| Metric | Value |
|--------|-------|
| **RAM Usage** | 700MB - 900MB (Quantized) |
| **Inference Speed** | ~10.5 Tokens/Second |
| **First Token Latency** | < 100ms |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/TinyLlama-CPU-Engine.git
cd TinyLlama-CPU-Engine
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
streamlit run src/app.py
```

---

## 🔗 Integration

This module is designed to work seamlessly with the [Loan Buddy](https://github.com/yourusername/loan-buddy) project as a privacy-focused inference backend.

---

## 💡 Use Cases

- Private financial document analysis
- Offline KYC verification
- Local bank statement processing
- Edge AI applications requiring data privacy

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [ONNX Runtime](https://onnxruntime.ai/)
- Model: [TinyLlama](https://github.com/jzhang38/TinyLlama)
- Dashboard: [Streamlit](https://streamlit.io/)

---

<div align="center">
  <sub>Built with ❤️ for privacy-conscious AI applications</sub>
</div>
