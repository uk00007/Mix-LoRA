# MixLoRA: Mixture-of-Experts LoRA for Llama-3.2

# Ujjwal Krishna (Roll no: 240102266)
# Paras Jindal (Roll no: 240104072)
This project implements a **MixLoRA (Mixture-of-LoRA-Experts)** architecture from scratch and injects it into the `meta-llama/Llama-3.2-1B-Instruct` model.

Unlike standard LoRA, which applies a single low-rank adapter to a layer, this project replaces the Feed-Forward Networks (MLPs) with a routing mechanism that dynamically selects specific LoRA adapters ("experts") for each token, increasing model capacity without significantly increasing inference cost.

## 🧠 Architecture Overview

The model employs a **Hybrid Adaptation Strategy**:

1. **Attention Layers:** Standard LoRA is applied to `q_proj`, `k_proj`, `v_proj`, and `o_proj` using the Hugging Face `PEFT` library.
2. **MLP Layers:** The standard MLPs are replaced with a custom `MixLoRAFFN_V2` module.

### MixLoRA Logic

For every token passing through an MLP layer:

1. **Base Computation:** The original frozen MLP computes the base output.
2. **Routing:** A `TopKRouter` selects the top **2 experts** out of **8 available experts**.
3. **Expert Computation:** The selected LoRA experts process the input.
4. **Aggregation:** Final Output = `Base Output + Weighted Sum(Expert Outputs)`.

### Technical Specs

* **Base Model:** Llama-3.2-1B-Instruct (4-bit quantization)
* **LoRA Rank (r):** 16
* **Experts:** 8 per layer
* **Top-K:** 2
* **Dataset:** Mental Health Counseling Conversations

---

## 🛠️ Setup & Installation

### Prerequisites

* Python 3.10+
* GPU with CUDA support (Tested on T4/A100)

### Dependencies

Install the required libraries:

```bash
pip install torch transformers peft datasets bitsandbytes accelerate

```

*Note: You must have a Hugging Face token with access to Llama-3.2 to run this.*

---

## 📂 Project Structure

The implementation is contained within the Jupyter Notebook, structured as follows:

1. **Model Loading:** Loads Llama-3.2 in 4-bit precision with Gradient Checkpointing.
2. **Standard LoRA:** Applies basic LoRA to attention mechanisms.
3. **MixLoRA Definition:** Defines `TopKRouter`, `LoRAExpert`, and `MixLoRAFFN_V2`.
4. **Injection:** Iterates through model layers and physically replaces `layer.mlp` with the MixLoRA module.
5. **Training Loop:** A custom training loop (skipping the standard Trainer) to handle **Auxiliary Loss** for load balancing.

---

## 🚀 Key Implementation Details

### 1. The MixLoRA Layer

The core innovation is the replacement of the feed-forward block. The code dynamically injects this custom module:

```python
class MixLoRAFFN_V2(nn.Module):
    def __init__(self, base_ffn, hidden_dim, num_experts=8, k=2, r=16):
        # ... (initialization)
        self.router = TopKRouter(hidden_dim, num_experts, k)
        self.experts = nn.ModuleList([
            LoRAExpert(hidden_dim, r=r) for _ in range(num_experts)
        ])

    def forward(self, x):
        # 1. Compute Base Output (Frozen)
        with torch.no_grad():
            base_out = self.base_ffn(x)
        
        # 2. Route to Top-K Experts
        weights, indices, router_probs = self.router(x)
        
        # 3. Compute Weighted Mixture of Experts
        # ...
        return base_out + expert_out

```

### 2. Custom Loss Function

To prevent "mode collapse" (where the router always picks the same experts), the training loop adds an auxiliary loss:

* **Load Balance Loss:** Penalizes the model if experts are not selected equally.
* **Entropy Loss:** Encourages the router to be decisive (high confidence).

---

## 📊 Training Configuration

| Parameter | Value |
| --- | --- |
| **Learning Rate** | `2e-4` |
| **Optimizer** | AdamW |
| **Batch Size** | 4 |
| **Precision** | `bfloat16` (Experts), `int4` (Base) |
| **Max Token Length** | 512 |
| **Aux Loss Weight** | 0.01 |

---

## 🧪 Usage

### Training

Run the training cells in the notebook. The loop provides real-time feedback on Main Loss vs. Aux Loss:

```text
Step 0: Loss 2.6937 | Aux Loss 10.0000
...
Step 870: Loss 5.2724 | Aux Loss 10.0000

```

*Note: The Aux Loss is clamped to max 10.0 in the implementation to prevent gradient explosions.*

### Inference

The notebook includes an inference block to test the fine-tuned model:

```python
prompt = "I'm not feeling good. What should I do?"
# ... generation code ...

```

---

## ⚠️ Known Limitations & Future Work

* **VRAM Usage:** While efficient, loading 8 experts per layer increases memory usage compared to standard LoRA.
* **Inference Latency:** The routing mechanism adds a small computational overhead during generation.
* **Generation Quality:** As seen in the training logs, the loss curve fluctuates significantly.

---

## 📜 Acknowledgments

* **Base Model:** Meta Llama 3.2
* **Dataset:** `Amod/mental_health_counseling_conversations`
* **Libraries:** Hugging Face Transformers & PEFT

---


