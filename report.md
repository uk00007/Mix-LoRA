**Project Report: MixLoRA Implementation for Mental Health LLM
Subject: Fine-tuning Llama-3.2 using Mixture-of-Experts LoRA (MixLoRA)**

1. Executive Summary
This project explores the implementation of MixLoRA (Mixture-of-LoRA-Experts), a parameter-efficient fine-tuning technique applied to the Llama-3.2-1B-Instruct Large Language Model. The objective was to enhance the model's capability in mental health counseling scenarios by replacing standard Feed-Forward Networks (FFNs) with a dynamic routing mechanism that selects specialized LoRA experts per token.

2. Methodology & Architecture
2.1 Base Model
The project utilizes Meta Llama 3.2 1B Instruct, loaded in 4-bit quantization to optimize memory efficiency during training.

2.2 Hybrid Adaptation Strategy
Instead of applying a uniform adaptation across the entire network, a hybrid approach was adopted:

Attention Layers: Standard Low-Rank Adaptation (LoRA) was applied to the query, key, value, and output projection matrices (q_proj, k_proj, v_proj, o_proj).

FFN Layers (The Core Innovation): The model's original Multi-Layer Perceptrons (MLPs) were physically replaced with a custom MixLoRA Layer (MixLoRAFFN_V2).

2.3 The MixLoRA Mechanism
The custom architecture introduces a sparse Mixture-of-Experts (MoE) dynamic:

Router: A TopKRouter evaluates the input and assigns a probability distribution over the experts.

Selection: For every token, the top 2 experts (out of 8 available) are activated.

Computation: The output is a weighted sum of the activated experts plus the original frozen base layer output.

3. Implementation Details
3.1 Dataset
The model was fine-tuned on the Amod/mental_health_counseling_conversations dataset. Data processing involved tokenizing conversation pairs (Context + Response) with a maximum sequence length of 512 tokens.

3.2 Training Configuration
A custom training loop was implemented to handle the complex loss landscape of MoEs:

Optimizer: AdamW with a learning rate of 2e-4.

Batch Size: 4.

Loss Function: A composite loss combining the primary Causal Language Modeling loss with an Auxiliary Loss (Load Balancing + Entropy) to ensure diverse expert usage.

4. Experimental Results
4.1 Training Dynamics
The training spanned approximately 870 steps.

Initial Phase: The model started with a standard cross-entropy loss of ~2.69.

Loss Behavior: The training exhibited instability. The loss did not converge downwards; instead, it drifted upwards, reaching ~5.27 - 6.0+ by step 800.

Auxiliary Loss: The auxiliary loss (used to balance experts) remained clamped at its maximum value (10.00) throughout the training, suggesting the router struggled to balance the load effectively.

4.2 Qualitative Evaluation
Inference was tested with the prompt: "I'm not feeling good. What should I do?" The generated output displayed some context awareness (mentioning "therapist", "anxiety") but suffered from hallucination and grammatical incoherence (e.g., "My friend is my parents... I think I have been an 12 disorder"). This correlates with the rising loss values observed during training.

5. Conclusion
The implementation successfully demonstrated the mechanical feasibility of injecting MixLoRA layers into a pre-trained Llama-3 model. The custom routing architecture functioned technically, allowing for dynamic forward passes.

However, the rising loss curve indicates that the model experienced divergence. Future work must focus on stabilizing the Router Loss. The current auxiliary loss weight might be too high, or the learning rate for the router parameters needs to be decoupled from the rest of the model to prevent the experts from fighting against the base model's knowledge.
