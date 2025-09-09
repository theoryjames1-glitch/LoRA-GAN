

# 🌀 LoRA-GAN Theory

*A lightweight framework for adversarial training of language models with LoRA adapters.*

---

## 📖 Overview

This project introduces a simple but powerful **theory**:

> Attach **two LoRA adapters** to the same base language model:
>
> * **Generator LoRA** → learns to generate responses from prompts.
> * **Discriminator LoRA** → learns to approximate a reward signal (e.g., cosine similarity, human feedback).

Training runs in **step() cycles**:

1. Generator produces a response for a given prompt.
2. Reward is computed (cosine similarity with target, or any metric).
3. Discriminator learns to predict that reward from `(prompt + response)`.
4. Generator optionally receives **adversarial feedback** from the discriminator to maximize predicted reward.

The result is an **online adversarial reward-guided fine-tuning loop**, built entirely with LoRA adapters — lightweight, modular, and checkpointable.

---

## ⚙️ Features

* ✅ **Two LoRA heads** (generator + discriminator) on one base model.
* ✅ **Plug-and-play reward function** (default: cosine similarity via Sentence-BERT).
* ✅ **Flexible training**: update generator, discriminator, or both.
* ✅ **Adversarial mode**: generator maximizes discriminator’s predicted reward.
* ✅ **Checkpointing**: saves LoRA weights + optimizer states per step.
* ✅ **Resume training** exactly where you left off.

---

## 🏗 Architecture

```
Prompt ──► Generator (LoRA) ──► Response
             │
             ▼
       (Prompt + Response)
             │
             ▼
       Discriminator (LoRA) ──► Predicted Reward
             │
             ▼
        Ground Truth Reward (e.g. cosine similarity)
```

* **Generator**: `AutoModelForCausalLM + LoRA`
* **Discriminator**: `AutoModelForSequenceClassification + LoRA`
* **Reward Function**: plug-in metric (default: cosine similarity using `all-MiniLM-L6-v2`).

---

## 🚀 Quickstart

### Install

```bash
pip install torch transformers peft sentence-transformers
```

### Example

```python
from lora_gan import LoraGAN

gan = LoraGAN("gpt2")

# Train both generator & discriminator
out = gan.step(
    prompt="Translate English to French: Hello world",
    target_text="Bonjour le monde",
    train_gen=True,
    train_disc=True
)

print(out)
# {
#   "prompt": "Translate English to French: Hello world",
#   "response": "Bonjour le monde",
#   "true_reward": 0.98,
#   "disc_loss": 0.0021,
#   "pred_reward": 0.95,
#   "gen_loss": 1.34
# }
```

---

## 💾 Checkpointing

Save progress every N steps:

```python
gan.save_checkpoint(step=100)
```

Resume later (with optimizer states intact):

```python
gan.load_checkpoint(step=100)
```

Adapters and optimizers are saved under:

```
./checkpoints/step-100/
  ├── generator/
  │   ├── adapter_config.json
  │   ├── adapter_model.bin
  │   └── optimizer.pt
  └── discriminator/
      ├── adapter_config.json
      ├── adapter_model.bin
      └── optimizer.pt
```

---

## 🔧 Custom Rewards

Replace `compute_reward()` with any function:

* Cosine similarity
* BLEU, ROUGE, METEOR
* Learned reward models (human preference fine-tuned)
* Domain-specific scores (toxicity, factuality, style)

Example:

```python
def compute_reward(response, target):
    return 1.0 if response.strip() == target.strip() else 0.0
```

---

## 🌌 Why This Matters

Traditional fine-tuning requires:

* Huge datasets
* Full model updates

This Theory shows that with **two LoRA adapters**, you can:

* Fine-tune generation + reward modeling **on the fly**.
* Swap adapters independently.
* Keep the base model frozen and lightweight.

It’s a **GAN-inspired framework** for **language model alignment** — simple, modular, and practical.

---

## 📜 License

MIT License. Free to use, modify, and extend.

---

✨ With this framework, you don’t just train models — you train **theory into practice**.
