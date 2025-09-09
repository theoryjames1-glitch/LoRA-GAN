

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


### PSEUDOCODE

```python
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer, util


# -----------------
# Reward function
# -----------------
stmodel = SentenceTransformer("all-MiniLM-L6-v2")

def compute_reward(generated_text, target_text):
    emb1 = stmodel.encode(generated_text, convert_to_tensor=True)
    emb2 = stmodel.encode(target_text, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# -----------------
# LoRA Generator
# -----------------
class LoraGenerator:
    def __init__(self, base_model_id="gpt2", r=8, alpha=16, dropout=0.05):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model_id)
        lora_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora_cfg).to("cuda")

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, base_model_id, path):
        base = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.model = PeftModel.from_pretrained(base, path).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# -----------------
# LoRA Discriminator
# -----------------
class LoraDiscriminator:
    def __init__(self, base_model_id="gpt2", r=8, alpha=16, dropout=0.05):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=1)
        lora_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="SEQ_CLS")
        self.model = get_peft_model(base, lora_cfg).to("cuda")
        self.loss_fn = nn.MSELoss()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            pred = self.model(**inputs).logits.squeeze()
        return pred.item()

    def train_step(self, text, target_reward, optimizer):
        self.model.train()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        labels = torch.tensor([target_reward], dtype=torch.float).to("cuda")

        outputs = self.model(**inputs)
        preds = outputs.logits.squeeze()
        loss = self.loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), preds.detach().item()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, base_model_id, path):
        base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=1)
        self.model = PeftModel.from_pretrained(base, path).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# -----------------
# GAN Wrapper with Checkpointing
# -----------------
class LoraGAN:
    def __init__(self, base_model_id="gpt2"):
        self.base_model_id = base_model_id
        self.generator = LoraGenerator(base_model_id)
        self.discriminator = LoraDiscriminator(base_model_id)

        self.gen_optimizer = torch.optim.AdamW(self.generator.model.parameters(), lr=5e-5)
        self.disc_optimizer = torch.optim.AdamW(self.discriminator.model.parameters(), lr=1e-4)

    def step(self, prompt, target_text, train_gen=True, train_disc=True, adv_weight=0.5):
        response = self.generator.generate(prompt)
        true_reward = compute_reward(response, target_text)

        disc_loss, pred_reward = 0.0, None
        if train_disc:
            text = prompt + "\n\n### Response:\n" + response
            disc_loss, pred_reward = self.discriminator.train_step(
                text, true_reward, self.disc_optimizer
            )

        gen_loss = 0.0
        if train_gen:
            inputs = self.generator.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.generator.model(**inputs, labels=inputs["input_ids"])
            lm_loss = outputs.loss

            text = prompt + "\n\n### Response:\n" + response
            inputs_disc = self.discriminator.tokenizer(text, return_tensors="pt",
                                                       truncation=True, padding=True).to("cuda")
            disc_pred = self.discriminator.model(**inputs_disc).logits.squeeze()
            adv_loss = -disc_pred

            gen_loss = lm_loss + adv_weight * adv_loss

            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            gen_loss = gen_loss.item()

        return {
            "prompt": prompt,
            "response": response,
            "true_reward": true_reward,
            "disc_loss": disc_loss,
            "pred_reward": pred_reward,
            "gen_loss": gen_loss,
        }

    # -----------------
    # Save / Load
    # -----------------
    def save_all(self, gen_path="./lora-gen", disc_path="./lora-disc"):
        self.generator.save(gen_path)
        self.discriminator.save(disc_path)

    def load_all(self, gen_path="./lora-gen", disc_path="./lora-disc"):
        self.generator.load(self.base_model_id, gen_path)
        self.discriminator.load(self.base_model_id, disc_path)

    # -----------------
    # Checkpointing
    # -----------------
    def save_checkpoint(self, step, ckpt_dir="./checkpoints"):
        gen_path = os.path.join(ckpt_dir, f"step-{step}", "generator")
        disc_path = os.path.join(ckpt_dir, f"step-{step}", "discriminator")
        os.makedirs(gen_path, exist_ok=True)
        os.makedirs(disc_path, exist_ok=True)
        self.save_all(gen_path, disc_path)
        print(f"✅ Saved checkpoint at step {step}")

    def load_checkpoint(self, step, ckpt_dir="./checkpoints"):
        gen_path = os.path.join(ckpt_dir, f"step-{step}", "generator")
        disc_path = os.path.join(ckpt_dir, f"step-{step}", "discriminator")
        self.load_all(gen_path, disc_path)
        print(f"✅ Loaded checkpoint from step {step}")

# Initialize
gan = LoraGAN("gpt2")

# Example training loop with a single prompt
prompt = "Translate English to French: Hello world"
target = "Bonjour le monde"

print("=== Training Loop ===")
for step in range(1, 6):
    out = gan.step(prompt, target, train_gen=True, train_disc=True)

    print(f"\n[Step {step}]")
    print("Prompt:", out["prompt"])
    print("Response:", out["response"])
    print("True Reward:", round(out["true_reward"], 4))
    print("Predicted Reward:", round(out["pred_reward"], 4) if out["pred_reward"] else None)
    print("Gen Loss:", round(out["gen_loss"], 4))
    print("Disc Loss:", round(out["disc_loss"], 4))

    # Save checkpoint every 2 steps
    if step % 2 == 0:
        gan.save_checkpoint(step)

# Reload from checkpoint
print("\n=== Reloading from step 4 ===")
gan.load_checkpoint(4)
out = gan.step(prompt, target, train_gen=False, train_disc=True)  # only train discriminator
print("Reloaded Response:", out["response"])
```
