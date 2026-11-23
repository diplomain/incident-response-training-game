"""Optional Transformer Continued Pretraining Script for Incident Sequences.

This script demonstrates how continued pretraining on a causal Transformer LM
(e.g., distilgpt2 or SmolLM) would be run if PyTorch and HuggingFace Transformers
are installed in the environment.

Documented procedure:
  - Base Pretrained Model: Hugging Face causal LM (e.g., 'distilgpt2').
  - What is pretrained: Pre-existing transformer blocks, attention heads, token embeddings.
  - What is continued pretrained: The model is further trained on verbalized incident sequences
    (e.g., "Heavy_Rain -> Waterlogging -> Traffic_Disruption -> Cleanup").
  - Hardware: Requires PyTorch and suitable CPU/GPU memory.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset import load_incidents


def verbalize_sequence(seq):
    return " -> ".join(seq) + " <END>"


def run_transformer_continued_pretraining(
    model_name="distilgpt2",
    output_dir="data/transformer_lm_checkpoint",
    epochs=3,
    batch_size=4,
    lr=5e-5,
):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
    except ImportError as e:
        print("\n[INFO] PyTorch or Transformers is not installed in the current environment.")
        print("To run transformer continued pretraining:")
        print("  pip install torch transformers\n")
        print("For offline, lightweight CPU usage, IncidentIQ provides the built-in pure-NumPy")
        print("NeuralNextEventLM in `app/neural_lm.py` which trains locally without torch.")
        return False

    print(f"Loading base pretrained model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    incidents = load_incidents()
    texts = [verbalize_sequence(s) for s in incidents]
    
    tmp_corpus = Path("data/verbalized_incidents.txt")
    tmp_corpus.parent.mkdir(parents=True, exist_ok=True)
    tmp_corpus.write_text("\n".join(texts), encoding="utf-8")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(tmp_corpus),
        block_size=64,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    print("Beginning continued pretraining...")
    trainer.train()
    print(f"Continued pretraining finished. Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Transformer continued pretraining complete.")
    return True


if __name__ == "__main__":
    run_transformer_continued_pretraining()
