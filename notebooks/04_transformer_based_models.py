import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments,
        TrainerCallback,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import matplotlib.pyplot as plt
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore")
    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        Path,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        mo,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # T5 Fine-tuning for Text Simplification

    This notebook demonstrates fine-tuning T5-base on the ASSET text simplification dataset using HuggingFace Trainer.

    Reference: https://www.philschmid.de/fine-tune-flan-t5
    """)
    return


@app.cell
def _(mo):
    asset_folder_path = mo.ui.file_browser(
        selection_mode="directory",
        multiple=False,
        label="Asset folder",
        initial_path="../data"
    )
    asset_folder_path
    return (asset_folder_path,)


@app.cell
def _(Dataset):
    def load_asset_data(asset_folder_path, split="valid"):
        """Load ASSET dataset from folder path and return HuggingFace Dataset"""
        import os

        src_sentences = []
        tgt_sentences = []


        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]  # Remove .split.orig extension
                orig_path = os.path.join(asset_folder_path, file_name)

                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                simp_files = [
                    os.path.join(asset_folder_path, simp_file_name)
                    for simp_file_name in os.listdir(asset_folder_path)
                    if simp_file_name.startswith(base_name) and f".{split}.simp." in simp_file_name
                ]

                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences = [line.strip() for line in f if line.strip()]
                        simp_sentences_list.append(simp_sentences)


                for i, orig_sentence in enumerate(orig_sentences):
                    for simp_sentences in simp_sentences_list:
                        if i < len(simp_sentences):
                            src_sentences.append("Simplify: " + orig_sentence)
                            tgt_sentences.append(simp_sentences[i])

        if not src_sentences or not tgt_sentences:
            print("Warning: No data loaded. Check the folder structure and file naming conventions.")

        # Create HuggingFace Dataset
        data_dict = {
            "source": src_sentences,
            "target": tgt_sentences,
        }

        return Dataset.from_dict(data_dict)
    return (load_asset_data,)


@app.cell
def _(AutoTokenizer, asset_folder_path, load_asset_data, mo):
    tokenizer = AutoTokenizer.from_pretrained("t5-base", legacy=False)

    train_dataset_t5 = load_asset_data(asset_folder_path.path(index=0), split="valid")
    test_dataset_t5 = load_asset_data(asset_folder_path.path(index=0), split="test")

    mo.md(f"""
    ### Dataset Loaded!

    - **Training samples**: {len(train_dataset_t5):,}
    - **Test samples**: {len(test_dataset_t5):,}
    - **Model**: T5-base
    - **Tokenizer vocabulary size**: {len(tokenizer):,}

    **Example pair:**
    - Complex: {train_dataset_t5[0]['source']}
    - Simple: {train_dataset_t5[0]['target']}
    """)
    return test_dataset_t5, tokenizer, train_dataset_t5


@app.cell
def _(tokenizer):
    def preprocess_function(examples, max_length=256):
        """Tokenize the examples and prepare decoder inputs"""
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        # labels["input_ids"] = [
        #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        # ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    return (preprocess_function,)


@app.cell
def _(preprocess_function, test_dataset_t5, train_dataset_t5):
    tokenized_train = train_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )
    tokenized_test = test_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )

    print(f"Tokenized training samples: {len(tokenized_train):,}")
    print(f"Tokenized test samples: {len(tokenized_test):,}")
    return tokenized_test, tokenized_train


@app.cell
def _(AutoModelForSeq2SeqLM):
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return (model_t5,)


@app.cell
def _(DataCollatorForSeq2Seq, model_t5, tokenizer):
    # label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_t5,
    )
    return (data_collator,)


@app.cell
def _(mo):
    mo.md("""
    ## Training Hyperparameters

    Configure the training parameters:
    """)
    return


@app.cell
def _(mo):
    learning_rate_t5 = mo.ui.slider(
        1e-5, 10e-5, value=5e-5, step=1e-6, label="Learning Rate"
    )
    n_epochs_t5 = mo.ui.slider(1, 10, value=5, step=1, label="Number of Epochs")
    batch_size_t5 = mo.ui.slider(8, 64, value=16, step=8, label="Batch Size")
    warmup_ratio_t5 = mo.ui.slider(0.0, 0.2, value=0.1, step=0.01, label="Warmup Ratio")

    mo.hstack([
        mo.vstack([learning_rate_t5, n_epochs_t5]),
        mo.vstack([batch_size_t5, warmup_ratio_t5])
    ])
    return batch_size_t5, learning_rate_t5, n_epochs_t5, warmup_ratio_t5


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    batch_size_t5,
    data_collator,
    learning_rate_t5,
    mo,
    model_t5,
    n_epochs_t5,
    plt,
    tokenized_test,
    tokenized_train,
    torch,
    train_button,
    warmup_ratio_t5,
):
    # Determine device
    if torch.cuda.is_available():
        device_t5 = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_t5 = torch.device("mps")
    else:
        device_t5 = torch.device("cpu")


    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=n_epochs_t5.value,
        per_device_train_batch_size=batch_size_t5.value,    
        per_device_eval_batch_size=batch_size_t5.value,
        warmup_ratio=warmup_ratio_t5.value,
        learning_rate=learning_rate_t5.value,
        fp16=False, # Overflows with fp16
        # weight_decay=0.01,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        # optim="adamw_torch_fused",
        # label_smoothing_factor=0.1,
        # max_grad_norm=0.5,
        use_mps_device=(device_t5.type == "mps"),
    )

    # Custom callback to track losses for plotting
    class LossCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)

    loss_callback = LossCallback()

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model_t5,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    fig_t5, ax_t5 = plt.subplots(figsize=(10, 6))

    if train_button.value:
        print(f"Training on device: {device_t5}")
        print(f"Model parameters: {sum(p.numel() for p in model_t5.parameters()):,}\n")
        print("Training Configuration:")
        print("=" * 40)
        print(f"Learning Rate:    {learning_rate_t5.value}")
        print(f"Number of Epochs: {n_epochs_t5.value}")
        print(f"Batch Size:       {batch_size_t5.value}")
        print(f"Warmup Ratio:     {warmup_ratio_t5.value}")
        print("=" * 40 + "\n")

        # Train the model
        train_result = trainer.train()

        # Get final evaluation
        eval_result = trainer.evaluate()

        # Plot losses
        ax_t5.clear()
        if loss_callback.train_losses:
            ax_t5.plot(loss_callback.train_steps, loss_callback.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)

        if loss_callback.eval_losses:
            ax_t5.plot(loss_callback.eval_steps, loss_callback.eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o', markersize=6)

        ax_t5.set_xlabel("Training Steps", fontsize=12)
        ax_t5.set_ylabel("Loss", fontsize=12)
        ax_t5.set_title("T5 Training Progress", fontsize=14, fontweight="bold")
        ax_t5.legend(fontsize=11)
        ax_t5.grid(True, alpha=0.3)
        plt.tight_layout()

        mo.md(f"""
        ### Training Complete!

        **Final Training Loss**: {train_result.training_loss:.4f}

        **Final Evaluation Loss**: {eval_result['eval_loss']:.4f}

        **Training Time**: {train_result.metrics['train_runtime']:.2f} seconds ({train_result.metrics['train_runtime']/60:.1f} minutes)
        """)

        mo.output.append(ax_t5)
    return device_t5, trainer


@app.cell
def _(Path, tokenizer, trainer):


    save_path = Path("./fine_tuned_t5_simplification")
    save_path.mkdir(exist_ok=True)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Simplification

    Test the fine-tuned model and compare with pre-trained models:
    """)
    return


@app.cell
def _(mo):
    test_input = mo.ui.text_area(
        label="Enter a complex sentence to simplify:",
        value="A Georgian inscription around the drum attests his name.",
        rows=3,
    )
    test_input
    return (test_input,)


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    device_t5,
    mo,
    model_t5,
    test_input,
    tokenizer,
    torch,
):
    def simplify_with_t5(text, model, tokenizer, device, max_length=128):
        """Simplify text using T5 model"""
        model.eval()

        input_text = "Simplify: " + text

        input_ids = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


    input_text_test = test_input.value.strip()

    # Our fine-tuned model
    simplified_ours = simplify_with_t5(
        input_text_test, model_t5, tokenizer, device_t5
    )

    # Load pre-trained models
    tokenizer_eilamc = AutoTokenizer.from_pretrained(
        "eilamc14/t5-base-text-simplification", legacy=False
    )
    model_eilamc = AutoModelForSeq2SeqLM.from_pretrained(
        "eilamc14/t5-base-text-simplification"
    ).to(device_t5)

    # tokenizer_mrm = T5Tokenizer.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification", legacy=False
    # )
    # model_mrm = T5ForConditionalGeneration.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification"
    # ).to(device_t5)

    # Simplify with pre-trained models
    simplified_eilamc = simplify_with_t5(
        input_text_test, model_eilamc, tokenizer_eilamc, device_t5
    )
    # simplified_mrm = simplify_with_t5(
    #     input_text_test, model_mrm, tokenizer_mrm, device_t5
    # )

    mo.md(f"""
    ### Model Comparison

    **Original (Complex):**
    > {input_text_test}

    ---

    | Model | Simplified Output |
    |-------|-------------------|
    | **Our Fine-tuned T5-base** | {simplified_ours} |
    | **eilamc14/t5-base-text-simplification** | {simplified_eilamc} |
    """)
    return


if __name__ == "__main__":
    app.run()
