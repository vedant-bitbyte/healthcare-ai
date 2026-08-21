import logging
import time
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from .config import TrainingConfig
from .dataset import load_and_validate_dataset
from .model import load_model_and_tokenizer

logger = logging.getLogger(__name__)

def formatting_prompts_func(example):
    """
    Formats the messages into a single prompt string if needed,
    but SFTTrainer natively supports conversational formatting if 
    we pass the raw messages to a chat template, or we can use 
    dataset_kwargs if the dataset has 'messages'.
    """
    return example

class CustomLoggingCallback(TrainerCallback):
    """Custom callback to log GPU memory and training time."""
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if hasattr(self, "start_time"):
                logs["training_time_sec"] = round(time.time() - self.start_time, 2)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                logs["gpu_memory_allocated_gb"] = round(allocated, 2)

def train(config: TrainingConfig) -> None:
    """
    Orchestrates the entire QLoRA fine-tuning process.
    Loads dataset, loads model, initializes SFTTrainer, 
    starts training, and saves the final checkpoints.
    
    Args:
        config (TrainingConfig): Configuration object.
    """
    logger.info("Initializing training pipeline...")
    
    # 1. Load Datasets
    train_dataset = load_and_validate_dataset(config.train_data_path)
    val_dataset = load_and_validate_dataset(config.val_data_path)
    
    # 2. Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Apply chat template if not already present (Phi-3 defaults exist, but this ensures standard chat format is recognized)
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        logger.warning("Tokenizer lacks chat_template. Setting a default basic template.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}{% endfor %}"

    # 3. Setup SFTConfig
    logger.info("Setting up SFTConfig")
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        max_length=config.max_seq_length,
        fp16=False, # Phi-3 and newer GPUs recommend bf16
        bf16=True,
        optim="paged_adamw_8bit", # Common optimizer for QLoRA to save memory
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir="logs/tensorboard"
    )
    
    # 4. Initialize SFTTrainer
    logger.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        # TRL handles 'messages' directly when passed, but requires the tokenizer to apply the chat_template
        processing_class=tokenizer,
        callbacks=[CustomLoggingCallback()]
    )
    
    # 5. Start Training
    logger.info("Starting training...")
    # Optional: We can accept a resume_from_checkpoint parameter in train(), but for now trainer.train() 
    # can take resume_from_checkpoint natively if we pass it, handled below.
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint if hasattr(config, 'resume_from_checkpoint') else None)
    
    # 6. Save Final Model
    logger.info(f"Training completed. Saving final adapter to {config.final_output_dir}...")
    trainer.model.save_pretrained(config.final_output_dir)
    tokenizer.save_pretrained(config.final_output_dir)
    logger.info("Final adapter saved successfully.")
