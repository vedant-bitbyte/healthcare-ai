from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    """Central configuration for QLoRA fine-tuning of Phi-3 Mini."""
    
    # Model parameters
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Dataset
    train_data_path: str = "data/chat/train_chat.jsonl"
    val_data_path: str = "data/chat/validation_chat.jsonl"
    
    # Paths
    output_dir: str = "checkpoints"
    final_output_dir: str = "outputs"
    
    # Training parameters
    epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 2
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    resume_from_checkpoint: bool = False
    
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )
