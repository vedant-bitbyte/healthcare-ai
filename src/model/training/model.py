import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import TrainingConfig

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(config: TrainingConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the Phi-3 tokenizer and model with 4-bit quantization,
    prepares it for k-bit training, and attaches PEFT LoRA adapters.
    
    Args:
        config (TrainingConfig): The configuration object containing model and LoRA parameters.
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The configured model and tokenizer.
    """
    logger.info(f"Loading tokenizer for {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
    )
    
    # Phi-3 typically requires setting the pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading model {config.model_name} in 4-bit quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    logger.info("Preparing model for k-bit training")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Attaching PEFT LoRA adapters")
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
        target_modules=config.target_modules
    )
    
    model = get_peft_model(model, peft_config)
    
    # Log trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    logger.info(
        f"Trainable params: {trainable_params:,d} || "
        f"All params: {all_param:,d} || "
        f"Trainable%: {100 * trainable_params / all_param:.4f}%"
    )
    
    return model, tokenizer
