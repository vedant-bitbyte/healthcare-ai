import time
import logging
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

def load_model_for_inference(base_model_name: str, adapter_path: str) -> Tuple[Any, Any]:
    """
    Loads the base model in 4-bit and attaches the trained LoRA adapter.
    """
    logger.info(f"Loading tokenizer for {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {base_model_name} in 4-bit quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_responses(
    model: Any, 
    tokenizer: Any, 
    test_data: List[Dict[str, Any]], 
    max_new_tokens: int = 512
) -> List[Dict[str, Any]]:
    """
    Generates responses for the test dataset and tracks latency.
    Expects test_data to be a list of dictionaries with a 'messages' key.
    """
    results = []
    
    for idx, sample in enumerate(test_data):
        messages = sample.get("messages", [])
        
        # Split into prompt (user) and ground truth (assistant)
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            logger.warning(f"Skipping sample {idx} due to unexpected format.")
            continue
            
        prompt_messages = messages[:-1]
        ground_truth = messages[-1]["content"]
        
        # Format input using chat template
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding for evaluation
                temperature=None,
                top_p=None
            )
            
        latency = time.time() - start_time
        
        # Extract only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_length:]
        
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response_length = len(generated_ids)
        
        results.append({
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "prediction": response_text,
            "latency_sec": latency,
            "response_length_tokens": response_length
        })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_data)} samples...")
            
    return results
