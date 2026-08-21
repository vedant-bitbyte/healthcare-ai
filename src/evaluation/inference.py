import time
import logging
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from pathlib import Path

logger = logging.getLogger(__name__)

def load_model_for_inference(base_model_name: str, adapter_path: str) -> Tuple[Any, Any]:
    """
    Loads the base model in 4-bit and attaches the trained LoRA adapter.
    """
    adapter_path_obj = Path(adapter_path).resolve()
    if not adapter_path_obj.exists() or not adapter_path_obj.is_dir():
        raise FileNotFoundError(
            f"The LoRA adapter directory was not found at: {adapter_path_obj}\n"
            f"Ensure the adapter path is configured correctly in settings.py or .env."
        )
    
    config_path = adapter_path_obj / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing adapter_config.json in adapter directory: {adapter_path_obj}\n"
            f"Found files: {[f.name for f in adapter_path_obj.iterdir()]}"
        )

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

def generate_answers(
    model: Any,
    tokenizer: Any,
    questions: List[str],
    max_new_tokens: int = 512
) -> List[Dict[str, Any]]:
    """
    Generates answers for a list of questions using deterministic decoding.
    Tracks latency in milliseconds.
    
    Args:
        model: The PEFT model.
        tokenizer: The tokenizer.
        questions: A list of question strings.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        List of dictionaries containing 'question', 'answer', and 'latency_ms'.
    """
    results = []
    
    for idx, question in enumerate(questions):
        messages = [{"role": "user", "content": question}]
        
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0
            )
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        input_length = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_length:]
        
        answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        results.append({
            "question": question,
            "answer": answer_text,
            "latency_ms": latency_ms
        })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(questions)} questions...")
            
    return results
