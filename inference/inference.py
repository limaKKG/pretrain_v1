import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import LLaMAForCausalLM
from config.training_config import LLaMAConfig
import torch.nn.functional as F

def generate(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=512, 
    temperature=0.1, 
    top_p=0.9, 
    top_k=100,
    device="cuda"
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated = input_ids
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs["logits"][:, -1, :]
            logits = logits / temperature
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop if EOS or im_end (for ChatML)
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Check for <|im_end|> if the tokenizer has it
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if next_token.item() == im_end_id:
                    break
                
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def generate_hf(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature is not None and float(temperature) > 0.0
    stop_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            stop_ids.append(im_end_id)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--model_dir", type=str, default="/from_s3/pretrain_model", help="HF model directory (config + weights)")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Tokenizer directory (defaults to model_dir)")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory to load weights from (overrides model_dir weights)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = LLaMAConfig()

    model_path = args.model_dir
    tokenizer_path = args.tokenizer_dir or model_path
    weights_path_dir = args.weights_dir or model_path

    # Robust tokenizer loading
    tokenizer = None
    tokenizer_candidates = [tokenizer_path]
    if os.path.isdir(tokenizer_path):
        tokenizer_candidates.extend([
            os.path.join(tokenizer_path, "best"),
            os.path.join(tokenizer_path, "final")
        ])
    
    # Also add model_path as a fallback
    if model_path != tokenizer_path:
        tokenizer_candidates.append(model_path)
        if os.path.isdir(model_path):
            tokenizer_candidates.extend([
                os.path.join(model_path, "best"),
                os.path.join(model_path, "final")
            ])

    for cand in tokenizer_candidates:
        if not os.path.exists(cand):
            continue
        try:
            print(f"Attempting to load tokenizer from {cand}...")
            tokenizer = AutoTokenizer.from_pretrained(cand, trust_remote_code=True, local_files_only=True)
            print(f"Successfully loaded tokenizer from {cand}")
            break
        except Exception as e:
            print(f"[INFO] Could not load tokenizer from {cand}: {e}")

    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer from any of the candidates: {tokenizer_candidates}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Prefer Hugging Face weights (model.safetensors + config.json).
    hf_loaded = False
    model = None
    try:
        print(f"Loading HF model config from {model_path}...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        
        # Determine weights location
        actual_weights_path = None
        for name in ["model.safetensors", "pytorch_model.bin"]:
            cand = os.path.join(weights_path_dir, name)
            if os.path.exists(cand):
                actual_weights_path = cand
                break
        
        if actual_weights_path:
            print(f"Loading HF model with weights from {actual_weights_path}...")
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                state_dict=torch.load(actual_weights_path, map_location="cpu") if actual_weights_path.endswith(".bin") else None,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            # If it was safetensors, from_pretrained might have already loaded from model_path.
            # If we wanted specifically from weights_path_dir for safetensors:
            if actual_weights_path.endswith(".safetensors") and weights_path_dir != model_path:
                from safetensors.torch import load_file
                model.load_state_dict(load_file(actual_weights_path))
            
            model.to(device)
            model.eval()
            hf_loaded = True
        else:
            print(f"[WARN] No weights found in {weights_path_dir}. Trying default from_pretrained.")
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            model.to(device)
            model.eval()
            hf_loaded = True

    except Exception as e:
        print(f"[WARN] Failed to load HF model: {e}")

    # Fallback: load custom model weights if present (legacy path).
    if model is None:
        print(f"Falling back to custom model code with LLaMAConfig: {model_cfg}")
        model_cfg.vocab_size = len(tokenizer)
        model = LLaMAForCausalLM(model_cfg)
        
        # Try to find weights in weights_path_dir first
        weights_path = os.path.join(weights_path_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(model_path, "pytorch_model.bin")
             
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            print(f"[WARN] weights not found at {weights_path}. Using random initialization.")
        model.to(device)
        model.eval()
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Всего параметров: {total_params:,}")
    # print(f"Обучаемых параметров: {trainable_params:,}")
    # print(f"Размер модели: {total_params / 1e9:.2f}B параметров")
    
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        if hf_loaded:
            response = generate_hf(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
            )
        else:
            response = generate(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
            )
        print(f"\nResponse:\n{response}")
    else:
        print("\nModel ready for inference. Type 'exit' to quit.")
        while True:
            try:
                prompt = input("\nEnter prompt: ")
            except EOFError:
                break
                
            if prompt.lower() == 'exit':
                break
                
            if not prompt.strip():
                continue
                
            print("\nGenerating...")
            if hf_loaded:
                response = generate_hf(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    device=device,
                )
            else:
                response = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    device=device,
                )
            print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()
