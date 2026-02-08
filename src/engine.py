import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
model_file = os.path.join(model_dir, "onnx", "model_q4f16.onnx")

# --- 2. MEMORY OPTIMIZATION ---
options = ort.SessionOptions()
options.add_session_config_entry("session.use_mmap", "1") # Uses HDD as buffer
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

# --- 3. LOAD COMPONENTS ---
tokenizer = AutoTokenizer.from_pretrained(model_dir)
session = ort.InferenceSession(model_file, sess_options=options, providers=['CPUExecutionProvider'])

def generate_full_response(user_input, system_prompt="You are a helpful assistant.", max_new_tokens=150):
    # DYNAMIC PROMPT TEMPLATE
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
    
    all_tokens = tokenizer(formatted_prompt, return_tensors="np")["input_ids"][0].tolist()
    input_ids = np.array([all_tokens], dtype='int64')
    
    past_key_values = None
    start_time = time.time()
    tokens_generated = 0

    for _ in range(max_new_tokens):
        seq_len = input_ids.shape[1]
        
        # KV-CACHE LOGIC
        if past_key_values is None:
            input_feed = {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, seq_len), dtype='int64'),
                "position_ids": np.arange(seq_len).reshape(1, seq_len).astype('int64'),
            }
            for i in range(22):
                input_feed[f"past_key_values.{i}.key"] = np.zeros((1, 4, 0, 64), dtype=np.float32)
                input_feed[f"past_key_values.{i}.value"] = np.zeros((1, 4, 0, 64), dtype=np.float32)
        else:
            input_feed = {
                "input_ids": input_ids[:, -1:], 
                "attention_mask": np.ones((1, seq_len), dtype='int64'),
                "position_ids": np.array([[seq_len - 1]], dtype='int64'),
            }
            input_feed.update(past_key_values)

        outputs = session.run(None, input_feed)
        
        # UPDATE KV-CACHE
        past_key_values = {}
        for i in range(22):
            past_key_values[f"past_key_values.{i}.key"] = outputs[i*2 + 1]
            past_key_values[f"past_key_values.{i}.value"] = outputs[i*2 + 2]

        # SAMPLING LOGIC (Intelligence Boost)
        logits = outputs[0][:, -1, :] / 0.7  # Temperature 0.7
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        next_token_id = np.random.choice(len(probs[0]), p=probs[0])
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
        all_tokens.append(next_token_id)
        input_ids = np.array([all_tokens], dtype='int64')
        tokens_generated += 1

    # DECODE FINAL RESPONSE
    full_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    assistant_reply = full_text.split("assistant")[-1].strip()
    
    end_time = time.time()
    tps = tokens_generated / (end_time - start_time) if tokens_generated > 0 else 0
    return assistant_reply, tps