import os
from loguru import logger
from transformers import Qwen2Config, Qwen2ForCausalLM

def record_model_config(model_name, config):
    logger.info(f"记录模型配置: model_name = {model_name}")
    logger.info(f"hidden_size = {config.hidden_size}, intermediate_size = {config.intermediate_size}")
    logger.info(f"max_position_embeddings = {config.max_position_embeddings}, num_hidden_layers = {config.num_hidden_layers}")
    logger.info(f"num_key_value_heads = {config.num_key_value_heads}, num_attention_heads = {config.num_attention_heads}")

def create_qwen_model(model_name, rank, hidden_size, max_position_embeddings=2048):
    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=hidden_size*4,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=32,
            num_hidden_layers=22,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
    elif model_name == "qwen_small":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=hidden_size*4,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=8,
            num_hidden_layers=8,
            num_key_value_heads=8,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
    else:
        assert 0, f"model {model_name} not supported"
    if rank == 0:
        record_model_config(model_name, config)
    return Qwen2ForCausalLM(config)