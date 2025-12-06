from loguru import logger
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM

def record_model_config(model_name, config):
    logger.info(f"记录模型配置: model_name = {model_name}")
    logger.info(f"hidden_size = {config.hidden_size}, intermediate_size = {config.intermediate_size}")
    logger.info(f"max_position_embeddings = {config.max_position_embeddings}, num_hidden_layers = {config.num_hidden_layers}")
    logger.info(f"num_key_value_heads = {config.num_key_value_heads}, num_attention_heads = {config.num_attention_heads}")

def create_model(model_name, rank, hidden_size, max_position_embeddings=1024):
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
        model=Qwen2ForCausalLM(config)
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
        model=Qwen2ForCausalLM(config)
    elif model_name == "llama_130m":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,  # LLaMA通常使用1作为BOS
            eos_token_id=2,  # LLaMA通常使用2作为EOS
            hidden_act="silu",
            hidden_size=792,
            initializer_range=0.02,
            intermediate_size=2048,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=12,
            num_hidden_layers=16,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,  # RoPE基础频率
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,  # LLaMA-2 tokenizer词汇表大小
        )
        model = LlamaForCausalLM(config)

    elif model_name == "llama_350m":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=2560,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=16,
            num_hidden_layers=30,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,
        )
        model = LlamaForCausalLM(config)

    elif model_name == "llama_1.1b":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=5632,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=24,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,
        )
        model = LlamaForCausalLM(config)
    else:
        logger.error(f"不支持的模型: {model_name}")
        return None
    if rank == 0:
        record_model_config(model_name, config)
    return model