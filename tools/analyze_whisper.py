
import torch
import transformers
from transformers import WhisperConfig, WhisperModel

def analyze_whisper_architecture(model_name="ivrit-ai/whisper-large-v3"):
    print(f"Analyzing architecture for: {model_name}")
    try:
        config = WhisperConfig.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load config for {model_name}: {e}")
        print("Falling back to openai/whisper-large-v3")
        model_name = "openai/whisper-large-v3"
        try:
            config = WhisperConfig.from_pretrained(model_name)
        except Exception as e:
             print(f"Could not load config for {model_name}: {e}")
             return

    print("\n=== General Configuration ===")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Hidden Size (d_model): {config.d_model}")
    print(f"Num Hidden Layers (Encoder): {config.encoder_layers}")
    print(f"Num Hidden Layers (Decoder): {config.decoder_layers}")
    print(f"Num Attention Heads (Encoder): {config.encoder_attention_heads}")
    print(f"Num Attention Heads (Decoder): {config.decoder_attention_heads}")
    print(f"FFN Dimension (Encoder): {config.encoder_ffn_dim}")
    print(f"FFN Dimension (Decoder): {config.decoder_ffn_dim}")
    print(f"Max Source Positions: {config.max_source_positions}")
    print(f"Max Target Positions: {config.max_target_positions}")

    print("\n=== Convolutional Frontend ===")
    # Whisper usually has 2 Conv1d layers
    # We can infer strides/kernels from config usually, or default structure
    # But usually it's standard.
    # Let's see if config has conv params.
    # It seems config.num_mel_bins is key input dim
    print(f"Num Mel Bins: {config.num_mel_bins}")


    print("\n=== Layer Replacement Strategy ===")
    print("Identifying Dense Layers suitable for DeepSpectralLinear:")

    # Encoder
    print(f"\n[Encoder x {config.encoder_layers}]")
    print(f"  - Self-Attention Q, K, V Projections: Linear({config.d_model}, {config.d_model}) -> [Target for Spectral]")
    print(f"  - Self-Attention Output Projection: Linear({config.d_model}, {config.d_model}) -> [Target for Spectral]")
    print(f"  - FFN fc1: Linear({config.d_model}, {config.encoder_ffn_dim}) -> [Target for Spectral]")
    print(f"  - FFN fc2: Linear({config.encoder_ffn_dim}, {config.d_model}) -> [Target for Spectral]")

    # Decoder
    print(f"\n[Decoder x {config.decoder_layers}]")
    print(f"  - Self-Attention Q, K, V, Out: Similar to Encoder -> [Target for Spectral]")
    print(f"  - Cross-Attention Q, K, V, Out: Linear({config.d_model}, {config.d_model}) -> [Target for Spectral]")
    print(f"  - FFN fc1, fc2: Similar to Encoder -> [Target for Spectral]")

    print("\n=== Dimension Mapping for WHT ===")
    # Check if dimensions are powers of 2

    def check_pow2(d, name):
        import math
        log2 = math.log2(d)
        is_pow2 = log2.is_integer()
        next_pow2 = 2**math.ceil(log2)
        print(f"{name}: {d} -> Is Power of 2? {is_pow2}. Next Pow2: {next_pow2}. Pad required: {next_pow2 - d}")

    check_pow2(config.d_model, "d_model")
    check_pow2(config.encoder_ffn_dim, "encoder_ffn_dim")

    # Usually d_model=1280 for Large. 1280 is not pow2. 1024 or 2048.
    # ffn is 4 * d_model usually. 4 * 1280 = 5120. Next pow2 8192.

    print("\n=== Activation Flows ===")
    print(f"Activation Function: {config.activation_function}")
    print("Spectral ViT used GELU. Whisper uses GELU usually.")

if __name__ == "__main__":
    analyze_whisper_architecture()
