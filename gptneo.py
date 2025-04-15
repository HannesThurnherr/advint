# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch


def forward(model, 
            input, 
            start_at_layer=None, 
            stop_at_layer=None, 
            attention_mask=None, 
            past_key_values = None,
            output_attentions = None,
            **kwargs):

    if start_at_layer is None:
        start_at_layer = 0
        
        transformer = model.transformer
                # 2. Get word token embeddings
        input_ids = input
        inputs_embeds = transformer.wte(input_ids) # Shape: (batch_size, seq_length, hidden_size)

        seq_length = inputs_embeds.shape[1] # Get seq_length from embeddings
        device = inputs_embeds.device        # Get device from embeddings

        cache_position = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = cache_position.unsqueeze(0) # Shape: (1, seq_length) -> broadcasts to (batch_size, seq_length)

        position_embeds = transformer.wpe(position_ids) # Shape: (batch_size, seq_length, hidden_size)
        activations = inputs_embeds + position_embeds
        activations = transformer.drop(activations) # 
        
        past_key_values, output_attentions = None, None
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        causal_mask = model.transformer._update_causal_mask(attention_mask, 
                                                   inputs_embeds, cache_position, past_key_values, output_attentions)
        
    else:
        if start_at_layer < 0:
            start_at_layer = model.config.num_layers + start_at_layer
        activations = input
        causal_mask = attention_mask
     
    assert causal_mask is None or causal_mask.ndim == 4, f"causal_mask.ndim: {causal_mask.ndim}"
        
    if stop_at_layer is None:
        my_stop_at_layer = model.config.num_layers
    elif stop_at_layer < 0:
        my_stop_at_layer = model.config.num_layers + stop_at_layer
    else:
        my_stop_at_layer = stop_at_layer

    for layer in range(start_at_layer, my_stop_at_layer):
        activations = model.transformer.h[layer](activations, attention_mask = causal_mask)[0] #attention_mask, etc.

    if stop_at_layer is not None:
        return {'activations': activations, 'attention_mask': causal_mask}

    activations_norm = model.transformer.ln_f(activations)
    logits = model.lm_head(activations_norm)
    return {'logits': logits, 'attention_mask': causal_mask}

if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    prompt = ["Once upon a time there was", "my dog is nice"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    model.to(device)

    output = model(input_ids, output_hidden_states=True)
    hidden = output.hidden_states
    logits = output.logits

    """
    For some idiot reason, the last hidden state is normalized by layer_norm, but not unembedded.
    This is not in line with transformer_lens, where the last hidden state is prior to both layer_norm and unembedding.
    """
    from tqdm import tqdm
    runner = tqdm(range(model.config.num_layers+1))
    for layer in runner:
        print(f"Testing Layer {layer}")
        input_ids = torch.tensor(input_ids).to(model.device)
        output = forward(model, input_ids, stop_at_layer=layer, attention_mask=attention_mask)
        activations, causal_mask = output['activations'], output['attention_mask']
        mask = attention_mask == 1
        if layer == 4:
            torch.testing.assert_close(model.transformer.ln_f(activations)[mask], hidden[layer][mask])
        else:
            torch.testing.assert_close(activations[mask], hidden[layer][mask])

        my_logits = forward(model, activations, start_at_layer=layer, attention_mask=causal_mask)['logits']
        
        torch.testing.assert_close(my_logits[mask], logits[mask])



# %%
# %%
