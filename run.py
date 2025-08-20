import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import transformers
import warnings

warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message="Matplotlib currently does not support")

# unembedded layers
ATTN = "attention"
ATTN_RS = "attention_residual"
MLP = "mlp"
MLP_RS = "mlp_residual"

SEP = "`"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogitLens:
    def __init__(self, model, tokenizer, apply_norm_before_unembed=True, layer_stride=1, submodules=[ATTN, ATTN_RS, MLP, MLP_RS]):
        self.model = model
        self.tokenizer = tokenizer
        self.submodules = sorted(submodules)
        self.apply_norm_before_unembed = apply_norm_before_unembed
        
        self.hooks = []
        self.hook_results = {}
        self.layer_states = {} 
        
        for i, layer in enumerate(self.model.model.layers):
            if i % layer_stride == 0 or i == len(self.model.model.layers) - 1:  # last layer will always be included
                self._register_layer_hooks(layer, i)
    
    def _register_layer_hooks(self, layer, layer_idx):
        self.hook_results[layer_idx] = {ATTN: [], ATTN_RS: [], MLP: [], MLP_RS: []} # (seq [batch_size, vocab_size])
        self.layer_states[layer_idx] = {}
        
        def residual_hook(module, input):
            self.layer_states[layer_idx]['residual_attention'] = input[0].clone()
        self.hooks.append(layer.register_forward_pre_hook(residual_hook))

        def attn_hook(module, args, output):
            hidden_states = output[0]
            self.hook_results[layer_idx][ATTN].append(self.model.lm_head(self._apply_norm_if_needed(hidden_states))[:, -1, :])
        self.hooks.append(layer.self_attn.register_forward_hook(attn_hook))
        
        def attn_rs_hook(module, args, output):
            hidden_states = self.layer_states[layer_idx]['residual_attention'] + output[0]
            self.hook_results[layer_idx][ATTN_RS].append(self.model.lm_head(self._apply_norm_if_needed(hidden_states))[:, -1, :])

            self.layer_states[layer_idx]['residual_mlp'] = hidden_states.clone()
        self.hooks.append(layer.self_attn.register_forward_hook(attn_rs_hook))
        
        def mlp_hook(module, args, output):
            hidden_states = output
            self.hook_results[layer_idx][MLP].append(self.model.lm_head(self._apply_norm_if_needed(hidden_states))[:, -1, :])
        self.hooks.append(layer.mlp.register_forward_hook(mlp_hook))

        def mlp_rs_hook(module, args, output):
            hidden_states = self.layer_states[layer_idx]['residual_mlp'] + output
            self.hook_results[layer_idx][MLP_RS].append(self.model.lm_head(self._apply_norm_if_needed(hidden_states))[:, -1, :])
        self.hooks.append(layer.mlp.register_forward_hook(mlp_rs_hook))

    def _apply_norm_if_needed(self, hidden_states):
        return self.model.model.norm(hidden_states) if self.apply_norm_before_unembed else hidden_states
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hook_results.clear()
        self.layer_states.clear()
    
    def reset_results(self):
        for layer_results in self.hook_results.values():
            for key in layer_results:
                layer_results[key].clear()
        for layer_state in self.layer_states.values():
            layer_state.clear()

    def generate_and_visualize(self, prompts, max_new_tokens, view_topk=5):
        self.reset_results()
        
        # generation
        inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(device) 
        outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=max_new_tokens, do_sample=False)

        # prepare all data
        layer_names = []
        current_layer_prob, current_layer_logit = {}, {} # {layer:[b, s]}
        current_layer_token, current_layer_topk_tokens = {}, {} 
        last_layer_logit, last_layer_prob, last_layer_rank = {}, {}, {}

        last_layer_token_ids = outputs[:, len(inputs.input_ids[0]):]
        for layer_idx in self.hook_results.keys():
            for key in self.submodules: 
                layer_name = f"l{layer_idx}_{key}"
                layer_names.insert(0, layer_name) # reverse layers, bottom up
                
                stack_logits = torch.stack(self.hook_results[layer_idx][key], dim=1) # [b, s, v]
                stack_probs = stack_logits.softmax(dim=-1)

                topk = stack_probs.topk(view_topk, dim=-1) # [b, s, k]
                top1_index, top1_prob = topk.indices[:,:,0], topk.values[:,:,0] # [b, s]
                current_layer_prob[layer_name] = top1_prob.tolist()
                current_layer_logit[layer_name] = torch.gather(stack_logits, dim=-1, index=top1_index.unsqueeze(-1)).squeeze(-1).tolist()
                current_layer_token[layer_name] = [[SEP + self.tokenizer.decode(token_id) + SEP for token_id in batch_ids] for batch_ids in top1_index]
                current_layer_topk_tokens[layer_name] = [[SEP + "|".join([self.tokenizer.decode(top_token_id) for top_token_id in token_ids]) + SEP for token_ids in batch_ids] for batch_ids in topk.indices]

                last_layer_logit[layer_name] = torch.gather(stack_logits, dim=-1, index=last_layer_token_ids.unsqueeze(-1)).squeeze(-1).tolist()
                last_layer_prob_tmp = torch.gather(stack_probs, dim=-1, index=last_layer_token_ids.unsqueeze(-1))
                last_layer_prob[layer_name] = last_layer_prob_tmp.squeeze(-1).tolist()
                last_layer_rank[layer_name] = ((stack_probs > last_layer_prob_tmp).sum(dim=-1) + 1).tolist()

        # check data
        # assert torch.equal(last_layer_token_ids, torch.stack([logits.argmax(dim=-1) for logits in self.hook_results[31][MLP_RS]], dim=1)) # check hook correction

        # plot heatmaps
        self.plot(prompts, layer_names, last_layer_token_ids, current_layer_topk_tokens, current_layer_logit, "Current Layer Logit View")
        self.plot(prompts, layer_names, last_layer_token_ids, current_layer_topk_tokens, current_layer_prob, "Current Layer Prob View")
        self.plot(prompts, layer_names, last_layer_token_ids, current_layer_token, last_layer_logit, "Last Layer Logit View")
        self.plot(prompts, layer_names, last_layer_token_ids, current_layer_token, last_layer_prob, "Last Layer Prob View")
        self.plot(prompts, layer_names, last_layer_token_ids, last_layer_rank, last_layer_rank, "Last Layer Rank View")

    def plot(self, prompts, layer_names, last_layer_token_ids, raw_content_matrix, raw_color_matrix, tag):
        num_layers = len(layer_names)
        for batch_idx in range(last_layer_token_ids.shape[0]):
            # for each sequence, cut off things after eos
            eos_mask = (last_layer_token_ids[batch_idx] == self.tokenizer.eos_token_id)
            seq_len = eos_mask.int().argmax().item() + 1 if eos_mask.any() else last_layer_token_ids.shape[1]

            # rebuild matrices base on new sequence length
            content_matrix = np.empty((num_layers, seq_len), dtype=object)
            color_matrix = np.zeros((num_layers, seq_len))
            max_content_length = 0
            for i, layer_name in enumerate(layer_names):
                for j in range(seq_len):
                    content_matrix[i, j] = raw_content_matrix[layer_name][batch_idx][j]
                    if isinstance(content_matrix[i, j], str):
                        content_matrix[i, j] = content_matrix[i, j].replace('$', r'\$').replace('_', r'\_').replace('^', r'\^').replace('\n', '\\n') # transform symbols
                        max_content_length = max(len(str(content_matrix[i, j])), max_content_length)
                    color_matrix[i, j] = raw_color_matrix[layer_name][batch_idx][j]
            
            plt.subplots(figsize=(max(2, max_content_length * 0.12) * seq_len, 0.35 * num_layers))
            sns.heatmap(
                    color_matrix, annot=content_matrix, fmt="", 
                    xticklabels=[""] + [SEP + self.tokenizer.decode(token_id) + SEP for token_id in last_layer_token_ids[batch_idx][:seq_len-1]],
                    yticklabels=layer_names,
                    cmap="viridis", 
                    norm=mpl.colors.LogNorm(vmin=1e-5, vmax=self.model.vocab_size) if "rank" in tag.lower() else None, # finer granularity for rank view
                    vmin=np.percentile(np.array(color_matrix), 5) if "rank" not in tag.lower() else None, # shrink valid area for logits & probs view
                    vmax=np.percentile(np.array(color_matrix), 95) if "rank" not in tag.lower() else None, 
            )
            plt.title(f"{tag}")
            plt.xlabel("Prompt: " + prompts[batch_idx].replace('\n', '\\n'), loc='center', fontsize=8, wrap=True)
            plt.tight_layout()
            plt.savefig(f"./logitlens_{''.join(tag.lower().split())}_bsz{batch_idx}.png", dpi=150, bbox_inches='tight')

    def __del__(self):
        self.clear_hooks()

if __name__ == "__main__": 
    # Import Model
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = True 
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "left"

    # LogitLens
    logit_lens = LogitLens(model, tokenizer, layer_stride=4, submodules=[ATTN, ATTN_RS, MLP, MLP_RS])
    logit_lens.generate_and_visualize([
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite one word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a story that tell a very very old myth.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ], max_new_tokens=10)
