# LogitLens

## Function Overview
Logit Lens presents results in the form of heatmaps. Currently, five visualization modes are supported, and additional ones can be added by expanding the data and/or chart options in `generate_and_visualize`.
- **Current Layer Logit/Prob View**: Text shows the current layer’s top-5 tokens; color encodes the logit/probability of the current layer’s top-1 token.
- **Last Layer Logit/Prob View**: Text shows the current layer’s top-1 token; color encodes the logit/probability of the final layer’s token at the current layer.
- **Last Layer Rank View**: Both text and color represent the vocabulary rank of the final layer’s token within the current layer.

Features
- Supports inspection of attention, attention residual, MLP, and MLP residual.
- Allows inspection at “every N layers”, while always including the final layer.
- Provides an option to normalize before unembedding.
- Supports batch inference: each input generates one heatmap, trimmed to effective length according to the EOS token.
- Currently does not support teacher-forcing inference, viewing by custom intervals/indices, inspection of prompts or the embedding layer.

Code is implemented based on LLaMA. To adapt to different architectures, modify `_register_layer_hooks`.

## Demonstration
<img width="2710" height="1645" alt="1" src="https://github.com/user-attachments/assets/ece9b8d9-1067-4e86-97fa-236598d78291" />
<img width="2688" height="1873" alt="2" src="https://github.com/user-attachments/assets/fb255639-557e-49e5-9fdf-6b4dc172fc7b" />

## Reference
- https://github.com/nostalgebraist/transformer-utils
- https://github.com/zhenyu-02/LogitLens4LLMs
