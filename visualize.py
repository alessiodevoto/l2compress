import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis
import numpy as np
import torch

TICKANGLE = 90
MAX_DISPLAY_LABELS = 35
HEIGHT = 300
FONT_SIZE = 16
AVG_POOL_SIZE = None

def plot_kv_cache_norms(past_key_values, key_or_value='key', layers_to_inspect=None, heads_to_inspect=None, out_file=None, labels=None):
    """
    Plots the norms of the key or value vectors in the key-value cache.

    Args:
        past_key_values (list): A list of tensors representing the key-value cache at different time steps.
        key_or_value (str, optional): Specifies whether to plot the norms of the 'key' or 'value' vectors. Defaults to 'key'.
        layers_to_inspect (list, optional): A list of layer indices to inspect. Defaults to None, which inspects all layers.
        heads_to_inspect (list, optional): A list of head indices to inspect. Defaults to None, which inspects all heads.
        out_file (str, optional): The file path to save the plot. Defaults to None, which displays the plot without saving. If the file path ends with '.html', the plot is saved as an interactive HTML file. Otherwise, the plot is saved as an image file. Defaults to None.
        labels (list, optional): A list of labels for the x-axis, that should correspond to the tokens. Defaults to None.
    """

    if key_or_value == 'query':
        key_or_value = 'key'
        past_key_values = [[q] for q in past_key_values]
        

    key_or_value = 0 if key_or_value == 'key' else 1

    num_layers = len(past_key_values)
    num_heads = past_key_values[0][key_or_value].shape[1]

    layers_to_inspect = layers_to_inspect or list(range(0, num_layers))
    heads_to_inspect = heads_to_inspect or list(range(0, num_heads))

    if labels:
        labels = [l.strip('▁') for l in labels]
        labels_len = len(labels) if len(labels) < MAX_DISPLAY_LABELS else len(labels)//2

    fig = make_subplots(
        rows=len(layers_to_inspect),
        cols=len(heads_to_inspect),
        # subplot_titles=[f'Head {head}' for head in heads_to_inspect] + [""] * (len(layers_to_inspect)+1),
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    #print(past_key_values[0][0].shape)
    #print(past_key_values[0][1].shape)
    for l, layer in enumerate(layers_to_inspect):
        for i, head in enumerate(heads_to_inspect):
            norms = past_key_values[layer][key_or_value][0, head, :, :].norm(dim=-1, p=2).detach().cpu().numpy()
            # color should be based on norm value
            bar = go.Bar(y=norms, marker=dict(color=np.log(norms), colorscale='Viridis'))
           
            
            if labels: fig.update_xaxes(tickangle=-TICKANGLE, tickmode='array', tickvals=list(range(len(labels))), ticktext=labels, showgrid=False, tickfont=dict(size=FONT_SIZE))
            fig.update_yaxes(showgrid=False, showticklabels=True)  
            # if i == 0: fig.update_yaxes(title_text=f'Layer {layer}', row=l + 1, col=i + 1,  title_font_size=FONT_SIZE)

            fig.update_layout(bargap=0.2, bargroupgap=0, barmode='group')
            fig.update_layout(plot_bgcolor='white')

            fig.add_trace(bar, row=l + 1, col=i + 1)
            

    fig.update_layout(
        height=HEIGHT * len(layers_to_inspect),
        width=400 * len(heads_to_inspect),
        # title_text='Key-Value Cache Norms',
        showlegend=False,
        margin=dict(l=5, r=5, t=30, b=0),
    )

    fig.update_annotations(font_size=FONT_SIZE)

    if out_file:
        if out_file.endswith('.html'):
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)
    





def plot_attentions(attention_scores, layers_to_inspect=None, heads_to_inspect=None, out_file=None, labels=None):
    """
    Plot the attention scores for each layer and head.

    Args:
        attention_scores (list): List of attention score tensors for each layer.
        layers_to_inspect (list, optional): List of layer indices to inspect. Defaults to None.
        heads_to_inspect (list, optional): List of head indices to inspect. Defaults to None.
        out_file (str, optional): File path to save the plot. Defaults to None. If the file path ends with '.html', the plot is saved as an interactive HTML file. Otherwise, the plot is saved as an image file. Defaults to None.
        labels (list, optional): A list of labels for the x-axis and y-axis, that should correspond to the tokens. Defaults to None.
    """
    num_heads = attention_scores[0].shape[1]
    num_layers = len(attention_scores)

    layers_to_inspect = layers_to_inspect or list(range(0, num_layers))
    heads_to_inspect = heads_to_inspect or list(range(0, num_heads))

    # Create subplots
    fig = make_subplots(
        rows=len(layers_to_inspect),
        cols=len(heads_to_inspect),
        subplot_titles=[f'Head {head}' for head in heads_to_inspect] + [""] * (len(layers_to_inspect)+1),
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    if labels:
        labels = [l.strip('▁') for l in labels]
        labels_len = len(labels) if len(labels) < MAX_DISPLAY_LABELS else len(labels)//2
        

    # Add heatmaps to the subplots
    for l, layer in enumerate(layers_to_inspect):
        for i, head in enumerate(heads_to_inspect):
            attention_matrix = attention_scores[layer][0, head, :, :].detach().cpu().numpy()
            # print(attention_matrix.shape)
            # mask the upper triangle
            mask = np.triu_indices_from(attention_matrix, k=1)
            attention_matrix[mask] = np.nan
            #print(mask.shape) 
            # print(attention_matrix)
            # make sure background is white
            
            heatmap = go.Heatmap(
                #autocolorscale=False,
                #connectgaps=False,
                z=attention_matrix, 
                #x=labels, 
                #y=labels, 
                colorscale='Viridis',
                zmin = 0,
                zmax = 1,
                xgap = 0.5, # Sets the horizontal gap (in pixels) between bricks
                ygap = 0.5,
                hoverongaps=False,
                showscale=False
                )
            
            fig.update_yaxes(autorange='reversed')
            #if labels: fig.update_xaxes(scaleanchor='y', scaleratio=1, tickangle=-TICKANGLE, tickmode='array', tickvals=list(range(attention_matrix.shape[1]-1)), ticktext=labels, tickfont=dict(size=FONT_SIZE))
            #if labels: fig.update_yaxes(scaleanchor='x', scaleratio=1, tickmode='array', tickvals=[], ticktext=[],showticklabels=False)
            if labels: fig.update_xaxes(tickangle=-TICKANGLE, tickmode='array', tickvals=list(range(attention_matrix.shape[1]-1)), ticktext=labels, tickfont=dict(size=FONT_SIZE))
            if labels: fig.update_yaxes(tickmode='array', tickvals=[], ticktext=[],showticklabels=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(plot_bgcolor='white')
            

            fig.add_trace(heatmap, row=l + 1, col=i + 1)


            # reversed bcs plotly has reversed y-axis by default
            # see https://community.plotly.com/t/heatmap-y-axis-is-reversed-by-default-going-against-standard-convention-for-matrices/32180
            # if i == 0: fig.update_yaxes(title_text=f'Layer {layer}', row=l + 1, col=i + 1,  title_font_size=FONT_SIZE) 
            
            
                
                

    fig.update_layout(
        height=400 * len(layers_to_inspect),
        width=400 * len(heads_to_inspect),
        margin=dict(l=5, r=5, t=30, b=0),
        #title_text='Attention Scores'
    )

    fig.update_annotations(font_size=FONT_SIZE)


    if out_file:
        if out_file.endswith('.html'):
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)





def plot_token_embedding(past_key_values, key_or_value='key', layers_to_inspect=None, heads_to_inspect=None, out_file=None, token_idx=0, normalize=False, token_label=None):
    """
    Plots the embeddings of a token at a specific index in the input sequence.

    Args:
        past_key_values (list): A list of tensors representing the key-value cache at different time steps.
        key_or_value (str, optional): Specifies whether to plot the norms of the 'key' or 'value' vectors. Defaults to 'key'.
        layers_to_inspect (list, optional): A list of layer indices to inspect. Defaults to None, which inspects all layers.
        heads_to_inspect (list, optional): A list of head indices to inspect. Defaults to None, which inspects all heads.
        out_file (str, optional): The file path to save the plot. Defaults to None, which displays the plot without saving. If the file path ends with '.html', the plot is saved as an interactive HTML file. Otherwise, the plot is saved as an image file. Defaults to None.
        token_idx (int, optional): The index of the token in the input sequence. Defaults to 0, i.e. the bos token.
        normalize (bool, optional): Whether to normalize the embeddings between -1 and 1. Defaults to True.
    """

    if key_or_value == 'query':
        key_or_value = 'key'
        past_key_values = [[q] for q in past_key_values]

    key_or_value = 0 if key_or_value == 'key' else 1

    num_layers = len(past_key_values)
    num_heads = past_key_values[0][key_or_value].shape[1]

    layers_to_inspect = layers_to_inspect or list(range(0, num_layers))
    heads_to_inspect = heads_to_inspect or list(range(0, num_heads))

    fig = make_subplots(
        rows=len(layers_to_inspect),
        cols=len(heads_to_inspect),
        # subplot_titles=[f'Head {head}' for head in heads_to_inspect] * len(layers_to_inspect),
        vertical_spacing=0.01,
        horizontal_spacing=0.01
    )

    for l, layer in enumerate(layers_to_inspect):
        for i, head in enumerate(heads_to_inspect):
            token_emb = past_key_values[layer][key_or_value][0, head, token_idx, :].squeeze().detach().cpu().numpy()
            # color should be based on norm value
            # normalize the embedding between -1 and 1
            if normalize: 
                token_emb = np.abs(token_emb)
                token_emb = (token_emb - token_emb.min()) / (token_emb.max() - token_emb.min())
            p = go.Bar(x=list(range(token_emb.shape[0])), y=token_emb, marker=dict(color=token_emb, colorscale='Viridis'))
            # make line plot instead of bar plot
            #print('Layer:', layer, 'Head:', head)
            #print(bos_token_emb.shape)
            # p = go.Scatter(x=list(range(bos_token_emb.shape[0])), y=bos_token_emb, mode='lines+markers')
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False, showticklabels=True)
            fig.update_layout(plot_bgcolor='white')
            fig.add_trace(p, row=l + 1, col=i + 1)

            #if i == 0: fig.update_yaxes(title_text=f'Layer {layer}', row=l + 1, col=i + 1,  title_font_size=FONT_SIZE)

    fig.update_layout(
        height=200 * len(layers_to_inspect),
        width=400 * len(heads_to_inspect),
        # title_text=f'Embeddings of token with idx {token_idx}',
        title_text= token_label.replace('▁', ''),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )

    fig.update_annotations(font_size=FONT_SIZE)

    if out_file:
        if out_file.endswith('.html'):
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)


def plot_kv_cache_kurtosis(past_key_values, key_or_value='key', layers_to_inspect=None, heads_to_inspect=None, out_file=None, labels=None):
    """
    Plots the kurtosis of each token in the kv-cache.

    Args:
        past_key_values (list): A list of tensors representing the key-value cache at different time steps.
        key_or_value (str, optional): Specifies whether to plot the norms of the 'key' or 'value' vectors. Defaults to 'key'.
        layers_to_inspect (list, optional): A list of layer indices to inspect. Defaults to None, which inspects all layers.
        heads_to_inspect (list, optional): A list of head indices to inspect. Defaults to None, which inspects all heads.
        out_file (str, optional): The file path to save the plot. Defaults to None, which displays the plot without saving. If the file path ends with '.html', the plot is saved as an interactive HTML file. Otherwise, the plot is saved as an image file. Defaults to None.
        labels (list, optional): A list of labels for the x-axis, that should correspond to the tokens. Defaults to None.
    """

    key_or_value = 0 if key_or_value == 'key' else 1

    num_layers = len(past_key_values)
    num_heads = past_key_values[0][key_or_value].shape[1]

    layers_to_inspect = layers_to_inspect or list(range(0, num_layers))
    heads_to_inspect = heads_to_inspect or list(range(0, num_heads))

    if labels:
        labels = [l.strip('▁') for l in labels]
        labels_len = len(labels) if len(labels) < MAX_DISPLAY_LABELS else len(labels)//2

    fig = make_subplots(
        rows=len(layers_to_inspect),
        cols=len(heads_to_inspect),
        #subplot_titles=[f'Head {head}' for head in heads_to_inspect] + [""] * (len(layers_to_inspect)+1),
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        # subplots title font size

    )

    for l, layer in enumerate(layers_to_inspect):
        for i, head in enumerate(heads_to_inspect):
            tokens = past_key_values[layer][key_or_value][0, head, :, :].squeeze().detach().cpu().numpy()
            tokens_kurtosis = kurtosis(tokens, fisher=False, axis=1)
            p = go.Bar(x= labels or list(range(tokens_kurtosis.shape[0])), y=tokens_kurtosis) #, marker=dict(color=tokens_kurtosis, colorscale='Viridis'))
            #p = go.Histogram(x= labels or list(range(tokens_kurtosis.shape[0])), y=tokens_kurtosis, marker=dict(color=tokens_kurtosis, colorscale='Viridis'))
            # make line plot instead of bar plot
            #print('Layer:', layer, 'Head:', head)
            #print(bos_token_emb.shape)
            # p = go.Scatter(x=list(range(bos_token_emb.shape[0])), y=bos_token_emb, mode='lines+markers')
            if labels: fig.update_xaxes(tickangle=-TICKANGLE, tickmode='linear', nticks=labels_len, showgrid=False, tickfont=dict(size=FONT_SIZE))
            fig.update_layout( barmode='group')
            fig.update_layout(plot_bgcolor='white')
            # fig.update_yaxes(showgrid=False, showticklabels=False)

            fig.add_trace(p, row=l + 1, col=i + 1)

            if i == 0: fig.update_yaxes(title_text=f'Layer {layer}', row=l + 1, col=i + 1, title_font_size=FONT_SIZE)
            

    fig.update_layout(
        height=HEIGHT * len(layers_to_inspect),
        width=400 * len(heads_to_inspect),
        # title_text=f'Kurtosis of tokens in key-value cache',
        margin=dict(l=5, r=5, t=30, b=0),
        showlegend=False,
        # increse font for titles and axes

    )

    fig.update_annotations(font_size=FONT_SIZE)

    if out_file:
        if out_file.endswith('.html'):
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)



def plot_kv_cache_entropy(past_key_values, key_or_value='key', layers_to_inspect=None, heads_to_inspect=None, out_file=None,):
    """
    Plots the entropy of each token in the kv-cache.

    Args:
        past_key_values (list): A list of tensors representing the key-value cache at different time steps.
        key_or_value (str, optional): Specifies whether to plot the norms of the 'key' or 'value' vectors. Defaults to 'key'.
        layers_to_inspect (list, optional): A list of layer indices to inspect. Defaults to None, which inspects all layers.
        heads_to_inspect (list, optional): A list of head indices to inspect. Defaults to None, which inspects all heads.
        out_file (str, optional): The file path to save the plot. Defaults to None, which displays the plot without saving. If the file path ends with '.html', the plot is saved as an interactive HTML file. Otherwise, the plot is saved as an image file. Defaults to None.
    """

    key_or_value = 0 if key_or_value == 'key' else 1

    num_layers = len(past_key_values)
    num_heads = past_key_values[0][key_or_value].shape[1]

    layers_to_inspect = layers_to_inspect or list(range(0, num_layers))
    heads_to_inspect = heads_to_inspect or list(range(0, num_heads))

    fig = make_subplots(
        rows=len(layers_to_inspect),
        cols=len(heads_to_inspect),
        subplot_titles=[f'Head {head}' for head in heads_to_inspect] * len(layers_to_inspect),
        # vertical_spacing=0.05,
        # horizontal_spacing=0.05
    )

    for l, layer in enumerate(layers_to_inspect):
        for i, head in enumerate(heads_to_inspect):
            tokens = past_key_values[layer][key_or_value][0, head, :, :].squeeze().detach().cpu()

            # before computing entropy, normalize the tokens between 0 and 1
            tokens = (tokens - tokens.min()) / (tokens.max() - tokens.min())
            # add small value to avoid log(0)
            tokens_entropy = -1 * (tokens * (tokens+1e-9).log()).sum(axis=-1)
            p = go.Bar(x=list(range(tokens_entropy.shape[0])), y=tokens_entropy, marker=dict(color=tokens_entropy, colorscale='Viridis'))
            
            # make line plot instead of bar plot
            #print('Layer:', layer, 'Head:', head)
            #print(bos_token_emb.shape)
            # p = go.Scatter(x=list(range(bos_token_emb.shape[0])), y=bos_token_emb, mode='lines+markers')

            fig.add_trace(p, row=l + 1, col=i + 1)

            if i == 0: fig.update_yaxes(title_text=f'Layer {layer}', row=l + 1, col=i + 1, )

    fig.update_layout(
        height=400 * len(layers_to_inspect),
        width=400 * len(heads_to_inspect),
        # title_text=f'Entropy of tokens in key-value cache',
        showlegend=False
    )

    if out_file:
        if out_file.endswith('.html'):
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)
