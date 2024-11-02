from huggingface_hub import snapshot_download

snapshot_download(repo_id='yaofu/llama-2-7b-80k',
                  local_dir='../models/llama-2-7b-80k',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)

snapshot_download(repo_id="Yukang/Llama-2-7b-longlora-32k-ft",
                  local_dir='../models/llama-2-7b-longlora-32k-ft',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)

snapshot_download(repo_id='meta-llama/Llama-2-7b-hf',
                  local_dir='./models/llama-2-7b',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  token="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",
                  resume_download=True)