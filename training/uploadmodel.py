from huggingface_hub import HfApi, HfFolder
import os
HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))
api = HfApi()

# Upload all the content from the local 'model_merged' folder to your remote repository.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="/data/solr/models/askdocsproject/checkpoints_4pm_oct3/checkpoint-49499",
    repo_id="vtiyyal1/llama2-7b-AskDocs52k",
    repo_type="model",
    create_pr=1,
)
