from huggingface_hub import HfApi, HfFolder
import os
HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))
api = HfApi()

# Upload all the content from the local 'model_merged' folder to your remote repository.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="/content/outputs/checkpoint-4000",
    repo_id="vtiyyal1/gemma-7b-it-AskDocsEmpathy4k",
    repo_type="model",
    create_pr=1,
)



"""
from huggingface_hub import HfApi, HfFolder

# Authenticate with your token (ensure you keep your token secure)
api = HfApi()
token = HfFolder.get_token()
api.login(token=token)

# Create a new repository on the Hugging Face Hub
api.create_repo(
    token=token,
    name="gemma-7b-it-AskDocsEmpathy4k",
    organization="vtiyyal1",  # Use None if it's under your personal account
    private=False  # Set to True for a private repository
)

# Now upload your model files to the repository
api.upload_folder(
    folder_path="/content/outputs/checkpoint-5000",
    repo_id="vtiyyal1/gemma-7b-it-AskDocsEmpathy4k",
    repo_type="model",
    token=token,
    create_pr=1,
)

"""