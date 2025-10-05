from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your Streamlit app files
    repo_id="Suunil-Dabral/tourism_project",  # the target Hugging Face Space repo
    repo_type="space",                            # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)
