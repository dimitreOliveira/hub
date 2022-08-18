import tensorflow as tf
from tensorflow_hub.keras_layer import KerasLayer
from huggingface_hub import snapshot_download, create_repo, HfApi, HfFolder


def pull_from_hub(repo_id,
                  local_files_only=False,
                  allow_regex=None,
                  ignore_regex=None,
                  trainable=False,
):    
    """Downloads a repository from the HuggingFace Hub.
    
    Args:
        repo_id: A user or an organization name and a repo name separated by a / (usually follows 'username/repository_name).
        local_files_only:  If True, avoid downloading the file and return the path to the local cached file if it exists.
        allow_regex: Regex pattern that can be specified to select a subset of files to be included.
        ignore_regex: Regex pattern that can be specified to select a subset of files to be excluded.
        trainable: If True the model's weights won't be frozen (fine-tunable).
    """
    model_path = snapshot_download(repo_id=repo_id,
                                   local_files_only=local_files_only,
                                   allow_regex=allow_regex,
                                   ignore_regex=ignore_regex,
                                )
    return KerasLayer(handle=model_path, 
                      trainable=trainable,
                    )

def push_to_hub(model,
                repo_id,
                folder_path,
                path_in_repo="./",
                token=None,
):
    """Persist a model to the local file system then uploads it to the HuggingFace Hub, 
    it will create a new repository if there isn't one already. In this case where we have models 
    saved with "saved model" format the model will be a set of files and folders.
    
    Args:
        model: Keras model that will be persisted and uploaded to HF Hub.
        repo_id: A user or an organization name and a repo name separated by a / (usually follows 'username/repository_name).
        folder_path: Path to the folder to upload on the local file system.
        path_in_repo: Relative path of the directory in the repo, for example: "checkpoints/1fec34a/results". Will default to the root folder of the repository.
        token: An authentication token (See https://huggingface.co/settings/token).
    """
  
    if token is None:
        token = HfFolder.get_token()

    if token is None:
        raise ValueError(
            "You must login to the Hugging Face hub on this computer by typing"
            " `huggingface-cli login` and entering your credentials, "
            " if you are on a notebook environment you may use the `notebook_login` function "
            " Alternatively, you can pass your own token as the `token` argument."
        )
  
    # Persist model to local file system
    tf.saved_model.save(obj=model, 
                        export_dir=folder_path)
    
    # Create HF Hub repository if it doesn't exists
    create_repo(repo_id=repo_id,
                token=token,
                exist_ok=True)

    # Upload all relevant model files to HF Hub
    HfApi().upload_folder(repo_id=repo_id,
                          folder_path=folder_path,
                          path_in_repo=path_in_repo,
                          token=token
                        )