from huggingface_hub import list_repo_files

repo_id = "friedrichor/MSR-VTT"
files = list_repo_files(repo_id=repo_id, repo_type="dataset")

print("Files in repo:")
for f in files:
    print(f)
