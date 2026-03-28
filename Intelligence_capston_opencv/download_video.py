import kagglehub

# Download latest version
path = kagglehub.dataset_download("dpamgautam/video-file-for-lane-detection-project")

print("Path to dataset files:", path)