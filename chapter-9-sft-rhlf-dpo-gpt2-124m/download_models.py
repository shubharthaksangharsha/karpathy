import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"

BUCKET_NAME = "gpt2-ultrafineweb"

# -----------------------------
# Update these with the files you want
# -----------------------------

FILES_TO_DOWNLOAD = [
    # SFT
    ("sft_checkpoints/sft_best.pt", "sft_best.pt"),

    # DPO best
    ("dpo_custom/dpo_best.pt", "dpo_best.pt"),

    # Recommended step (change which one you want)
    ("dpo_custom/dpo_step_00040.pt", "dpo_step_00040.pt"),
    ("dpo_custom/dpo_step_00080.pt", "dpo_step_00080.pt"),
    ("dpo_custom/dpo_step_00120.pt", "dpo_step_00120.pt"),
]

# -----------------------------
# Download Function
# -----------------------------

def download_from_gcs(blob_path, local_path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        print(f"❌ File not found in bucket: {blob_path}")
        return

    print(f"⬇️ Downloading: {blob_path} → {local_path}")
    blob.download_to_filename(local_path)
    print(f"✅ Saved: {local_path}")

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    for blob_path, local_path in FILES_TO_DOWNLOAD:
        download_from_gcs(blob_path, local_path)

