import os
from google.cloud import storage

# ------------------------------------------------------------------
# CONFIG – change these if needed
# ------------------------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"

BUCKET_NAME = "gpt2-ultrafineweb"
#GCS_PATH = "dpo_checkpoints/dpo_final.pt"
GCS_PATH = "qa-sft_best.pt"


LOCAL_DIR = "qa_sft_checkpoints"
LOCAL_PATH = f"{LOCAL_DIR}/qa-sft_best.pt"

# ------------------------------------------------------------------
# Download function
# ------------------------------------------------------------------
def download_from_gcs(bucket_name, gcs_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"⬇️ Downloading gs://{bucket_name}/{gcs_path}")
    blob.download_to_filename(local_path)
    print(f"✅ Saved to {local_path}")


if __name__ == "__main__":
    download_from_gcs(BUCKET_NAME, GCS_PATH, LOCAL_PATH)

