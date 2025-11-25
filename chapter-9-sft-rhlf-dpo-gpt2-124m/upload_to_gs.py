from google.cloud import storage
import os

# -----------------------------------------------------
# CONFIG — change if needed
# -----------------------------------------------------
BUCKET_NAME = "gpt2-ultrafineweb"
DATA_FOLDER = "data"
DEST_PREFIX = "datasets/sft/"  # folder inside GCS bucket
FILES = ["sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"]
# -----------------------------------------------------

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

for file in FILES:
    local_path = os.path.join(DATA_FOLDER, file)
    blob_path = f"{DEST_PREFIX}{file}"

    print(f"🚀 Uploading {local_path} → gs://{BUCKET_NAME}/{blob_path}")
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

print("\n✅ All dataset files uploaded to Google Cloud Storage")
print(f"🔗 gs://{BUCKET_NAME}/{DEST_PREFIX}")
