from google.cloud import storage
import os

# -----------------------
# CONFIG
# -----------------------
LOCAL_FILE = "qa-sft_best.pt"          # The file YOU want to upload
BUCKET_NAME = "gpt2-ultrafineweb"            # Your bucket name
DEST_PATH = "qa-sft_best.pt"  # Path in bucket (folder optional)

# Make sure key is set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"

def upload_file():
    print("🔄 Connecting to GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    print(f"⬆️ Uploading {LOCAL_FILE} → gs://{BUCKET_NAME}/{DEST_PATH}")
    blob = bucket.blob(DEST_PATH)

    # Upload
    blob.upload_from_filename(LOCAL_FILE)

    print("✅ Upload complete!")

if __name__ == "__main__":
    upload_file()

