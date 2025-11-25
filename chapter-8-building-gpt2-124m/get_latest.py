import os 
from google.cloud import storage 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"
def download_latest_checkpoint(bucket_name, prefix="checkpoints", local_dir="checkpoints"):
    """
    Download only the latest checkpoint (.pt) file from GCS into local_dir.
    Skips download if same file already exists locally with the same size.
    Returns: local file path or None if not found.
    """
    os.makedirs(local_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # list blobs inside the prefix folder
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".pt")]

    if not blobs:
        print("❌ No checkpoints found in GCS.")
        return None

    # sort based on filename (model_00001.pt → latest is last)
    blobs.sort(key=lambda b: b.name)

    latest_blob = blobs[-1]
    filename = latest_blob.name.split("/")[-1]
    local_path = os.path.join(local_dir, filename)

    # If file already exists AND size matches, skip
    if os.path.exists(local_path) and os.path.getsize(local_path) == latest_blob.size:
        print(f"✅ Latest checkpoint already exists locally: {filename}")
        return local_path

    print(f"⬇️ Downloading latest checkpoint from GCS:\n  {latest_blob.name} → {local_path}")
    latest_blob.download_to_filename(local_path)

    print(f"✅ Download complete: {local_path}")
    return local_path


latest_ckpt = download_latest_checkpoint(bucket_name="gpt2-ultrafineweb")
