import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"

def delete_sft_checkpoints(local_dir="checkpoints"):
    """
    Delete all local checkpoint files that start with 'sft_'.
    """
    if not os.path.exists(local_dir):
        return

    removed = 0
    for fname in os.listdir(local_dir):
        if fname.startswith("sft_") and fname.endswith(".pt"):
            fpath = os.path.join(local_dir, fname)
            os.remove(fpath)
            removed += 1
            print(f"🗑️ Deleted SFT checkpoint: {fname}")

    if removed == 0:
        print("ℹ️ No SFT checkpoints found to delete.")
    else:
        print(f"✅ Deleted {removed} SFT checkpoints.")

def download_latest_checkpoint(bucket_name, prefix="checkpoints", local_dir="checkpoints"):
    """
    Download only the latest NON-SFT checkpoint (.pt) file from GCS.
    """
    os.makedirs(local_dir, exist_ok=True)

    # delete local SFT checkpoints
    delete_sft_checkpoints(local_dir)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # list blobs inside the prefix folder
    blobs = [
        b for b in bucket.list_blobs(prefix=prefix)
        if b.name.endswith(".pt") and not os.path.basename(b.name).startswith("sft_")
    ]

    if not blobs:
        print("❌ No NON-SFT checkpoints found in GCS.")
        return None

    # sort based on filename (model_00001.pt → latest is last)
    blobs.sort(key=lambda b: b.name)

    latest_blob = blobs[-1]
    filename = os.path.basename(latest_blob.name)
    local_path = os.path.join(local_dir, filename)

    # If file already exists AND size matches, skip
    if os.path.exists(local_path) and os.path.getsize(local_path) == latest_blob.size:
        print(f"✅ Latest checkpoint already exists locally: {filename}")
        return local_path

    print(f"⬇️ Downloading latest NON-SFT checkpoint:\n  {latest_blob.name} → {local_path}")
    latest_blob.download_to_filename(local_path)

    print(f"✅ Download complete: {local_path}")
    return local_path


# Run it
latest_ckpt = download_latest_checkpoint(bucket_name="gpt2-ultrafineweb")
