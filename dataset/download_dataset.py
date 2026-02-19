import os
import tarfile
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

# =====================================================
# CONFIGURATION
# =====================================================

# Enable hf_transfer for faster downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# HuggingFace dataset repository
repo_id = "miaoweiwang/BiMotion"

# Number of parallel workers (increase if network is fast)
MAX_WORKERS = 8

# =====================================================
# PROCESS A SINGLE FILE
# =====================================================

def process_file(filename, download_dir, extract_dir):
    """
    Download -> Extract -> Delete archive
    """
    try:
        print(f"⬇️ Downloading: {filename}")

        # 1. Download file to custom directory
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=download_dir,
            local_dir_use_symlinks=False,  # Use real files instead of symlinks
        )

        # 2. Determine category folder
        if "DT4D" in filename:
            category = "DT4D"
        elif "ObjaverseV1" in filename:
            category = "ObjaverseV1"
        elif "ObjaverseXL" in filename:
            category = "ObjaverseXL"
        else:
            category = "Others"

        # Create extraction folder
        extract_path = os.path.join(extract_dir, category)
        os.makedirs(extract_path, exist_ok=True)

        # 3. Extract tar.gz file
        print(f"📦 Extracting: {filename}")
        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

        # 4. Delete compressed file after extraction
        os.remove(local_path)
        print(f"🗑️ Deleted: {filename}")

        print(f"✅ Done: {filename}")
        return filename

    except Exception as e:
        print(f"❌ Failed {filename}: {e}")
        return None


def download_captions_folder(base_dir):
    """
    Download the captions folder from the dataset repository
    """
    print("📝 Downloading captions folder...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="bspline_motion_dataset/captions/*",
        local_dir=base_dir,
        local_dir_use_symlinks=False,
    )
    print("✅ Captions folder downloaded")


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    # 1️⃣ Read command-line argument
    if len(sys.argv) < 2:
        print("Usage: python download.py /path/to/base_dir")
        sys.exit(1)

    BASE_DIR = sys.argv[1]

    # 2️⃣ Set directories
    DOWNLOAD_DIR = os.path.join(BASE_DIR, "tmp_download")
    EXTRACT_DIR = os.path.join(BASE_DIR, "bspline_motion_dataset")

    # 3️⃣ Create necessary directories
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # 4️⃣ Fetch file list
    print("📡 Fetching file list from HuggingFace...")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    tar_files = [f for f in all_files if f.endswith(".tar.gz")]
    print(f"📁 Found {len(tar_files)} archive files")
    print(f"⚡ Using {MAX_WORKERS} parallel workers")
    """
    # 5️⃣ Parallel download and extraction
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda f: process_file(f, DOWNLOAD_DIR, EXTRACT_DIR), tar_files))

    # 6️⃣ Count successful downloads
    success = len([r for r in results if r is not None])
    print("\n==============================")
    print(f"🎉 Finished: {success}/{len(tar_files)} succeeded")
    print("===============================")

    # 7️⃣ Clean temporary directory
    print(f"🧹 Cleaning up temporary directory: {DOWNLOAD_DIR}")
    try:
        if os.path.exists(DOWNLOAD_DIR):
            shutil.rmtree(DOWNLOAD_DIR)
            print("✨ Temporary directory removed successfully.")
    except Exception as e:
        print(f"⚠️ Could not remove temp directory: {e}")
    print("===============================")
    """
    # 8️⃣ Download captions folder
    download_captions_folder(BASE_DIR)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
