import hashlib
from pathlib import Path, PurePosixPath


def upload_run(directory, bucket_name, prefix):
    from google.cloud import storage

    directory = Path(directory).resolve()
    if not (directory / "run.json").is_file():
        raise ValueError("Upload requires an experiment directory containing run.json")
    if not bucket_name or "/" in bucket_name or not prefix.strip("/"):
        raise ValueError("Supply a bare bucket name and a nonempty backup prefix")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uploaded = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = str(PurePosixPath(prefix.strip("/")) / path.relative_to(directory).as_posix())
        blob = bucket.blob(name)
        blob.metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        blob.upload_from_filename(str(path), if_generation_match=0)
        uploaded.append(name)
    return {"bucket": bucket_name, "objects": uploaded,
            "note": "Create-only backup: use a new prefix per upload. No bucket is made public."}
