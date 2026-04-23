"""Back up paragraphs.db to the private HF dataset daxmavy/attack-llm-judge-db.

Uploads two artifacts each run:
  - paragraphs.db         — consistent SQLite snapshot (online backup API, WAL-safe)
  - paragraphs.sql.gz     — gzip-compressed SQL text dump (portable restore path)

Both are timestamped in the commit message so HF's commit history gives you
automatic versioning.

WARNING: contains human-annotator research labels (paragraphs.human_mean_clarity,
human_agreement_score). Repo MUST stay private.

Usage:
  python3 scripts/backup_db.py                          # snapshot + upload now
  python3 scripts/backup_db.py --skip-sql-dump          # skip the .sql.gz dump
  python3 scripts/backup_db.py --repo daxmavy/other-db  # override target repo

Restore (local):
  huggingface_hub.snapshot_download("daxmavy/attack-llm-judge-db",
                                    repo_type="dataset",
                                    local_dir="/home/max/attack-llm-judge/data_restore")
"""
import argparse
import gzip
import os
import sqlite3
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
REPO_DEFAULT = "daxmavy/attack-llm-judge-db"


def consistent_snapshot(src_path: str, dst_path: str) -> None:
    """SQLite online backup — WAL-safe, works during concurrent writes."""
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    dst.close()
    src.close()


def sql_dump_gz(src_path: str, out_path: str) -> None:
    """.dump equivalent in Python, gzipped."""
    conn = sqlite3.connect(src_path)
    with gzip.open(out_path, "wt", compresslevel=6) as gz:
        for line in conn.iterdump():
            gz.write(line + "\n")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--skip-sql-dump", action="store_true")
    ap.add_argument("--private", action="store_true", default=True,
                    help="always private (contains human-annotator data)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set — source .env first")

    tmp_dir = Path("/tmp/db_backup")
    tmp_dir.mkdir(exist_ok=True)
    snap_path = tmp_dir / "paragraphs.db"
    sql_path = tmp_dir / "paragraphs.sql.gz"

    print(f"[{time.strftime('%H:%M:%S')}] taking consistent snapshot of {DB}...", flush=True)
    t0 = time.time()
    consistent_snapshot(DB, str(snap_path))
    print(f"  snapshot {snap_path.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s", flush=True)

    if not args.skip_sql_dump:
        print(f"[{time.strftime('%H:%M:%S')}] dumping SQL (gzip)...", flush=True)
        t0 = time.time()
        sql_dump_gz(str(snap_path), str(sql_path))
        print(f"  sql.gz {sql_path.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] ensuring repo {args.repo} exists (private)...", flush=True)
    api = HfApi(token=token)
    create_repo(args.repo, repo_type="dataset", token=token, exist_ok=True,
                private=args.private)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    c = sqlite3.connect(str(snap_path))
    paragraphs_rows = c.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
    rewrite_rows = c.execute("SELECT COUNT(*) FROM attack_rewrites").fetchone()[0]
    c.close()
    # Clean up any -shm/-wal files the read connection may have created;
    # those are transient and would confuse upload_folder's directory scan.
    for tmp in tmp_dir.glob("paragraphs.db-*"):
        tmp.unlink(missing_ok=True)
    commit_msg = (f"backup {stamp} — paragraphs={paragraphs_rows} "
                  f"attack_rewrites={rewrite_rows}")

    print(f"[{time.strftime('%H:%M:%S')}] uploading to {args.repo}: {commit_msg}", flush=True)
    t0 = time.time()
    upload_files = ["paragraphs.db"] + ([] if args.skip_sql_dump else ["paragraphs.sql.gz"])
    api.upload_folder(
        folder_path=str(tmp_dir),
        repo_id=args.repo,
        repo_type="dataset",
        token=token,
        commit_message=commit_msg,
        allow_patterns=upload_files,
    )
    print(f"  upload done in {time.time()-t0:.1f}s", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] backup complete", flush=True)


if __name__ == "__main__":
    main()
