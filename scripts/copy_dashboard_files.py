import os
import shutil

source_dir="/Users/hardik/Downloads/v2_feats"
dest_dir="/Users/hardik/coding/streamlit/blank-app/data/non_canonical_dashboards"


def copy_files(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.html'):
                rel_path = os.path.relpath(root, src)
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, rel_path, file)

                os.makedirs(os.path.dirname(dst_file), exist_ok=True)

                # Copy file if it doesn't exist, skip if it does
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {dst_file}")
                else:
                    print(f"Skipped (already exists): {dst_file}")


if __name__ == '__main__':
    copy_files(source_dir, dest_dir)