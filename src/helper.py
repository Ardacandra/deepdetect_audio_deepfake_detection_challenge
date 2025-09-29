import os
from collections import Counter, defaultdict

def get_subfolders(path):
    subfolders = []
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            subfolders.append(os.path.relpath(os.path.join(root, d), path))
    return sorted(subfolders)

def get_file_stats(folder_path):
    file_prefixes = []
    file_extensions = []
    file_count = 0

    for root, _, files in os.walk(folder_path):
        for f in files:
            file_count += 1

            name, ext = os.path.splitext(f)
            prefix = ''.join(ch for ch in name if not ch.isdigit())
            ext = ext.lower().lstrip(".")

            file_prefixes.append(prefix if prefix else "NO_PREFIX")
            file_extensions.append(ext if ext else "NO_EXT")


    return file_count, Counter(file_prefixes), Counter(file_extensions)