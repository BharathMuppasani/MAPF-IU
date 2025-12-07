#!/usr/bin/env python3
"""
Script to clean git merge conflict markers from JSON files.

This script finds all JSON files with merge conflict markers and either:
1. Keeps the HEAD version (your changes)
2. Keeps the incoming version (other branch)
3. Lists files for manual review

Usage:
    python scripts/clean_merge_conflicts.py --list          # List corrupted files
    python scripts/clean_merge_conflicts.py --clean-head    # Keep HEAD version
    python scripts/clean_merge_conflicts.py --clean-theirs  # Keep incoming version
    python scripts/clean_merge_conflicts.py --delete        # Delete corrupted files
"""

import argparse
import re
from pathlib import Path


def find_conflicted_files(base_dir: Path) -> list:
    """Find all JSON files with git merge conflict markers."""
    conflicted = []

    # Search in logs directory
    logs_dir = base_dir / 'logs'
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return conflicted

    for json_file in logs_dir.rglob('*.json'):
        try:
            content = json_file.read_text()
            if '<<<<<<<' in content or '=======' in content or '>>>>>>>' in content:
                conflicted.append(json_file)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return conflicted


def clean_file_keep_head(file_path: Path) -> bool:
    """
    Clean a file by keeping the HEAD (current branch) version.

    Conflict format:
    <<<<<<< HEAD:path/to/file
    HEAD content here
    =======
    Incoming content here
    >>>>>>> branch:path/to/file
    """
    try:
        content = file_path.read_text()

        # Pattern to match conflict blocks and keep HEAD version
        # This handles the case where conflict markers span multiple lines
        pattern = r'<<<<<<<[^\n]*\n(.*?)=======\n.*?>>>>>>>[^\n]*\n'

        cleaned = re.sub(pattern, r'\1', content, flags=re.DOTALL)

        # Check if cleaning was successful (valid JSON structure)
        if '<<<<<<<' in cleaned or '=======' in cleaned or '>>>>>>>' in cleaned:
            print(f"  Warning: Could not fully clean {file_path}")
            return False

        file_path.write_text(cleaned)
        return True

    except Exception as e:
        print(f"  Error cleaning {file_path}: {e}")
        return False


def clean_file_keep_theirs(file_path: Path) -> bool:
    """
    Clean a file by keeping the incoming (theirs) version.
    """
    try:
        content = file_path.read_text()

        # Pattern to match conflict blocks and keep incoming version
        pattern = r'<<<<<<<[^\n]*\n.*?=======\n(.*?)>>>>>>>[^\n]*\n'

        cleaned = re.sub(pattern, r'\1', content, flags=re.DOTALL)

        if '<<<<<<<' in cleaned or '=======' in cleaned or '>>>>>>>' in cleaned:
            print(f"  Warning: Could not fully clean {file_path}")
            return False

        file_path.write_text(cleaned)
        return True

    except Exception as e:
        print(f"  Error cleaning {file_path}: {e}")
        return False


def delete_file(file_path: Path) -> bool:
    """Delete a corrupted file."""
    try:
        file_path.unlink()
        return True
    except Exception as e:
        print(f"  Error deleting {file_path}: {e}")
        return False


def validate_json(file_path: Path) -> bool:
    """Check if a file is valid JSON."""
    import json
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Clean git merge conflict markers from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clean_merge_conflicts.py --list          # List all corrupted files
  python scripts/clean_merge_conflicts.py --clean-head    # Keep your version (HEAD)
  python scripts/clean_merge_conflicts.py --clean-theirs  # Keep incoming version
  python scripts/clean_merge_conflicts.py --delete        # Delete corrupted files
        """
    )

    parser.add_argument('--list', action='store_true',
                       help='List all files with merge conflicts')
    parser.add_argument('--clean-head', action='store_true',
                       help='Clean files by keeping HEAD version')
    parser.add_argument('--clean-theirs', action='store_true',
                       help='Clean files by keeping incoming version')
    parser.add_argument('--delete', action='store_true',
                       help='Delete all corrupted files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate JSON files after cleaning')

    args = parser.parse_args()

    # Get base directory
    base_dir = Path(__file__).parent.parent

    print("Scanning for files with merge conflicts...")
    conflicted_files = find_conflicted_files(base_dir)

    if not conflicted_files:
        print("No files with merge conflicts found!")
        return

    print(f"\nFound {len(conflicted_files)} files with merge conflicts:\n")

    # Group by directory for better display
    by_dir = {}
    for f in conflicted_files:
        parent = f.parent.relative_to(base_dir)
        if parent not in by_dir:
            by_dir[parent] = []
        by_dir[parent].append(f.name)

    for dir_path, files in sorted(by_dir.items()):
        print(f"  {dir_path}/")
        for fname in sorted(files):
            print(f"    - {fname}")

    if args.list:
        # Just list, don't do anything
        print(f"\nTotal: {len(conflicted_files)} corrupted files")
        return

    if args.clean_head:
        print("\nCleaning files (keeping HEAD version)...")
        success = 0
        failed = 0
        for f in conflicted_files:
            if clean_file_keep_head(f):
                print(f"  Cleaned: {f.relative_to(base_dir)}")
                success += 1
            else:
                failed += 1
        print(f"\nCleaned {success} files, {failed} failed")

    elif args.clean_theirs:
        print("\nCleaning files (keeping incoming version)...")
        success = 0
        failed = 0
        for f in conflicted_files:
            if clean_file_keep_theirs(f):
                print(f"  Cleaned: {f.relative_to(base_dir)}")
                success += 1
            else:
                failed += 1
        print(f"\nCleaned {success} files, {failed} failed")

    elif args.delete:
        print("\nDeleting corrupted files...")
        confirm = input(f"Are you sure you want to delete {len(conflicted_files)} files? (yes/no): ")
        if confirm.lower() == 'yes':
            success = 0
            for f in conflicted_files:
                if delete_file(f):
                    print(f"  Deleted: {f.relative_to(base_dir)}")
                    success += 1
            print(f"\nDeleted {success} files")
        else:
            print("Aborted.")
    else:
        print("\nNo action specified. Use --list, --clean-head, --clean-theirs, or --delete")

    # Validate if requested
    if args.validate and (args.clean_head or args.clean_theirs):
        print("\nValidating cleaned files...")
        valid = 0
        invalid = 0
        for f in conflicted_files:
            if f.exists():
                if validate_json(f):
                    valid += 1
                else:
                    print(f"  Invalid JSON: {f.relative_to(base_dir)}")
                    invalid += 1
        print(f"\nValidation: {valid} valid, {invalid} invalid")


if __name__ == '__main__':
    main()
