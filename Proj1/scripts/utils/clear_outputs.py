"""
Clear outputs script: Remove old clustering analysis outputs.

This script safely removes all output files from the clustering analysis folder
before re-running the pipeline. Run this before re-executing clustering_pipeline.py
and cluster_analyzer.py to start fresh.

Folders deleted:
- reports/ (txt files)
- plots/ (png files)
- models/ (pkl files and npy arrays)
- data/ (csv files)

The Proj1/reports/clustering_analysis/ folder will be preserved but emptied.
"""

import pathlib
import shutil


def clear_outputs(confirm=True):
    """Clear all output files from clustering analysis folder."""
    output_dir = pathlib.Path(__file__).resolve().parents[2] / "reports" / "clustering_analysis"
    
    if not output_dir.exists():
        print(f"ℹ️  No outputs folder found at: {output_dir}")
        return
    
    # Find subdirectories to delete
    subdirs = ['reports', 'plots', 'models', 'data']
    dirs_to_delete = [output_dir / subdir for subdir in subdirs if (output_dir / subdir).exists()]
    
    if not dirs_to_delete:
        print(f"✅ Outputs folder is already clean: {output_dir}")
        return
    
    # Count total files
    total_files = sum(len(list(d.rglob('*'))) for d in dirs_to_delete)
    
    print(f"\n{'=' * 80}")
    print(f"CLEAR CLUSTERING OUTPUTS")
    print(f"{'=' * 80}")
    print(f"\n📁 Outputs folder: {output_dir}")
    print(f"\n📋 Folders to delete ({len(dirs_to_delete)}):")
    for subdir in dirs_to_delete:
        files_in_dir = list(subdir.rglob('*'))
        print(f"   - {subdir.name}/ ({len(files_in_dir)} items)")
        for f in sorted(list(subdir.iterdir()))[:3]:  # Show first 3 items
            print(f"      • {f.name}")
        if len(files_in_dir) > 3:
            print(f"      ... and {len(files_in_dir) - 3} more")
    
    if confirm:
        response = input(f"\n⚠️  Delete {total_files} files across {len(dirs_to_delete)} folders? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Cancelled. No files deleted.")
            return
    
    # Delete directories
    deleted_count = 0
    for subdir in dirs_to_delete:
        try:
            shutil.rmtree(subdir)
            deleted_count += 1
        except Exception as e:
            print(f"⚠️  Failed to delete {subdir.name}/: {e}")
    
    print(f"\n✅ Deleted {deleted_count}/{len(dirs_to_delete)} folder(s)")
    print(f"\n💡 Ready to re-run:")
    print(f"   python src/models/clustering_pipeline.py")
    print(f"   python scripts/analysis/cluster_analyzer.py")
    print()


if __name__ == "__main__":
    clear_outputs(confirm=True)
