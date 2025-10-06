# Deprecation Notice

## GPU Stabilizer Feature Removed

**Date**: December 2024

The GPU Stabilizer feature (advanced_gpu_stabilizer.py) has been completely removed from the MyXTTS project as it was not needed for current requirements and added unnecessary complexity.

### What was removed:
- `optimization/advanced_gpu_stabilizer.py` module
- `--enable-gpu-stabilizer` and `--disable-gpu-stabilizer` command line flags
- All GPU stabilizer initialization and usage code from trainer
- All GPU stabilizer references from documentation and scripts

### Impact:
- Training now uses standard GPU usage without the stabilizer layer
- All training scripts work without GPU stabilizer flags
- Documentation has been updated to reflect the changes

### Archived Files:
Files in the `archive/summaries` directory may still reference GPU Stabilizer. These references are historical and the feature is no longer available in the codebase.

For standard GPU optimization, use the `--optimization-level enhanced` flag instead.
