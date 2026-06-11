# Baseline Environment Notes

Dataset: `D:\mask\baseline_runs\patch_dataset`

Working training environment:

- `D:\anaconda\envs\yolo11\python.exe`
- PyTorch 2.10.0 + CUDA 13.0
- GPU available: NVIDIA GeForce RTX 5070 Ti
- Used for all completed baseline runs.

Completed baselines:

- `unet`
- `fpn`
- `aspp`
- `tv_deeplab_mnv3`
- `tv_lraspp_mnv3`
- `segformer_tiny`
- `hrnet_lite`
- `ocr_lite`

Official MMSeg status:

- `D:\mm\env_mmseg` has `mmseg 1.2.2`, `mmengine 0.11.0rc2`, `mmcv-lite 2.1.0`.
- Import works when `KMP_DUPLICATE_LIB_OK=TRUE` is set.
- Training is not usable on RTX 5070 Ti because this environment uses PyTorch 2.5.1 / CUDA 11.8, which does not support `sm_120`.
- `mmcv-lite` also lacks compiled `mmcv._ext`, so Mask2Former-style ops are unavailable.

Official Mask2Former status:

- `D:\mask2former_proj\env_m2f` has PyTorch 2.5.1 but no `detectron2`.
- `D:\mask2former_proj\env` does not currently import PyTorch correctly.
- A full Mask2Former run requires a Detectron2 build compatible with the active PyTorch/CUDA stack.

Recommended next environment step:

- Use the working `yolo11` CUDA 13 environment as the base.
- Install or build compatible `mmengine`, `mmcv`, `mmsegmentation`, and Detectron2 against PyTorch/CUDA that supports RTX 5070 Ti.
- Then rerun official SegFormer/HRNet/OCRNet/Mask2Former configs for publication-grade comparison.
