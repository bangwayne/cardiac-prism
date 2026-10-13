# Cardiac segmentation model

## Release identity

| Field | Value |
| --- | --- |
| Model | Cardiac Prism cardiac segmentation v0.1 |
| Architecture | K-Prism mixture-of-experts interactive segmentation |
| Checkpoint | `weight/cardiac_achilleas_epoch10.pth` |
| Profile | `weight/cardiac_achilleas_epoch10.json` |
| Release | `v0.1.0` |
| File size | 178,498,401 bytes |
| SHA-256 | `2ba5f7c3a33e798ec5533184e0fe0fed7af4e565f0a5e8ae3bfb3a15a5f0e703` |
| License | [PolyForm Noncommercial 1.0.0](../LICENSE) |

The release distributes the cardiac fine-tuned checkpoint used by this workstation.
Its model tensors are unchanged; packaging removes private training-path metadata.
The original filename is retained to preserve compatibility with existing configuration.
Keep the JSON profile next to the checkpoint; the profile selects the matching input
transforms and query IDs. The download script checks the full checkpoint hash before
installing it. Obtain the checkpoint from the release
linked in the versioned [asset manifest](../assets/manifest.json).

## Intended use

Research annotation and interactive review of short-axis cardiac MR images. The model
predicts the right ventricular cavity (RV), left ventricular myocardium (MYO), and left
ventricular cavity (LV). It supports automatic segmentation, positive and negative
clicks, and reference-based propagation. The workstation supports native 3D volumes
and 4D cine images, but the model operates on individual 2D acquisition slices.

| Export label | Structure | Model query |
| --- | --- | --- |
| 0 | Background | — |
| 1 | RV | 16 |
| 2 | MYO | 17 |
| 3 | LV | 18 |

A click is attached to one time frame and one acquisition layer. Propagation and batch
segmentation are explicit actions. When segmenting all structures, the highest
eligible probability assigns each pixel to a single label, with a background
threshold of 0.5. Manual editing and local locks remain available for review.

## Input processing

Input images may have different in-plane dimensions; they do not need to be 512 × 512.
The model uses the versioned `mm_slice_p0_p98_256_v1` recipe:

1. For each original `(t, z)` plane, calculate its minimum and 98th percentile, clip
   intensities to that interval, and scale using `(I - p0) / (p98 - p0 + 0.0001)`.
2. Resample the image planes to the trained 256 × 256 grid using the original spacing.
3. Map the current time frame to the range 0–255.
4. Resize query images to 512 × 512 using float bilinear interpolation, then convert
   to `uint8`. Reference images use the corresponding trained `uint8`/OpenCV path.
5. Normalize using mean 0.26 and standard deviation 0.30 after dividing by 255.

The resulting model tensor range is approximately −0.87 to 2.47. The implementation
rejects non-finite or invalid prepared inputs. Probability maps are restored to the
native image grid; viewing window settings do not change model preprocessing. Native
image geometry and exported masks remain in the imported image's coordinate system.

The precise implementation is in [server/cardiac_model.py](../server/cardiac_model.py).
Use that implementation and the release profile together rather than substituting a
different resize or normalization sequence.

## Training provenance and evaluation scope

This is the project's epoch-10 cardiac fine-tuned checkpoint, based on
[K-Prism](https://github.com/bangwayne/K-Prism). The release provides inference code,
weights and preprocessing configuration. It does not include the original private
training corpus or a complete training reproduction package.

Installation and release checks verify model loading, supported inference paths,
result persistence and consistency with the supplied checkpoint. They do not establish
accuracy across scanners, acquisition protocols or clinical populations. The ACDC
sample is supplied for trying the workflow; it is not an independent benchmark of
this checkpoint, and no held-out status is claimed for it.

## Limitations

- Predictions require review. Positive or negative clicks are prompts, not a guarantee
  of a particular change in overlap metrics.
- A 2D model does not enforce full spatial or temporal consistency across a cine scan.
- Intensity normalization adapts numerical scale but does not remove differences in
  anatomy, acquisition protocol, image artifacts or disease distribution.
- The release is for noncommercial research use and has not been validated as a
  clinical diagnostic device.

## Download and offline installation

```bash
python3 scripts/download_assets.py --weights
python3 scripts/download_assets.py --weights --verify
```

While the repository is private, the release URL requires project access. If that
URL is unavailable anonymously, the installer retrieves the same verified file from
the repository's versioned `models/v0.1.0` tag using **Git LFS** and the Git credentials for
your clone's `origin`. Clone through an authorized GitHub account, install Git LFS,
and keep that account's SSH or Git credential configuration available when running
the installer. No token is written into the asset manifest. For Docker deployment,
run asset setup on the host before starting the container.

For a server without Internet access, download the checkpoint from the
[v0.1.0 release](https://github.com/bangwayne/cardiac-prism/releases/tag/v0.1.0) on another
machine, copy it to the server, and run:

```bash
python3 scripts/download_assets.py --weights-file /path/to/cardiac_achilleas_epoch10.pth
```

The same size and SHA-256 checks apply to offline installation. The script preserves
an existing installed file if a replacement download fails verification.
