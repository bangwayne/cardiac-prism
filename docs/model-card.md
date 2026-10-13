# Cardiac-Prism segmentation model

## Release identity

| Field | Value |
| --- | --- |
| Model | Cardiac-Prism cardiac segmentation v0.1 |
| Architecture | K-Prism mixture-of-experts interactive segmentation |
| Checkpoint | `weight/cardiac-prism-v0.1.pth` |
| Profile | `weight/cardiac-prism-v0.1.json` |
| Workstation release | `v0.1.2` |
| File size | 178,498,401 bytes |
| SHA-256 | `2ba5f7c3a33e798ec5533184e0fe0fed7af4e565f0a5e8ae3bfb3a15a5f0e703` |
| License | [PolyForm Noncommercial 1.0.0](../LICENSE) |

The release distributes the cardiac fine-tuned checkpoint used by this workstation.
Its model tensors are unchanged; packaging removes private training-path metadata.
Version 0.1.1 standardizes the checkpoint and profile filenames; the model remains
version 0.1 and its checkpoint bytes and SHA-256 are unchanged from release 0.1.0.
Keep the JSON profile next to the checkpoint; the profile selects the matching input
transforms and query IDs. The download script checks the full checkpoint hash before
installing it. The `main` branch and `v0.1.2` source tag include the checkpoint through
Git LFS, so a normal clone with Git LFS installed retrieves both source and model.
The versioned [asset manifest](../assets/manifest.json) also identifies its GitHub
Release download. No Hugging Face account or model host is required.
The architecture is packaged as `cardiac_prism/`; its upstream K-Prism
attribution and license remain intact.

## Intended use

Research annotation and interactive review of **cardiac cine MRI**. The current
checkpoint and workstation support the right ventricular cavity (RV), left
ventricular myocardium (MYO), and left ventricular cavity (LV). The training data
include short-axis images and M&Ms-2 four-chamber images. It supports automatic
segmentation, positive and negative clicks, and reference-based propagation.
The workstation supports native 3D volumes
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
[K-Prism](https://github.com/bangwayne/K-Prism). Fine-tuning used expert ED/ES
annotations from **ACDC, M&Ms, and M&Ms-2**: 780 training subjects, 2,280 annotated
volumes, and 18,070 2D planes. The M&Ms-2 subset includes both short-axis and
four-chamber views. ACDC patient101–150 were excluded from this fine-tuning run;
all available annotated M&Ms and M&Ms-2 partitions were pooled for training.

The final epoch was used without selecting an epoch on the ACDC test set.
The initialization checkpoint's complete pretraining history is not established,
so this split does not establish that ACDC test subjects were unseen during all
earlier training. See [training data and fine-tuning](training-data.md) for the
dataset sources, counts, split policy, label conversion, and recorded recipe.

The release provides inference code, weights, and preprocessing configuration.
The training datasets are obtained separately under their own terms. The
workstation package is focused on inference and annotation; the
[upstream K-Prism repository](https://github.com/bangwayne/K-Prism#training)
provides the general training implementation and configurations. Fine-tuning on
additional datasets and contributing compatible model integrations are welcome.
New structures or modalities require corresponding adapter, label, profile, and
interface changes and evaluation.

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

The downloaded checkpoint is normally already present after cloning. The commands
above verify it and recover a missing file. The repository and checkpoint are
public; a GitHub account or token is not required for HTTPS downloads. If the
Release URL is unavailable, the installer retrieves the same verified file from
the immutable `v0.1.2` source
tag using **Git LFS** and your clone's `origin`. No separate
model branch is required. Earlier `models/v0.1.x` tags are retained for existing
installations. No token is written into the asset manifest. For Docker deployment,
run asset setup on the host before starting the container.

For a server without Internet access, download the checkpoint from the
[v0.1.2 release](https://github.com/bangwayne/cardiac-prism/releases/tag/v0.1.2) on another
machine, copy it to the server, and run:

```bash
python3 scripts/download_assets.py --weights-file /path/to/cardiac-prism-v0.1.pth
```

The same size and SHA-256 checks apply to offline installation. The script preserves
an existing installed file if a replacement download fails verification.

## Citation

Please cite **K-Prism (ICLR 2026)** and **VerSe (IPMI 2025)** when using this
workstation in research. Full references and BibTeX are in the
[repository citation section](../README.md#acknowledgments-and-citation), with
machine-readable metadata in [CITATION.cff](../CITATION.cff). Cite the relevant
datasets separately when using their images or annotations.
