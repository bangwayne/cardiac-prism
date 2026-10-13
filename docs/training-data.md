# Training data and fine-tuning

The released Cardiac-Prism checkpoint segments the **left ventricular cavity
(LV), right ventricular cavity (RV), and left ventricular myocardium (MYO) in
cardiac cine MRI**. It is a cardiac fine-tune of
[K-Prism](https://github.com/bangwayne/K-Prism), whose upstream implementation
supports semantic, interactive, and reference-based segmentation across multiple
medical imaging datasets and modalities.

## Datasets used for the released checkpoint

Supervision used expert annotations at **end diastole (ED)** and **end systole
(ES)**. A phase volume is one annotated 3D image at ED or ES; it is not a fully
annotated cine across every time point.

| Dataset | View | Training subjects | Annotated training phase volumes | Source and access |
| --- | --- | --- | --- | --- |
| ACDC | Short axis | 100 | 200 | [Official ACDC database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |
| M&Ms (MM) | Short axis | 320 | 640 | [Official M&Ms dataset](https://www.ub.edu/mnms/data.html) and [access instructions](https://www.ub.edu/mnms/download.php) |
| M&Ms-2 (MnM2) | Short axis and four chamber | 360 | 1,440: 720 short axis + 720 four chamber | [Official M&Ms-2 dataset](https://www.ub.edu/mnms-2/data.html) |
| **Total** | | **780** | **2,280** | |

Each M&Ms-2 subject contributes both views and is counted once in the subject
total. The training index contains **18,070 2D image planes** and **54,210
slice × structure targets**, including slices where a queried structure is
absent. All supervised masks are expert annotations; workstation predictions,
cached priors, and model-derived pseudo-labels were not used as ground truth.

### Split policy

- **ACDC:** patient001–100 supplied fine-tuning images and references.
  patient101–150 supplied the separate test partition: 50 subjects, 100 ED/ES
  volumes, and 1,076 2D planes.
- **M&Ms:** all 320 available annotated subjects in the public cohort were pooled
  for fine-tuning, including its original training, validation, and test
  partitions. The 25 unannotated subjects were excluded. These 320 subjects are
  not the full 375-subject challenge cohort.
- **M&Ms-2:** all 360 subjects, both short-axis and four-chamber ED/ES annotations,
  were pooled for fine-tuning.

There was no separate fine-tuning validation set. The released checkpoint is the
final epoch-10 checkpoint; ACDC test results were not used to select an epoch.
Because M&Ms and M&Ms-2 partitions were pooled, results on their original test
partitions would not be held-out results for this checkpoint. The initialization
checkpoint's complete pretraining history is not established, so the ACDC split
does not prove that those subjects were unseen in every earlier training stage.

Kaggle cardiac data and Rescan were considered for optional external evaluation
and **were not used in this fine-tuning run**. Kaggle does not supply dense
segmentation ground truth; the Rescan masks reviewed for this project were
model-derived.

## Label mapping and preprocessing

| Structure | Workstation / ACDC label | Original M&Ms / M&Ms-2 label | Model query |
| --- | --- | --- | --- |
| Background | 0 | 0 | — |
| RV cavity | 1 | 3 | 16 |
| LV myocardium (MYO) | 2 | 2 | 17 |
| LV cavity | 3 | 1 | 18 |

M&Ms and M&Ms-2 labels were remapped once into the workstation convention.
Images and annotations were paired and checked for matching geometry. Training
retained all annotated acquisition layers, including empty structure targets;
an absent structure in an annotated slice is distinct from a missing annotation.

The released model uses `mm_slice_p0_p98_256_v1`: per-slice p0–p98 clipping and
normalization, a 256 × 256 preparation grid, 0–255 frame scaling, and a 512 × 512
model input normalized with mean 0.26 and standard deviation 0.30. See the
[model card](model-card.md#input-processing) and
[inference preprocessing implementation](../server/cardiac_model.py) for the
exact sequence and interpolation rules. Images may have different native
in-plane dimensions; exported masks retain the imported image's geometry.

## Recorded fine-tuning recipe

| Setting | Released cardiac checkpoint |
| --- | --- |
| Backbone / segmentation architecture | K-Prism mixture of experts |
| Training duration | 10 epochs; final epoch released |
| Optimizer | AdamW; initial learning rate 1e-5; weight decay 0 |
| Learning-rate schedule | Cosine; final / initial learning-rate ratio 0.1 |
| Batch size | 3 per GPU across 4 GPUs; global batch 12 |
| Precision | FP32 |
| Interactive training | Three modes with sampling probabilities 0.3 / 0.5 / 0.2; three interaction rounds |
| References | One annotated reference per training sample when the mode uses a reference |
| Sampling | Dataset-balanced weighted sampling with replacement |
| Seed | 0 |
| Gradient clipping | Norm 3.0 |
| Prepared data manifest SHA-256 | `26ef1af0950d75e6b0e9fac4b64a50662c42f2a7a94f68bfdd7f85d0c389e308` |

The training manifest hash identifies the recorded fine-tuning corpus; the full
datasets and patient-level manifest are not bundled with the workstation. The
[model card](model-card.md) records the released checkpoint's identity and hash.
Installation tests verify inference and persistence; they are separate from
medical accuracy evaluation.

## Fine-tuning and contributing new integrations

Research fine-tuning on additional annotated datasets is welcome. Start with the
[K-Prism training implementation](https://github.com/bangwayne/K-Prism#training)
and its dataset/model configurations. This repository supplies the workstation,
inference adapter, and cardiac checkpoint; it does not provide a one-command
reproduction package for the fine-tuning run described above.

For a model integration, document the dataset source and split, annotation
provenance, structure/query mapping, preprocessing, checkpoint version, and
evaluation. Match the workstation adapter to the trained model's input and
output contract. Additional organs or imaging modalities require corresponding
controls and validation alongside the new weights. Preserve the original image
geometry and frame/layer lock behavior.

Open an [issue](https://github.com/bangwayne/cardiac-prism/issues) to discuss an
extension, then submit a pull request to **`develop`** following
[Contributing](../CONTRIBUTING.md). Model or dataset files should have documented
distribution terms and a versioned asset profile; do not attach clinical studies
to a public pull request.

## Data attribution

Obtain the training datasets from their providers and follow their access,
attribution, and usage terms. The workstation's software/model license does not
replace any dataset license. The installer downloads only the documented
[ACDC example](sample-data.md), not the full training datasets; patient001 belongs
to the fine-tuning training partition and is a workflow example.

- **ACDC:** O. Bernard et al., “Deep Learning Techniques for Automatic MRI Cardiac
  Multi-structures Segmentation and Diagnosis: Is the Problem Solved?”,
  *IEEE Transactions on Medical Imaging*, 37(11), 2514–2525, 2018.
  [DOI](https://doi.org/10.1109/TMI.2018.2837502).
- **M&Ms:** V. M. Campello et al., “Multi-Centre, Multi-Vendor and Multi-Disease
  Cardiac Segmentation: The M&Ms Challenge”, *IEEE Transactions on Medical
  Imaging*, 40(12), 3543–3554, 2021. See the
  [official challenge website](https://www.ub.edu/mnms/) for dataset attribution
  and access requirements.
- **M&Ms-2:** follow the attribution and access requirements on the
  [official challenge website](https://www.ub.edu/mnms-2/).

For the segmentation methods, please cite **K-Prism (ICLR 2026)** and
**VerSe (IPMI 2025)**. References and BibTeX are in the
[README](../README.md#acknowledgments-and-citation).
