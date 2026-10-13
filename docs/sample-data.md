# ACDC example data

The optional examples come directly from the official
[Automated Cardiac Diagnosis Challenge (ACDC) repository](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
The installer downloads patient 001's original cine image, end-diastolic volume and
end-diastolic reference segmentation. These files are unmodified; no resizing,
intensity normalization, mask prediction or synthetic temporal interpolation is
applied to the downloaded data.

```bash
python3 scripts/download_assets.py --sample
python3 scripts/download_assets.py --sample --verify
```

After starting the workstation, open the library and select the ACDC cine example.
It contains **30 real time frames and 10 acquisition layers**, with native NIfTI
dimensions **216 × 256 × 10 × 30** in `(X, Y, Z, T)` order. Select a time frame or use
the player to inspect motion. Use **Segment all time frames** to generate masks for
the cine, or segment one frame or layer first. These calculations use the same model
and controls as a user-imported study.

The separate 3D example is the official end-diastolic image, with dimensions
216 × 256 × 10. Its expert reference mask is available for comparison. The cine is
not assigned an invented 30-frame reference mask: the distributed ACDC annotations
cover selected cardiac phases, not every time point.

| Installed path | Contents |
| --- | --- |
| `demo-data/images/patient001_4d.nii.gz` | Original 30-frame cine |
| `demo-data/images/patient001_frame01.nii.gz` | Original end-diastolic 3D image |
| `demo-data/references/patient001_frame01_gt.nii.gz` | Expert end-diastolic labels |

The spatial spacing is 1.5625 × 1.5625 × 10 mm. The cine header does not specify a
physical time unit; playback speed is a viewing preference, not a measurement of
the patient's heart rate. Reference labels are 0 = background, 1 = RV, 2 = MYO and
3 = LV, matching the workstation's label IDs.

## License and citation

ACDC data are separate from the workstation's software and model license. The
official terms specify **CC BY-NC-SA 4.0**, with use restricted to **noncommercial
scientific research**, and require the ACDC paper citation. Read the preserved
[official license terms](../licenses/ACDC-LICENSE-TERMS.md) and
[mandatory citation](../licenses/ACDC-MANDATORY-CITATION.md) before using the examples.
The upstream license text was obtained from the official Human Heart Project
[license file](https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/file/66e290e0961576b1bad4ee3d/download).

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., “Deep Learning Techniques
for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the
Problem Solved?”, *IEEE Transactions on Medical Imaging*, 37(11), 2514–2525, 2018.
[DOI: 10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).

## Reproducible downloads

The [asset manifest](../assets/manifest.json) pins official file URLs, byte sizes
and SHA-256 hashes. The download script verifies existing files before skipping
them, downloads into temporary files, and only installs a file after it passes
verification. Incomplete or invalid downloads do not replace an installed file.

Images are downloaded at setup time and are not embedded in the Git repository or
container image. For offline use, copy the three files into the paths above, then
run `python3 scripts/download_assets.py --sample --verify`. The provenance metadata
and license documents are already part of the repository.
