import { ChevronRight } from 'lucide-react';

/** Describe the native-file contract, rather than the internal model tensor. */
export default function InputFormatGuide({ backupEnabled, dicomEnabled = false }: { backupEnabled: boolean; dicomEnabled?: boolean }) {
  return <details className="input-format-guide" data-testid="input-format-guide">
    <summary><ChevronRight size={15} aria-hidden="true" /><span>Input format guide <small>3D / 4D images and matching masks</small></span></summary>
    <div className="input-format-content">
      <p>Open a scalar NIfTI image (<code>.nii</code> or <code>.nii.gz</code>): one intensity value per voxel.</p>
      <table aria-label="NIfTI image dimensions">
        <thead><tr><th scope="col">Image</th><th scope="col">File dimensions</th><th scope="col">Example</th></tr></thead>
        <tbody>
          <tr><th scope="row">3D volume</th><td><code>(X, Y, Z)</code><span>A stack of layers at one time</span></td><td><code>192 × 320 × 12</code></td></tr>
          <tr><th scope="row">4D cine</th><td><code>(X, Y, Z, T)</code><span>A 3D volume for each time frame</span></td><td><code>192 × 320 × 12 × 30</code></td></tr>
        </tbody>
      </table>
      <p><strong>X and Y</strong> are image width and height; <strong>Z</strong> is the layer count; <strong>T</strong> is the time-frame count. In a 4D NIfTI, time is the last file axis. The example cine has 12 layers in each of 30 frames.</p>
      <p><strong>Width and height may differ; neither is fixed to 256 or 512.</strong> These examples are illustrative. Keep the original image and spatial header. The model prepares and normalizes its input automatically and returns masks at the original image resolution; you do not need to resize or rescale the source image.</p>
      {dicomEnabled && <section aria-label="DICOM folder format"><h3>Open a DICOM folder</h3>
        <p>Choose <strong>Open DICOM folder</strong> and select the folder containing <strong>one classic MR image sequence</strong>. Files may have a <code>.dcm</code> extension or no extension. Width, height, layers and time frames are detected automatically; no manual resizing is needed.</p>
        <p>The folder must contain a complete 3D or 4D stack with consistent image dimensions and orientation. Choose a more specific folder if it contains multiple sequences. Missing, repeated or incompatible slices are reported before a study is created.</p>
        <p>This first version accepts uncompressed classic MR images. Enhanced multi-frame and compressed DICOM need conversion to NIfTI first. A DICOM ZIP cannot be opened as a Full study backup.</p>
        <p>DICOM folders: up to <strong>1,000 files, 512 MB in total and 128 million voxels</strong>.</p>
        <p>Frame order comes from DICOM time information. If the original frame interval is unknown, playback uses an adjustable default speed; MRI repetition time (TR) is never used as the cine frame interval.</p>
      </section>}
      <section aria-label="Matching annotation format">
        <h3>Already have a mask?</h3>
        <p>Open the source image first, then import the mask in the annotation panel. Its dimensions, voxel spacing, origin and orientation must match the source image.</p>
        <p>Use labels <strong>0 = background · 1 = RV · 2 = MYO · 3 = LV</strong>, or a binary <code>0 / 1</code> mask and choose its structure when importing.</p>
        <p>A 4D image needs a matching <strong>4D mask with the same time frames</strong>. A 3D mask cannot currently be imported into a 4D study.</p>
      </section>
      {backupEnabled && <section aria-label="Study backup format"><h3>Continue a saved study</h3>
        <p>Open a <strong>Full study backup .zip</strong> exported by this workstation to restore the image, masks, clicks, notes, slice locks and operation history. A ZIP containing separate masks is an annotation export, rather than a full study backup.</p>
      </section>}
      <p className="input-format-limits">NIfTI files: up to 512 MB each and 128 million voxels per study{backupEnabled ? ' · Full study backups: up to 2 GB' : ''}. RGB/vector images and 5D files are unsupported.</p>
    </div>
  </details>;
}
