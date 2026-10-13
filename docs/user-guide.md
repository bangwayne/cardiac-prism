# User guide

Cardiac Prism combines image viewing, automated segmentation, and manual annotation in one browser window. This guide describes the controls in the released workstation. Install and connect using the [installation guide](install.md).

## Start with the Library

**Library** contains example scans and saved studies. An example is an input image available for opening; a study is your working copy with masks, clicks, notes, locks, and edit history. Opening an example starts or reopens its study. The original example remains an input rather than becoming your annotation storage.

- **Open your NIfTI volume** accepts `.nii` and `.nii.gz` files, or a **Full study backup** `.zip` created by this workstation.
- **Browse folder** lists NIfTI images in a folder selected on your computer. Select an image to import; this is not a general folder or PACS import.
- **Open DICOM folder** imports one supported MR sequence from a selected folder.
- **My studies** contains saved working studies. Search matches study names and notes.
- **Completed** shows studies you marked complete. Completion is a workflow status, not a guarantee about the annotation.

**Complete** saves the current annotation and note before marking a case complete. **Reopen** returns it to work in progress. **Delete** removes the selected saved study, including its server-side working image copy, masks, prompts, priors, and notes; it does not delete the original file on your computer. Export a full backup before deleting work you may need later.

Removing an example hides that Library entry. **Removed examples** lets you restore it or permanently remove the entry. This does not delete separately saved studies created from that example.

## Input formats

### NIfTI images

Use a scalar volume with positive spacing and valid spatial geometry:

| Input | File dimensions | Example |
| --- | --- | --- |
| 3D image | `(X, Y, Z)` | `192 × 320 × 12`: one volume with 12 layers |
| 4D cine | `(X, Y, Z, T)` | `192 × 320 × 12 × 30`: 30 time frames, each containing 12 layers |

The first two dimensions may differ and do not need to be 256 or 512. Do not resize your original image to fit the model. The workstation prepares model inputs internally and maps predictions back to the original voxel grid. The final dimension of a 4D file must represent time, not a color channel or another spatial axis.

The current import limits are 512 MiB per NIfTI upload and 128 million voxels per study. Vector/color images, invalid geometry, coupled spatial/time directions, and unsupported dimensionalities are rejected.

**Image geometry** shows dimensions, spacing, input preparation, and detected intensity issues. Window/level adjusts display only; it does not change the source image or the model's preprocessing. Automatic intensity scaling cannot guarantee model accuracy on an unfamiliar acquisition protocol.

### A simple DICOM folder

1. Prepare a folder containing **one uncompressed classic MR series**. Files can have a `.dcm` extension or no extension.
2. Click **Open DICOM folder** and select it. Accept the browser's folder-upload prompt if shown.
3. Wait for the spatial/time checks and conversion. The imported study opens automatically.

The importer supports 3D and 4D series and identifies layers from their spatial positions and time frames from suitable DICOM timing metadata. It does not sort slices only by filename. It rejects mixed series or orientations, duplicate locations/phases, and incomplete identified frame/layer grids. It accepts up to 1,000 files, 512 MiB total, and 128 million voxels.

Enhanced multi-frame DICOM, compressed transfer syntaxes, DICOM ZIP archives, multi-series projects, and PACS connections are not supported in this release. If only phase numbering is available, playback remains possible but acquisition timing is reported as unknown; the chosen display FPS is not a measured heart rate. MR repetition time is not used as the cine frame interval.

### External masks

Open the source image first, then choose **Import annotation**. Select either:

- **Editable annotation** to replace the working masks on unlocked slices; or
- **Reference comparison** to overlay a comparison annotation without replacing the working masks.

The NIfTI annotation must match the source's dimensions, origin, spacing, and direction. A 4D source requires a matching 4D annotation. Matching width and height alone is insufficient, and the importer does not silently resample a misaligned annotation.

Choose the label format:

| Value | Structure |
| --- | --- |
| `0` | Background |
| `1` | RV: right ventricular cavity |
| `2` | MYO: myocardium |
| `3` | LV: left ventricular cavity |

Alternatively, use **Binary mask → RV / MYO / LV** for a single-structure binary annotation. A 3D ED/ES ground-truth file from a public dataset should be opened with its matching 3D phase image, rather than attached directly to a full 4D cine.

## Understand the four views

The first view is the **acquisition plane**, where clicking, brushing, and erasing operate. Two reconstruction views show the orthogonal directions in the scan grid. The fourth view is the **3D locator**, with X/Y/Z axes and a shared crosshair position.

All views share the same location and time cursor. Use **Locate** to click in a slice view and move the common position. The current acquisition layer follows that position. On oblique scans, reconstruction titles describe scan-grid directions; they need not be anatomical coronal and sagittal planes.

Use **Pan**, **Zoom**, and **Window** for navigation and display. **Fit** resets the view to the image extent. Expand a view with its corner control and collapse it to return to the four-view layout. The expanded view uses the same cursor as the other views.

The left display panel controls:

- Each structure's visibility and color-coded selection.
- **Mask opacity** for the 2D overlay.
- **Locator color** and **Locator opacity** for the crosshair.
- **Point opacity** for positive/negative click markers. Hiding markers does not delete prompts.
- **Reference contours** when a reference annotation has been imported.
- **3D mask opacity** and **Smooth 3D surfaces** for a generated 3D model.

The default structure colors are RV coral, MYO green, and LV purple. DICOM and NIfTI use the same palette and material settings.

## Browse a 4D cine

For a 4D study, the **CINE** controls provide **Play / Pause**, previous/next frame, a time slider, a requested FPS, and looping. Frame and layer numbers shown in the UI start at 1.

Playback synchronizes the image and existing mask for each frame. It does not copy a mask from one frame to another, generate a prediction, or propagate prompts. If only one frame has been segmented, the others remain unsegmented until you process or import them.

Pause before editing. Manually moving the time slider pauses playback. The requested FPS may exceed what your browser, network, or study size can sustain; the viewer reports actual playback performance.

### Render a 3D mask

Pause at the desired frame and click **Render 3D** in the locator view. This creates surfaces from that frame's existing RV / MYO / LV masks. **Smooth 3D surfaces** changes the surface appearance without changing any annotation pixels.

**Hide 3D** releases the surfaces. Starting playback, changing time frame, or editing a mask also removes them so the displayed surface cannot become stale. Pause and click **Render 3D** again when you want another surface. The locator axes remain available during ordinary 2D viewing.

## Segment and refine

First select **RV**, **MYO**, or **LV** in the structure list. The selected structure is shown in the editing panel.

| Control | Scope and effect |
| --- | --- |
| **Segment this slice** | Initializes the selected organ on the current frame and acquisition layer. |
| **Segment all** | Predicts RV, MYO, and LV on the current frame/layer; highest probability resolves overlaps among editable structures. |
| **Segment all layers** | Predicts all three structures on unlocked layers of the current time frame. |
| **Segment all time frames** | Predicts all three structures on unlocked frame/layer positions throughout the cine. |
| **Include** | Adds a positive point inside a region that should belong to the selected organ. |
| **Exclude** | Adds a negative point in an unwanted region. `Shift + click` while using Include also excludes. |
| **Brush** | Paints the selected organ in the current acquisition slice. |
| **Erase** | Removes selected-organ pixels in the current acquisition slice. |

**Clicks operate only on the current `(time frame, acquisition layer, organ)`**. They do not spread to neighboring layers or other time frames. A click submits a new prediction; wait for it to finish before evaluating the result. Predicted changes are not guaranteed to improve every case.

The default **Organ + clicks** mode combines the organ query and point prompts. **Clicks only** uses point prompts without the organ query. Changing modes starts a new prompt history for the next click. Brush and erase changes become the working annotation for later refinement.

The smallest brush/eraser size is **one original image pixel**, not one browser-screen pixel. Zoom in to work precisely. Painting an organ gives that selected organ priority at painted pixels and removes overlapping pixels from other **unlocked** structures. A locked structure remains protected; unlock it if an intentional replacement is required.

Whole-study jobs show preparation and per-GPU progress. The first job on a GPU takes extra time to load its model. Batch predictions are applied together when the job succeeds, so no partial result may be visible during computation. **Cancel** preserves the annotation from before the job. One **Undo** reverts the committed batch.

### Clear or resolve annotations

| Control | Effect |
| --- | --- |
| **Clear points** | Removes the selected organ's prompts on the current frame/layer. |
| **Clear this slice** | Clears the selected organ on the current frame/layer. |
| **Clear all masks this slice** | Clears RV, MYO, and LV on the current frame/layer. |
| **Clear all masks this layer** | Clears RV, MYO, and LV at the current layer across unlocked time frames. For a 3D study, this is the current slice. |
| **Resolve · prioritize [organ]** | Gives the selected organ priority when overlapping masks are detected on the current slice. |

All-mask clearing requires the organ locks to permit the operation. Slice locks remain protected. Undo restores an eligible clearing operation.

## Protect accepted work with locks

The smallest slice lock protects **all three organs at one time frame and one acquisition layer**. A blank slice can also be locked to preserve an accepted empty annotation.

To lock the current position, open **Edit History** and click **Lock slice**. To select a range:

1. Enter **Select / Lock**.
2. Select the desired numbers in **FRAMES** and **LAYERS**. `Shift + click` selects a continuous range.
3. Read the displayed frame/layer scope.
4. Choose **Lock selected** or **Unlock selected**.

The selection is the combination of selected frames and selected layers. For example, Frames 2–4 and Layers 3 and 5 select six slices. **All layers** deliberately selects whole volumes for the selected frames; **All frames** selects the chosen layers throughout the cine.

A lock prevents changes to mask pixels, clicks, and prediction context at its positions. Automatic segmentation, temporal propagation, imports, baseline reloads, and clearing skip protected positions. **Replace manual edits** cannot override a slice lock. Undo/redo also cannot alter a newly locked position; unlock it first if you intend to restore that annotation.

You can still browse, play, export, render, or use a locked slice as a propagation reference. Lock metadata is saved separately, but any other pending annotation changes still require **Save study**.

The structure list also has **organ locks**. These protect an entire organ independently of the local frame/layer locks. Visibility controls only affect drawing and do not protect masks.

## Propagate within a layer

Temporal propagation uses accepted masks from one layer to predict that same layer at other time frames. It does not propagate to other layers.

1. Pause and choose the acquisition layer to work on.
2. Correct its mask in one or more frames.
3. Expand **PROPAGATE IN TIME** beneath **Edit History**.
4. Select **1–3 reference frames**, or use **Use current frame**.
5. Select the target structures and review the target/skip summary.
6. Leave **Replace manual edits in target frames** off to retain manual corrections. Enable it only when you want unlocked manual edits replaced.
7. Click **Propagate to N frames** and review the completed result through the cine.

The reference image and working mask are frozen at the start of the job. Predictions do not become new references during that job. A missing organ in a reference is ignored for that organ; if every reference lacks it, that organ is skipped. Reference comparison overlays are not propagation inputs. Locks, reference positions, and other layers remain protected.

One Undo restores the entire completed propagation. Save afterward. Temporal propagation provides a starting point for review and can require further corrections.

## History, notes, and completion

**Edit History** can be collapsed with its arrow. Frame/layer tiles distinguish **Empty**, **Mask present**, and **Edited**. A lock icon indicates protection; partially locked frames show how many layers are protected.

Expand **Activity log** to review textual actions. **Current slice** filters to the current position; **Whole study** includes the study-wide history. Entries describe successful clicks, brush/erase operations, segmentation, reference selection and propagation results, imports, locks, clearing, and undo/redo. **Load older activity** retrieves older records. The activity log is a review record and is separate from the session undo stack.

Use **STUDY NOTE** for observations or a handoff message. Notes belong to the entire study. They save after a short pause in typing; check **Note saved** or click **Save note**. A concurrent-note conflict keeps your draft available for an explicit decision rather than silently replacing it.

**Annotation complete** or **Complete** marks a study as reviewed and moves it into the completed workflow. Use **Reopen** when more work is needed. Completion does not mean all frames were segmented, and it is not a substitute for frame/layer locks.

## Save, export, and restore

Click **Save study** to save working masks, prompts, and prediction context on the server. Confirm that **Unsaved edits** is cleared before leaving. Saved studies are available in **My studies** after a restart.

| Export format | Contains | Use it for |
| --- | --- | --- |
| **Full study backup** `.zip` | Source image, masks, reference annotation when present, notes, locks, activity records, prompts, and refinement context | Transfer or restore the complete working study |
| **Separate masks + edit record** `.zip` | One binary NIfTI per organ and an annotation record | Analysis in another tool; this is not a full-study restore archive |
| **Combined labelmap** `.nii.gz` | One integer labelmap with values 0–3 | External tools that expect one segmentation volume |
| **RV / MYO / LV binary mask** `.nii.gz` | The chosen structure's mask | Single-structure analysis |

Exports retain the source spatial geometry and time dimensions. Combined labelmap export is refused if independent masks overlap; resolve overlaps or export separate masks. Full-study ZIP restore is performed through the Library's main upload entry, not **Import annotation**.

A saved workspace or full backup preserves the current annotation state. **Undo and redo are session-only and do not survive a server restart or backup import.** Use a full backup to keep a durable checkpoint before major changes or deletion.

## Keyboard shortcuts

Shortcuts apply when focus is outside a text field or menu.

| Key | Action |
| --- | --- |
| `1` / `2` | Include / Exclude |
| `3` | Locate |
| `4` / `5` | Brush / Erase |
| `6` / `7` / `8` | Pan / Zoom / Window |
| `↑` / `↓` | Previous / next acquisition layer |
| `R` | Fit views |
| `Space` | Play / pause cine |
| `Ctrl+S` or `⌘S` | Save study |
| `Ctrl+Z` or `⌘Z` | Undo |
| `Ctrl+Shift+Z`, `⌘Shift+Z`, or `Ctrl+Y` | Redo |
| `Esc` | Pause playback or close the current dialog |

If a control is disabled, check playback, an active task, completion status, and the relevant slice or organ lock. Connection and installation problems are covered in [Troubleshooting](install.md#troubleshooting).
