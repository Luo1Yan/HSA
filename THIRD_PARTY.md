# Data and encoder provenance

HSA uses frozen pretrained encoders and public multimodal datasets. The original
data preparation was based on [CMCL](https://github.com/Xiaohao-Liu/CMCL), with
additional adapters for the datasets and protocols in this repository. The public
HSA entry points share one encoder interface and one task-input builder.

## Encoders

- [ImageBind](https://github.com/facebookresearch/ImageBind), fixed source commit
  `53680b02d7e37b19b124fa37bae4b6c98c38f5be`, with the official ImageBind-Huge weights.
- [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind), fixed source commit
  `7070c53375661cdb235801176b564b45f96f0648`. The exact Hugging Face checkpoint
  revisions are recorded in `configs/downloads.json`.
- The CMCL-derived LanguageBind protocols use Video_FT, Audio_FT, Image, Thermal
  and Depth checkpoints. Their text branch uses the Depth checkpoint. MSR-VTT
  retains its registered Video/Audio checkpoints and the Audio text branch.

Encoder source archives are downloaded directly from their official repositories
with their original licenses. Checkpoint and dataset terms continue to apply.
Compatibility adapters in `extract.py` handle the removed CLIP attention-mask
helper, torchvision imports and the modern PEFT parameter naming convention.
They preserve frozen checkpoint parameters.

## Data preparation

- CMCL supplies the provenance for the frozen encoder workflow, original UCF101
  and NYUv2 collections, and the basic image/text/audio/depth/thermal transforms.
- The HSA adapters preserve RGB middle frames, inverse-depth conversion,
  dataset-specific temporal windows, inertial channel order and official or
  registered sample partitions.
- [PMRL](https://github.com/Xiaohao-Liu/PMRL) supplies the original-data mirror
  used to acquire MSR-VTT. This project downloads original data and creates its
  features locally.
- The 80 text templates follow the OpenAI CLIP templates distributed with
  [OpenCLIP](https://github.com/mlfoundations/open_clip); official NYUv2 classification
  retains the depth-specific variants from LanguageBind's `d_cls/zero_shot_metadata.py`.

See the README data-source table for the dataset authors' original pages. Please
cite the original datasets and encoders when using their artifacts.

## Reproducibility details

Two pinned extraction environments retain the original and later classification
model-loading conventions. A fixed seed per observation now controls upstream
random video flips independently of execution order. Historical random flip
states were not recorded for all video collections; equality with those specific
cached video vectors remains unverified. The full validation scope is stated in
the README and is distinct from the availability of an adapter.
