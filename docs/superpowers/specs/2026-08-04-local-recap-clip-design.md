# Local ReCap-CLIP Loading Design

## Goal

Load the ReCap-CLIP model and tokenizer entirely from `weights/recap-clip` during Mosaic3D inference, without Hugging Face Hub identifiers, cache layout, or network access.

## Local Bundle Contract

The configured directory must contain:

- `open_clip_config.json`
- `open_clip_pytorch_model.bin`
- `added_tokens.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

`configuration.json` may remain in the directory but is not mandatory.

## Architecture

`MosaicConfig.text_model_path` stores a project-relative directory. A focused local loader reads `open_clip_config.json`, constructs `open_clip.CLIP` from its `model_cfg`, loads `open_clip_pytorch_model.bin` through OpenCLIP's checkpoint loader, and constructs `open_clip.HFTokenizer` from the same local directory. `local_files_only=True` prevents Transformers from resolving the original `bert-base-uncased` tokenizer name over the network.

The Mosaic3D adapter receives a ready model whose `text_tokenizer` attribute preserves the existing text-feature interface. No image preprocessing is constructed because SceneCompose only calls `encode_text` on ReCap-CLIP.

## Validation And Errors

Preflight verifies the local directory and each required file before workers start. The loader also validates the JSON structure and reports malformed `model_cfg`, missing text configuration, or checkpoint incompatibility at the point of use.

Changing `text_model_path` remains part of the Mosaic3D stage digest. The Mosaic3D checkpoint and every required ReCap-CLIP file also contribute size and modification time to the stage fingerprint, so replacing any local model artifact invalidates resumed results.

## Documentation

Both READMEs and the technical implementation document describe the flat local directory layout. Hugging Face cache environment variables are no longer required for ReCap-CLIP.

## Verification Scope

Per project instructions, verification is limited to Python AST/import-independent checks and configuration/document consistency. Model downloads and GPU inference are performed by the user on the server.
