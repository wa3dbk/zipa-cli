"""zipa-cli: batch phonetic decoding for ZIPA zipformer phone-recognition models.

A command-line tool to run greedy phonetic decoding with the ZIPA family of
zipformer models (https://aclanthology.org/2025.acl-long.961/) over many input
formats — single files, file lists, directories, CommonVoice-style TSVs, STM
segment files, HuggingFace datasets, and lhotse/icefall manifests — using either
the minimal-dependency ONNX backend (default) or the full PyTorch backend.

See ``zipa_cli/README.md`` for usage and examples.
"""

__version__ = "0.1.0"
