# Data Source

This package does not redistribute the Zandbergen-Landini EVA transcription file `ZL3b-n.txt` because it is third-party source data.

To reproduce the analyses, obtain the Zandbergen-Landini EVA transcription (ZL3b) from the original source and place the file at:

```text
data/raw/ZL3b-n.txt
```

The analysis scripts are written to expect this path. The `data/raw/` directory is retained in the repository with `.gitkeep` so that users can place the transcription file in the correct location after obtaining it.

The section metadata and processed outputs included in this package are retained for reproducibility documentation and inspection, but regeneration from raw text requires the external transcription file.
