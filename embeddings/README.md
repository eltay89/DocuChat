# Custom Embedding Models

This folder is designed to store custom embedding models for DocuChat. You can place your own embedding models here and reference them using the `--embedding_model` argument.

## Supported Model Types

### 1. Sentence Transformers Models
Place any sentence-transformers compatible model in this folder. The model should be in the standard format with:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer.json` and related tokenizer files
- `modules.json`

### 2. Hugging Face Models
You can also use any Hugging Face model identifier directly without downloading.

## Usage Examples

### Using a local model from this folder:
```bash
python docuchat.py --embedding_model ./embeddings/my-custom-model
```

### Using a Hugging Face model identifier:
```bash
python docuchat.py --embedding_model sentence-transformers/all-mpnet-base-v2
```

### Using the default model:
```bash
python docuchat.py --embedding_model all-MiniLM-L6-v2
```

## Popular Embedding Models

Here are some popular embedding models you might want to try:

- `all-MiniLM-L6-v2` (default) - Fast and efficient, good for most use cases
- `all-mpnet-base-v2` - Better quality, slightly slower
- `multi-qa-mpnet-base-dot-v1` - Optimized for question-answering
- `paraphrase-multilingual-MiniLM-L12-v2` - Supports multiple languages
- `sentence-transformers/all-distilroberta-v1` - Good balance of speed and quality

## Model Performance Considerations

- **Speed**: Smaller models like MiniLM are faster but may have lower quality
- **Quality**: Larger models like MPNet provide better embeddings but are slower
- **Memory**: Consider your system's RAM when choosing model size
- **Language**: Use multilingual models if you work with non-English documents

## Adding Your Own Models

1. Download or train your embedding model
2. Place it in this `embeddings/` folder
3. Use the relative path when running DocuChat
4. Ensure the model is compatible with sentence-transformers library

## Troubleshooting

- If a model fails to load, check that all required files are present
- Verify the model is compatible with your sentence-transformers version
- Check available disk space and memory
- Use `--verbose` flag for detailed error messages