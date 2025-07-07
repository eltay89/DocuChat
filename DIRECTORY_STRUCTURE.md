# DocuChat Directory Structure

This document explains the directory structure used by DocuChat for automatic file monitoring and vector database storage.

## Directory Layout

```
DocuChat/
├── documents/          # 📂 Document files (monitored automatically)
│   ├── *.pdf
│   ├── *.txt
│   ├── *.docx
│   └── *.md
├── vectordbs/          # 📂 Vector database and cache files
│   ├── chroma.sqlite3  # ChromaDB database
│   ├── file_metadata.pkl  # File monitoring cache
│   └── doc_metadata.pkl   # Document metadata
├── embeddings/         # 📂 Downloaded embedding models
├── models/             # 📂 LLM model files
└── docuchat.py         # 🐍 Main application
```

## Key Features

### 🔄 Automatic File Monitoring
- **Monitors**: `./documents/` folder for changes
- **Detects**: Added, modified, and removed files
- **Supports**: PDF, TXT, DOCX, DOC, MD files
- **Cache**: Stores file metadata in `./vectordbs/file_metadata.pkl`

### 🗄️ Persistent Vector Database
- **Location**: `./vectordbs/` folder
- **Database**: ChromaDB with persistent storage
- **Metadata**: Document chunks stored in `./vectordbs/doc_metadata.pkl`
- **Incremental**: Only processes changed files

## Usage Examples

### Basic Usage
```bash
# Start interactive chat (uses default ./documents folder)
python docuchat.py

# Specify custom document folder
python docuchat.py --folder_path ./my_documents
```

### Document Management
```bash
# Refresh documents (check for changes)
python docuchat.py --refresh

# Force complete re-processing of all documents
python docuchat.py --force-refresh

# Single query mode
python docuchat.py --query "What is the main topic?"
```

### Configuration Options
```bash
# Disable RAG (no document processing)
python docuchat.py --no-rag

# Custom chunk size for document processing
python docuchat.py --chunk_size 512

# Custom embedding model
python docuchat.py --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

## File Operations

### Adding Documents
1. Place files in `./documents/` folder
2. Run `python docuchat.py --refresh` or start the application
3. New files are automatically processed and added to the vector database

### Updating Documents
1. Modify files in `./documents/` folder
2. Run `python docuchat.py --refresh` or start the application
3. Modified files are automatically re-processed

### Removing Documents
1. Delete files from `./documents/` folder
2. Run `python docuchat.py --refresh` or start the application
3. Corresponding embeddings are automatically removed from the vector database

## Cache Management

### File Metadata Cache
- **File**: `./vectordbs/file_metadata.pkl`
- **Contains**: File hashes, modification times, processing timestamps
- **Purpose**: Detect file changes efficiently

### Document Metadata Cache
- **File**: `./vectordbs/doc_metadata.pkl`
- **Contains**: Document chunk information and file associations
- **Purpose**: Track which chunks belong to which files

### Clearing Cache
```bash
# Force complete refresh (clears all caches)
python docuchat.py --force-refresh
```

## Troubleshooting

### Common Issues

1. **Documents not found**
   - Ensure files are in `./documents/` folder
   - Check file extensions (PDF, TXT, DOCX, DOC, MD)
   - Run with `--refresh` flag

2. **Vector database issues**
   - Delete `./vectordbs/` folder and restart
   - Use `--force-refresh` to rebuild completely

3. **File monitoring not working**
   - Check file permissions
   - Ensure `./vectordbs/` folder is writable
   - Run test: `python test_directories.py`

### Testing
```bash
# Run directory structure test
python test_directories.py
```

## Technical Details

### File Monitoring Algorithm
1. Calculate SHA-256 hash of each file
2. Compare with cached metadata
3. Process only changed files
4. Update cache with new metadata

### Vector Database
- **Engine**: ChromaDB
- **Storage**: Persistent SQLite database
- **Embeddings**: Configurable sentence transformers
- **Chunking**: Configurable chunk size (default: 1000 characters)

### Performance
- **Incremental**: Only processes changed files
- **Efficient**: Hash-based change detection
- **Scalable**: Handles large document collections
- **Fast**: In-memory vector search with persistent storage