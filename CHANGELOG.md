## [2025-03-22] [PR#4](https://github.com/OpenWorkspace-o1/ow-camel-faiss-storage/pull/4)

### Added
- Added `_validate_and_convert_vectors` method to validate and convert `VectorRecord` instances in `FaissVectorStorage`.
- Added `ruff` as an optional development dependency in `pyproject.toml`.

### Changed
- Refactored index creation logic into a dedicated `_check_and_create_index` method in `FaissVectorStorage`.
- Updated `add` method to use `add_embeddings` for better vector management.

## [2025-03-22] [PR#2](https://github.com/OpenWorkspace-o1/ow-camel-faiss-storage/pull/2)

### Added
- Introduced `FaissVectorStorage` class for FAISS-based vector storage, supporting adding, querying, and deleting vector records.
- Added `pyproject.toml` for project configuration with dependencies like `faiss-cpu`, `langchain`, and `langchain-openai`.
- Initialized `__init__.py` to expose the `FaissVectorStorage` class.