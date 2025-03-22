# ========= Copyright 2025 @ OpenWorkspace-o1. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2025 @ OpenWorkspace-o1. All Rights Reserved. =========

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from camel.storages.vectordb_storages import (
    BaseVectorStorage,
    VectorDBQuery,
    VectorDBQueryResult,
    VectorDBStatus,
    VectorRecord,
)
from camel.utils import dependencies_required
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


@dependencies_required(["faiss-cpu", "langchain_community"])
class FaissVectorStorage(BaseVectorStorage):
    """
    A vector storage implementation using FAISS.
    Args:
        collection_name (str): The name of the collection to store the vectors in.
        embedding_model (str): The name of the embedding model to use.
        embedding_dim (int): The dimension of the embedding vectors.
        metadatas (List[dict], optional): A list of dictionaries containing metadata for each vector.
    """
    def __init__(self,
                 collection_name: str,
                 embedding_model: str,
                 embedding_dim: int,
                 metadatas: List[dict] | None = None
            ):
        super().__init__(collection_name)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.metadatas = metadatas

        # Initialize vector store
        embedding = OpenAIEmbeddings(model=self.embedding_model, dimensions=self.embedding_dim)
        self.FAISS_INDEX_NAME = os.path.join("ow-o1-vecs", f"faiss_index_{self.collection_name}")

        self._check_and_create_index(embedding, self.FAISS_INDEX_NAME)

    def _check_and_create_index(self, embedding: Embeddings, index_name: str) -> None:
        if os.path.exists(index_name):
            logger.info(f"Loading existing FAISS index from {index_name}.")
            # Load the existing FAISS index
            self.faiss_index = FAISS.load_local(index_name, embedding, allow_dangerous_deserialization=True)
        else:
            logger.info(f"Creating new FAISS index in {index_name}.")
            # Create a new FAISS index
            self.faiss_index = FAISS.from_texts([], embedding=embedding, metadatas=self.metadatas)

    def _validate_and_convert_vectors(
        self, records: List[VectorRecord]
    ) -> Tuple[List[List[float]], List[str], List[dict]]:
        r"""Validates and converts VectorRecord instances to the format
        expected by Milvus.

        Args:
            records (List[VectorRecord]): List of vector records to validate
            and convert.

        Returns:
            Tuple[List[List[float]], List[str], List[dict]]: A tuple of validated vectors, ids, and metadatas.
        """
        validated_vectors = []
        validated_ids = []
        validated_metadatas = []

        for record in records:
            validated_vectors.append(record.vector)
            validated_ids.append(record.id)
            validated_metadatas.append(record.payload)

        return validated_vectors, validated_ids, validated_metadatas

    def add(self, records: List[VectorRecord], **kwargs: Any) -> None:
        r"""Adds a list of vectors to the specified collection.

        Args:
            records (List[VectorRecord]): List of vectors to be added.
            **kwargs (Any): Additional keyword arguments pass to insert.

        Raises:
            RuntimeError: If there was an error in the addition process.
        """
        validated_vectors, validated_ids, validated_metadatas = self._validate_and_convert_vectors(records)
        self.faiss_index.add_embeddings(
            text_embeddings=validated_vectors,
            ids=validated_ids,
            metadatas=validated_metadatas
        )
        self.faiss_index.save_local(self.FAISS_INDEX_NAME)

    def query(self, query: str, top_k: int = 10) -> List[VectorRecord]:
        """
        Query the FAISS index for the top k most similar vectors to the query.
        """
        return self.faiss_index.similarity_search_with_score(query, k=top_k)

    def delete(self, ids: List[str]) -> None:
        """
        Delete a list of vector records from the FAISS index.
        """
        self.faiss_index.delete(ids)

    def status(self) -> VectorDBStatus:
        """
        Get the status of the FAISS index.
        """
        return VectorDBStatus(
            vector_dim=self.embedding_dim,
            vector_count=len(self.faiss_index.get_collection_info()),
        )