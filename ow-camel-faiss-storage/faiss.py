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

logger = logging.getLogger(__name__)


@dependencies_required(["faiss-cpu", "langchain_community"])
class FaissVectorStorage(BaseVectorStorage):
    """
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
        FAISS_INDEX_NAME = os.path.join("ow-o1-vecs", f"faiss_index_{self.collection_name}")

        # Check if the index already exists
        if os.path.exists(FAISS_INDEX_NAME):
            logger.info(f"Loading existing FAISS index from {FAISS_INDEX_NAME}.")
            # Load the existing FAISS index
            self.faiss_index = FAISS.load_local(FAISS_INDEX_NAME, embedding, allow_dangerous_deserialization=True)
        else:
            logger.info(f"Creating new FAISS index in {FAISS_INDEX_NAME}.")
            # Create a new FAISS index
            self.faiss_index = FAISS.from_texts([], embedding=embedding, metadatas=self.metadatas)

    def add(self, records: List[VectorRecord]) -> None:
        """
        Add a list of vector records to the FAISS index.
        """
        self.faiss_index.add_texts([record.text for record in records])

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