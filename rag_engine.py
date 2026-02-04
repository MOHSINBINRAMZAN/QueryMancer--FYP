"""
RAG Engine with FAISS Vector Database for QueryMancer

This module provides Retrieval-Augmented Generation (RAG) capabilities using FAISS 
vector database to enhance the Mistral LLM's performance by providing contextually
relevant schema information and query examples.

Compatible with existing QueryMancer architecture - integrates seamlessly with:
- agent.py (SchemaManager, LocalSQLTranslator)
- models.py (ModelManager)
- tools.py (LocalSchemaManager)
"""

import os
import json
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not installed. Install with: pip install faiss-cpu")

# LangChain imports for embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Ollama embeddings (using the same local Mistral setup)
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OLLAMA_EMBEDDINGS_AVAILABLE = False
    logging.warning("OllamaEmbeddings not available. Install: pip install langchain-ollama")

# Import configuration
try:
    from config import app_config, logger
except (ImportError, ValueError) as e:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from config import app_config, logger
    except (ImportError, ValueError):
        # Fallback if config has issues (e.g., missing DB credentials)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        app_config = None

# Constants
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), "schema.json")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db")
QUERY_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "query_examples.json")


@dataclass
class RAGConfig:
    """Configuration for RAG Engine"""
    # Embedding settings
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'mistral'))
    embedding_dim: int = 4096  # Mistral embedding dimension
    ollama_base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    
    # FAISS settings
    index_type: str = "IVFFlat"  # Options: "Flat", "IVFFlat", "HNSW"
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    
    # Retrieval settings
    top_k_tables: int = 5  # Top K tables to retrieve
    top_k_columns: int = 10  # Top K columns to retrieve
    top_k_examples: int = 3  # Top K query examples to retrieve
    similarity_threshold: float = 0.5  # Minimum similarity score
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    
    # Performance settings
    batch_size: int = 32
    use_gpu: bool = False


class LocalEmbeddingProvider:
    """
    Local embedding provider using Ollama + Mistral
    Falls back to simple TF-IDF based embeddings if Ollama is not available
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if OLLAMA_EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings = OllamaEmbeddings(
                    model=self.config.embedding_model,
                    base_url=self.config.ollama_base_url,
                )
                # Test embedding
                test_embedding = self.embeddings.embed_query("test")
                self.config.embedding_dim = len(test_embedding)
                logger.info(f"Ollama embeddings initialized. Dimension: {self.config.embedding_dim}")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama embeddings: {e}. Using fallback.")
                self.embeddings = None
        
        if self.embeddings is None:
            logger.info("Using TF-IDF fallback embeddings")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Initialize TF-IDF based fallback embeddings"""
        self.fallback_vocab = {}
        self.fallback_idf = {}
        self.config.embedding_dim = 768  # Fixed dimension for fallback
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embeddings:
            try:
                embedding = self.embeddings.embed_query(text)
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding error: {e}. Using fallback.")
        
        return self._fallback_embed(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.embeddings:
            try:
                embeddings = self.embeddings.embed_documents(texts)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Batch embedding error: {e}. Using fallback.")
        
        return np.array([self._fallback_embed(t) for t in texts], dtype=np.float32)
    
    def _fallback_embed(self, text: str) -> np.ndarray:
        """Simple fallback embedding using character n-grams"""
        # Create a simple hash-based embedding
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        # Normalize text
        text_lower = text.lower()
        words = text_lower.split()
        
        # Character n-gram hashing
        for word in words:
            for n in range(1, min(4, len(word) + 1)):
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                    idx = hash_val % self.config.embedding_dim
                    embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class FAISSVectorStore:
    """
    FAISS-based vector store for schema and query example storage
    """
    
    def __init__(self, config: RAGConfig, embedding_provider: LocalEmbeddingProvider):
        self.config = config
        self.embedding_provider = embedding_provider
        self.dimension = config.embedding_dim
        
        # Initialize indices
        self.table_index = None
        self.column_index = None
        self.example_index = None
        
        # Metadata storage
        self.table_metadata: List[Dict] = []
        self.column_metadata: List[Dict] = []
        self.example_metadata: List[Dict] = []
        
        # Paths
        self.vector_db_path = Path(VECTOR_DB_PATH)
        self.vector_db_path.mkdir(exist_ok=True)
        
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Vector store will not function.")
            return
        
        # Try to load existing indices
        if self._load_indices():
            logger.info("Loaded existing FAISS indices")
            return
        
        # Create new indices
        self._create_indices()
        logger.info("Created new FAISS indices")
    
    def _create_indices(self):
        """Create new FAISS indices - using Flat index for simplicity and reliability"""
        if not FAISS_AVAILABLE:
            return
        
        # Use simple Flat index which doesn't require training
        # This works well for datasets up to ~100k vectors and provides exact search
        self.table_index = faiss.IndexFlatIP(self.dimension)
        self.column_index = faiss.IndexFlatIP(self.dimension)
        self.example_index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Created FAISS Flat indices with dimension {self.dimension}")
    
    def _load_indices(self) -> bool:
        """Load existing indices from disk"""
        if not FAISS_AVAILABLE:
            return False
        
        try:
            table_index_path = self.vector_db_path / "table_index.faiss"
            column_index_path = self.vector_db_path / "column_index.faiss"
            example_index_path = self.vector_db_path / "example_index.faiss"
            metadata_path = self.vector_db_path / "metadata.pkl"
            
            if not all(p.exists() for p in [table_index_path, column_index_path, metadata_path]):
                return False
            
            self.table_index = faiss.read_index(str(table_index_path))
            self.column_index = faiss.read_index(str(column_index_path))
            
            if example_index_path.exists():
                self.example_index = faiss.read_index(str(example_index_path))
            else:
                self._create_example_index()
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.table_metadata = metadata.get('tables', [])
                self.column_metadata = metadata.get('columns', [])
                self.example_metadata = metadata.get('examples', [])
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load indices: {e}")
            return False
    
    def _create_example_index(self):
        """Create example index separately"""
        if self.config.index_type == "Flat":
            self.example_index = faiss.IndexFlatIP(self.dimension)
        else:
            self.example_index = faiss.IndexFlatIP(self.dimension)
    
    def save_indices(self):
        """Save indices to disk"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            faiss.write_index(self.table_index, str(self.vector_db_path / "table_index.faiss"))
            faiss.write_index(self.column_index, str(self.vector_db_path / "column_index.faiss"))
            
            if self.example_index and self.example_index.ntotal > 0:
                faiss.write_index(self.example_index, str(self.vector_db_path / "example_index.faiss"))
            
            with open(self.vector_db_path / "metadata.pkl", 'wb') as f:
                pickle.dump({
                    'tables': self.table_metadata,
                    'columns': self.column_metadata,
                    'examples': self.example_metadata
                }, f)
            
            logger.info("FAISS indices saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
    
    def add_table(self, table_name: str, table_info: Dict, embedding: np.ndarray):
        """Add a table to the vector store"""
        if not FAISS_AVAILABLE or self.table_index is None:
            return
        
        # Normalize embedding
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        self.table_index.add(embedding)
        self.table_metadata.append({
            'table_name': table_name,
            'info': table_info,
            'index': len(self.table_metadata)
        })
    
    def add_column(self, table_name: str, column_name: str, column_info: Dict, embedding: np.ndarray):
        """Add a column to the vector store"""
        if not FAISS_AVAILABLE or self.column_index is None:
            return
        
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        self.column_index.add(embedding)
        self.column_metadata.append({
            'table_name': table_name,
            'column_name': column_name,
            'info': column_info,
            'index': len(self.column_metadata)
        })
    
    def add_example(self, question: str, sql: str, description: str, embedding: np.ndarray):
        """Add a query example to the vector store"""
        if not FAISS_AVAILABLE or self.example_index is None:
            return
        
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        self.example_index.add(embedding)
        self.example_metadata.append({
            'question': question,
            'sql': sql,
            'description': description,
            'index': len(self.example_metadata)
        })
    
    def search_tables(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for relevant tables"""
        if not FAISS_AVAILABLE or self.table_index is None or self.table_index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Set nprobe for IVF index
        if hasattr(self.table_index, 'nprobe'):
            self.table_index.nprobe = self.config.nprobe
        
        k = min(top_k, self.table_index.ntotal)
        scores, indices = self.table_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.table_metadata):
                results.append((self.table_metadata[idx], float(score)))
        
        return results
    
    def search_columns(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for relevant columns"""
        if not FAISS_AVAILABLE or self.column_index is None or self.column_index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        if hasattr(self.column_index, 'nprobe'):
            self.column_index.nprobe = self.config.nprobe
        
        k = min(top_k, self.column_index.ntotal)
        scores, indices = self.column_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.column_metadata):
                results.append((self.column_metadata[idx], float(score)))
        
        return results
    
    def search_examples(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for relevant query examples"""
        if not FAISS_AVAILABLE or self.example_index is None or self.example_index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        if hasattr(self.example_index, 'nprobe'):
            self.example_index.nprobe = self.config.nprobe
        
        k = min(top_k, self.example_index.ntotal)
        scores, indices = self.example_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.example_metadata):
                results.append((self.example_metadata[idx], float(score)))
        
        return results


class RAGSchemaIndexer:
    """
    Indexes database schema into FAISS vector store for semantic retrieval
    """
    
    def __init__(self, config: RAGConfig, embedding_provider: LocalEmbeddingProvider, 
                 vector_store: FAISSVectorStore):
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.schema_data = None
        self.indexed = False
    
    def load_and_index_schema(self, schema_path: str = SCHEMA_FILE_PATH, force_reindex: bool = False):
        """Load schema from JSON and index into FAISS"""
        try:
            # Load schema
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema_data = json.load(f)
            
            logger.info(f"Loaded schema with {len(self.schema_data)} tables")
            
            # Check if already indexed
            if not force_reindex and self.vector_store.table_index.ntotal > 0:
                logger.info("Schema already indexed. Use force_reindex=True to re-index.")
                self.indexed = True
                return
            
            # Index tables and columns
            self._index_tables()
            self._index_columns()
            
            # Load and index query examples if available
            if os.path.exists(QUERY_EXAMPLES_PATH):
                self._index_examples()
            
            # Save indices
            self.vector_store.save_indices()
            self.indexed = True
            
            logger.info("Schema indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error indexing schema: {e}")
            raise
    
    def _index_tables(self):
        """Index all tables"""
        logger.info("Indexing tables...")
        
        for table_name, table_info in self.schema_data.items():
            # Skip metadata keys
            if table_name.startswith('_') or table_name in ['database_name', 'version']:
                continue
            
            # Create rich text representation for embedding
            text_repr = self._create_table_text_representation(table_name, table_info)
            
            # Generate embedding
            embedding = self.embedding_provider.embed_text(text_repr)
            
            # Add to vector store
            self.vector_store.add_table(table_name, table_info, embedding)
        
        logger.info(f"Indexed {self.vector_store.table_index.ntotal} tables")
    
    def _index_columns(self):
        """Index all columns"""
        logger.info("Indexing columns...")
        
        for table_name, table_info in self.schema_data.items():
            if table_name.startswith('_') or table_name in ['database_name', 'version']:
                continue
            
            columns = table_info.get('columns', [])
            
            # Handle both list and dict column formats
            if isinstance(columns, list):
                for col_name in columns:
                    text_repr = self._create_column_text_representation(table_name, col_name, {})
                    embedding = self.embedding_provider.embed_text(text_repr)
                    self.vector_store.add_column(table_name, col_name, {}, embedding)
            
            elif isinstance(columns, dict):
                for col_name, col_info in columns.items():
                    text_repr = self._create_column_text_representation(table_name, col_name, col_info)
                    embedding = self.embedding_provider.embed_text(text_repr)
                    self.vector_store.add_column(table_name, col_name, col_info, embedding)
        
        logger.info(f"Indexed {self.vector_store.column_index.ntotal} columns")
    
    def _index_examples(self):
        """Index query examples"""
        try:
            with open(QUERY_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            logger.info(f"Indexing {len(examples)} query examples...")
            
            for example in examples:
                question = example.get('question', '')
                sql = example.get('sql', '')
                description = example.get('description', '')
                
                # Combine for embedding
                text_repr = f"Question: {question}\nDescription: {description}"
                embedding = self.embedding_provider.embed_text(text_repr)
                
                self.vector_store.add_example(question, sql, description, embedding)
            
            logger.info(f"Indexed {self.vector_store.example_index.ntotal} examples")
            
        except Exception as e:
            logger.warning(f"Could not index examples: {e}")
    
    def _create_table_text_representation(self, table_name: str, table_info: Dict) -> str:
        """Create rich text representation of a table for embedding"""
        parts = [f"Table: {table_name}"]
        
        # Add variations (natural language names)
        variations = table_info.get('variations', [])
        if variations:
            parts.append(f"Also known as: {', '.join(variations)}")
        
        # Add schema
        schema = table_info.get('schema', 'dbo')
        parts.append(f"Schema: {schema}")
        
        # Add primary keys
        primary_keys = table_info.get('primary_keys', [])
        if primary_keys:
            parts.append(f"Primary keys: {', '.join(primary_keys)}")
        
        # Add foreign keys
        foreign_keys = table_info.get('foreign_keys', {})
        if foreign_keys:
            fk_strs = [f"{k} -> {v}" for k, v in foreign_keys.items()]
            parts.append(f"Foreign keys: {'; '.join(fk_strs)}")
        
        # Add column summary
        columns = table_info.get('columns', [])
        if columns:
            if isinstance(columns, list):
                col_sample = columns[:10]  # First 10 columns
            else:
                col_sample = list(columns.keys())[:10]
            parts.append(f"Key columns: {', '.join(col_sample)}")
        
        return '\n'.join(parts)
    
    def _create_column_text_representation(self, table_name: str, column_name: str, 
                                           column_info: Dict) -> str:
        """Create rich text representation of a column for embedding"""
        parts = [f"Column: {column_name}", f"Table: {table_name}"]
        
        # Add column type if available
        if isinstance(column_info, dict):
            col_type = column_info.get('type', column_info.get('data_type', ''))
            if col_type:
                parts.append(f"Type: {col_type}")
            
            description = column_info.get('description', '')
            if description:
                parts.append(f"Description: {description}")
        
        # Add semantic hints based on column name patterns
        name_lower = column_name.lower()
        
        if 'name' in name_lower:
            parts.append("Semantic: This column likely contains names or identifiers")
        elif 'date' in name_lower or 'time' in name_lower:
            parts.append("Semantic: This column likely contains date/time values")
        elif 'email' in name_lower:
            parts.append("Semantic: This column likely contains email addresses")
        elif 'phone' in name_lower or 'tel' in name_lower:
            parts.append("Semantic: This column likely contains phone numbers")
        elif 'id' in name_lower or 'record_id' in name_lower:
            parts.append("Semantic: This column likely contains identifiers or keys")
        elif 'amount' in name_lower or 'price' in name_lower or 'cost' in name_lower:
            parts.append("Semantic: This column likely contains monetary values")
        elif 'status' in name_lower:
            parts.append("Semantic: This column likely contains status values")
        
        return '\n'.join(parts)


class RAGQueryEnhancer:
    """
    Enhances user queries with retrieved context from FAISS vector store
    to improve LLM SQL generation accuracy
    """
    
    def __init__(self, config: RAGConfig, embedding_provider: LocalEmbeddingProvider,
                 vector_store: FAISSVectorStore, schema_data: Dict):
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.schema_data = schema_data
        
        # Query cache
        self._cache: Dict[str, Tuple[str, float]] = {}
    
    def enhance_query(self, user_question: str) -> Dict[str, Any]:
        """
        Enhance user query with relevant context from vector store
        
        Returns:
            Dict containing:
            - enhanced_context: Schema context enhanced with RAG retrieval
            - relevant_tables: List of relevant table names
            - relevant_columns: List of relevant columns with table info
            - similar_examples: List of similar query examples
            - confidence_scores: Retrieval confidence scores
        """
        # Check cache
        cache_key = hashlib.md5(user_question.encode()).hexdigest()
        if self.config.cache_enabled and cache_key in self._cache:
            cached_result, cache_time = self._cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl:
                return cached_result
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(user_question)
        
        # Retrieve relevant tables
        table_results = self.vector_store.search_tables(
            query_embedding, 
            top_k=self.config.top_k_tables
        )
        
        # Retrieve relevant columns
        column_results = self.vector_store.search_columns(
            query_embedding,
            top_k=self.config.top_k_columns
        )
        
        # Retrieve similar examples
        example_results = self.vector_store.search_examples(
            query_embedding,
            top_k=self.config.top_k_examples
        )
        
        # Build enhanced context
        enhanced_context = self._build_enhanced_context(
            user_question,
            table_results,
            column_results,
            example_results
        )
        
        # Extract relevant table names
        relevant_tables = [r[0]['table_name'] for r in table_results 
                         if r[1] >= self.config.similarity_threshold]
        
        # Extract relevant columns
        relevant_columns = [
            {
                'table': r[0]['table_name'],
                'column': r[0]['column_name'],
                'score': r[1]
            }
            for r in column_results
            if r[1] >= self.config.similarity_threshold
        ]
        
        # Extract similar examples
        similar_examples = [
            {
                'question': r[0]['question'],
                'sql': r[0]['sql'],
                'score': r[1]
            }
            for r in example_results
            if r[1] >= self.config.similarity_threshold
        ]
        
        result = {
            'enhanced_context': enhanced_context,
            'relevant_tables': relevant_tables,
            'relevant_columns': relevant_columns,
            'similar_examples': similar_examples,
            'confidence_scores': {
                'table_confidence': np.mean([r[1] for r in table_results]) if table_results else 0.0,
                'column_confidence': np.mean([r[1] for r in column_results]) if column_results else 0.0,
                'example_confidence': np.mean([r[1] for r in example_results]) if example_results else 0.0
            }
        }
        
        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = (result, time.time())
        
        return result
    
    def _build_enhanced_context(self, question: str, table_results: List, 
                                column_results: List, example_results: List) -> str:
        """Build enhanced schema context with RAG retrieval results"""
        context_parts = []
        
        # Header
        context_parts.append("=== RAG-ENHANCED DATABASE CONTEXT ===")
        context_parts.append(f"Query: {question}")
        context_parts.append("")
        
        # Most relevant tables section
        if table_results:
            context_parts.append("=== MOST RELEVANT TABLES (by semantic similarity) ===")
            
            for table_meta, score in table_results:
                table_name = table_meta['table_name']
                table_info = table_meta.get('info', {})
                
                context_parts.append(f"\n[TABLE: {table_name}] (relevance: {score:.3f})")
                
                # Schema
                schema = table_info.get('schema', 'dbo')
                context_parts.append(f"  Schema: {schema}")
                
                # Primary keys
                primary_keys = table_info.get('primary_keys', [])
                if primary_keys:
                    context_parts.append(f"  Primary Keys: {', '.join(primary_keys)}")
                
                # Foreign keys (relationships)
                foreign_keys = table_info.get('foreign_keys', {})
                if foreign_keys:
                    context_parts.append("  Foreign Keys (for JOINs):")
                    for fk_col, fk_target in foreign_keys.items():
                        context_parts.append(f"    - {fk_col} -> {fk_target}")
                
                # Columns
                columns = table_info.get('columns', [])
                if columns:
                    if isinstance(columns, list):
                        context_parts.append(f"  Columns: {', '.join(columns[:15])}")
                        if len(columns) > 15:
                            context_parts.append(f"    ... and {len(columns) - 15} more")
                    else:
                        col_names = list(columns.keys())[:15]
                        context_parts.append(f"  Columns: {', '.join(col_names)}")
                
                # Variations (natural language names)
                variations = table_info.get('variations', [])
                if variations:
                    context_parts.append(f"  Natural names: {', '.join(variations[:5])}")
        
        # Relevant columns section
        if column_results:
            context_parts.append("\n=== RELEVANT COLUMNS (semantic match) ===")
            
            # Group by table
            columns_by_table = {}
            for col_meta, score in column_results:
                table = col_meta['table_name']
                if table not in columns_by_table:
                    columns_by_table[table] = []
                columns_by_table[table].append((col_meta['column_name'], score))
            
            for table, cols in columns_by_table.items():
                col_strs = [f"{c[0]} ({c[1]:.3f})" for c in cols]
                context_parts.append(f"  {table}: {', '.join(col_strs)}")
        
        # Similar query examples section
        if example_results:
            context_parts.append("\n=== SIMILAR QUERY EXAMPLES (for reference) ===")
            
            for example_meta, score in example_results:
                context_parts.append(f"\n  Example (similarity: {score:.3f}):")
                context_parts.append(f"    Question: {example_meta['question']}")
                context_parts.append(f"    SQL: {example_meta['sql']}")
        
        # Important notes for LLM
        context_parts.append("\n=== IMPORTANT NOTES FOR SQL GENERATION ===")
        context_parts.append("1. Use ONLY the tables and columns listed above")
        context_parts.append("2. Use foreign key relationships for JOINs")
        context_parts.append("3. Use TOP instead of LIMIT (SQL Server syntax)")
        context_parts.append("4. Wrap all identifiers in square brackets: [TableName].[ColumnName]")
        context_parts.append("5. Reference similar examples when structuring your query")
        
        return '\n'.join(context_parts)
    
    def get_table_context(self, table_names: List[str]) -> str:
        """Get detailed context for specific tables (for fallback)"""
        if not self.schema_data:
            return ""
        
        context_parts = []
        
        for table_name in table_names:
            if table_name in self.schema_data:
                table_info = self.schema_data[table_name]
                context_parts.append(f"\n=== TABLE: {table_name} ===")
                
                # All columns
                columns = table_info.get('columns', [])
                if columns:
                    context_parts.append("Columns:")
                    if isinstance(columns, list):
                        for col in columns:
                            context_parts.append(f"  - {col}")
                    else:
                        for col_name, col_info in columns.items():
                            context_parts.append(f"  - {col_name}")
                
                # Foreign keys
                foreign_keys = table_info.get('foreign_keys', {})
                if foreign_keys:
                    context_parts.append("Relationships:")
                    for fk_col, fk_target in foreign_keys.items():
                        context_parts.append(f"  - {fk_col} -> {fk_target}")
        
        return '\n'.join(context_parts)


class QueryMancerRAG:
    """
    Main RAG integration class for QueryMancer
    
    This class provides the primary interface for RAG functionality,
    designed to integrate seamlessly with existing QueryMancer components.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for RAG engine"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: RAGConfig = None, schema_path: str = SCHEMA_FILE_PATH):
        """Initialize RAG engine"""
        if self._initialized:
            return
        
        self.config = config or RAGConfig()
        self.schema_path = schema_path
        
        # Initialize components
        logger.info("Initializing QueryMancer RAG Engine...")
        
        self.embedding_provider = LocalEmbeddingProvider(self.config)
        self.vector_store = FAISSVectorStore(self.config, self.embedding_provider)
        self.indexer = RAGSchemaIndexer(self.config, self.embedding_provider, self.vector_store)
        
        # Load and index schema
        self.indexer.load_and_index_schema(schema_path)
        
        # Initialize query enhancer
        self.query_enhancer = RAGQueryEnhancer(
            self.config,
            self.embedding_provider,
            self.vector_store,
            self.indexer.schema_data
        )
        
        self._initialized = True
        logger.info("QueryMancer RAG Engine initialized successfully")
    
    def enhance_query(self, user_question: str) -> Dict[str, Any]:
        """
        Main method to enhance a user query with RAG context
        
        Args:
            user_question: Natural language question from user
            
        Returns:
            Dict with enhanced context and metadata
        """
        return self.query_enhancer.enhance_query(user_question)
    
    def get_relevant_tables(self, user_question: str, top_k: int = 5) -> List[str]:
        """Get list of relevant table names for a query"""
        result = self.enhance_query(user_question)
        return result.get('relevant_tables', [])[:top_k]
    
    def get_enhanced_schema_context(self, user_question: str) -> str:
        """Get enhanced schema context string for LLM prompt"""
        result = self.enhance_query(user_question)
        return result.get('enhanced_context', '')
    
    def get_similar_examples(self, user_question: str, top_k: int = 3) -> List[Dict]:
        """Get similar query examples"""
        result = self.enhance_query(user_question)
        return result.get('similar_examples', [])[:top_k]
    
    def add_query_example(self, question: str, sql: str, description: str = ""):
        """Add a new query example to the vector store"""
        text_repr = f"Question: {question}\nDescription: {description}"
        embedding = self.embedding_provider.embed_text(text_repr)
        self.vector_store.add_example(question, sql, description, embedding)
        self.vector_store.save_indices()
        logger.info(f"Added new query example: {question[:50]}...")
    
    def reindex_schema(self, schema_path: str = None):
        """Force re-index of schema"""
        path = schema_path or self.schema_path
        self.indexer.load_and_index_schema(path, force_reindex=True)
        logger.info("Schema re-indexed successfully")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            'tables_indexed': self.vector_store.table_index.ntotal if self.vector_store.table_index else 0,
            'columns_indexed': self.vector_store.column_index.ntotal if self.vector_store.column_index else 0,
            'examples_indexed': self.vector_store.example_index.ntotal if self.vector_store.example_index else 0,
            'embedding_dimension': self.config.embedding_dim,
            'index_type': self.config.index_type,
            'cache_enabled': self.config.cache_enabled
        }


# Global RAG instance getter
_rag_instance: Optional[QueryMancerRAG] = None


def get_rag_engine(config: RAGConfig = None) -> QueryMancerRAG:
    """Get or create the global RAG engine instance"""
    global _rag_instance
    
    if _rag_instance is None:
        _rag_instance = QueryMancerRAG(config)
    
    return _rag_instance


def enhance_with_rag(user_question: str) -> str:
    """
    Convenience function to enhance a query with RAG context
    
    This function is designed to be called from existing code with minimal changes.
    
    Args:
        user_question: The user's natural language question
        
    Returns:
        Enhanced schema context string for LLM prompt
    """
    rag = get_rag_engine()
    return rag.get_enhanced_schema_context(user_question)


def get_rag_relevant_tables(user_question: str, top_k: int = 5) -> List[str]:
    """
    Convenience function to get relevant tables using RAG
    
    Args:
        user_question: The user's natural language question
        top_k: Maximum number of tables to return
        
    Returns:
        List of relevant table names
    """
    rag = get_rag_engine()
    return rag.get_relevant_tables(user_question, top_k)


# Module initialization check
if __name__ == "__main__":
    # Test RAG engine
    print("Testing QueryMancer RAG Engine...")
    
    rag = get_rag_engine()
    stats = rag.get_statistics()
    print(f"RAG Statistics: {json.dumps(stats, indent=2)}")
    
    # Test query enhancement
    test_question = "Show me all contacts and their accounts"
    result = rag.enhance_query(test_question)
    
    print(f"\nTest Question: {test_question}")
    print(f"Relevant Tables: {result['relevant_tables']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print("\nEnhanced Context Preview:")
    print(result['enhanced_context'][:500] + "...")
