# QueryMancer - Complete Project Explanation

## 📖 Overview

**QueryMancer** is an AI-powered Natural Language to SQL translation system designed as a Final Year Project (FYP). It converts plain English questions into executable SQL queries using a local Large Language Model (Mistral) combined with RAG (Retrieval-Augmented Generation) technology via FAISS vector database.

---

## 🏛️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          QueryMancer System Architecture                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐                                                                │
│  │   User       │  Natural Language Query: "Show all contacts from Lahore"      │
│  │   Input      │─────────────────────────────────────────┐                     │
│  └──────────────┘                                         │                     │
│                                                           ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        STREAMLIT WEB UI (ui.py, app.py)                  │   │
│  │                    - Dark-themed responsive interface                     │   │
│  │                    - Query input & result display                         │   │
│  │                    - Query history tracking                               │   │
│  └──────────────────────────────────────────┬──────────────────────────────┘   │
│                                              │                                   │
│                                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT LAYER (agent.py)                           │   │
│  │            ┌─────────────────────────────────────────┐                   │   │
│  │            │        LocalSQLTranslator               │                   │   │
│  │            │   - Manages LLM interactions            │                   │   │
│  │            │   - Coordinates RAG retrieval           │                   │   │
│  │            │   - Handles query generation            │                   │   │
│  │            └─────────────────────────────────────────┘                   │   │
│  │            ┌─────────────────────────────────────────┐                   │   │
│  │            │        SchemaManager                    │                   │   │
│  │            │   - Loads schema.json                   │                   │   │
│  │            │   - Provides context to LLM             │                   │   │
│  │            │   - Manages table/column mappings       │                   │   │
│  │            └─────────────────────────────────────────┘                   │   │
│  └──────────────────────────────────────────┬──────────────────────────────┘   │
│                                              │                                   │
│                 ┌────────────────────────────┼────────────────────────────┐     │
│                 │                            │                            │     │
│                 ▼                            ▼                            ▼     │
│  ┌────────────────────┐   ┌────────────────────────────┐   ┌────────────────┐  │
│  │   RAG ENGINE       │   │    OLLAMA + MISTRAL LLM    │   │    TOOLS       │  │
│  │  (rag_engine.py)   │   │                            │   │  (tools.py)    │  │
│  │                    │   │  - Local inference         │   │                │  │
│  │  ┌──────────┐      │   │  - Low temperature (0.05)  │   │ - SQL Execution│  │
│  │  │ Table    │      │   │  - Context window: 8192    │   │ - Validation   │  │
│  │  │ Index    │      │   │                            │   │ - Connection   │  │
│  │  │ (FAISS)  │      │   └────────────────────────────┘   │   Pool Mgmt    │  │
│  │  └──────────┘      │                                    └────────────────┘  │
│  │  ┌──────────┐      │                                                        │
│  │  │ Column   │      │                                                        │
│  │  │ Index    │      │                                                        │
│  │  │ (FAISS)  │      │                                                        │
│  │  └──────────┘      │                                                        │
│  │  ┌──────────┐      │                                                        │
│  │  │ Example  │      │                                                        │
│  │  │ Index    │      │                                                        │
│  │  │ (FAISS)  │      │                                                        │
│  │  └──────────┘      │                                                        │
│  └────────────────────┘                                                        │
│                                              │                                   │
│                                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       SQL SERVER DATABASE                                │   │
│  │                 (ODBC Driver 17 + Windows Authentication)                │   │
│  │                                                                          │   │
│  │   Tables: CONTACT, ACCOUNT, EVENT, EVENT_DELEGATE, DONATION, etc.       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure Explanation

### 🔵 Core Application Files

| File            | Purpose                        | Key Components                                                                              |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| **`app.py`**    | Main application entry point   | Streamlit app initialization, session management, route handling, logging configuration     |
| **`ui.py`**     | Enhanced user interface module | `EnhancedLocalUI` class - renders all UI components, chat interface, styling                |
| **`agent.py`**  | Core AI agent logic            | `LocalSQLTranslator`, `SchemaManager`, `ConversationMemory` - handles NL to SQL translation |
| **`tools.py`**  | Database tools & utilities     | `LocalSchemaManager`, SQL execution, query validation, connection pooling                   |
| **`models.py`** | Model management               | Model initialization, Ollama integration, response processing                               |
| **`config.py`** | Configuration management       | Environment variables, database config, model settings                                      |

### 🟢 RAG (Retrieval-Augmented Generation) Components

| File                      | Purpose                          | Key Components                                                                 |
| ------------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| **`rag_engine.py`**       | FAISS vector database management | `QueryMancerRAG`, `LocalEmbeddingProvider`, vector indexing, similarity search |
| **`query_examples.json`** | Example queries for RAG          | Pre-defined NL-SQL pairs for context retrieval                                 |
| **`schema.json`**         | Database schema definition       | Tables, columns, relationships, data types                                     |
| **`vector_db/`**          | FAISS index storage directory    | `table_index.faiss`, `column_index.faiss`, `example_index.faiss`               |

### 🟡 Database & Data Management

| File                               | Purpose                | Description                                            |
| ---------------------------------- | ---------------------- | ------------------------------------------------------ |
| **`create_tables_from_schema.py`** | Schema generation      | Generates schema.json from live database introspection |
| **`populate_database.py`**         | Data population        | Seeds database with sample data                        |
| **`generate_sample_data.py`**      | Sample data generation | Creates realistic test data                            |
| **`insert_sample_data.sql`**       | SQL insert statements  | Raw SQL for data insertion                             |
| **`execute_inserts.py`**           | Insert execution       | Runs insert statements against database                |
| **`insert_data_fixed.py`**         | Fixed data insertion   | Handles specific data insertion scenarios              |

### 🟠 Styling & UI Assets

| File                     | Purpose                   |
| ------------------------ | ------------------------- |
| **`style.css`**          | Main application styles   |
| **`progress-bar.css`**   | Progress indicator styles |
| **`welcome-styles.css`** | Welcome page styles       |
| **`load_css.py`**        | CSS loading utility       |

### 🔴 Testing & Validation

| File                            | Purpose                            |
| ------------------------------- | ---------------------------------- |
| **`test_rag_integration.py`**   | RAG system integration tests       |
| **`test_universal_support.py`** | Cross-platform compatibility tests |
| **`test_user_activity.py`**     | User activity logging tests        |
| **`test_fix.py`**               | Bug fix validation tests           |
| **`universal_test.py`**         | Comprehensive test suite           |

### 📊 Documentation & Reports

| File                          | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| **`README.md`**               | Project documentation & setup guide      |
| **`COMPARATIVE_ANALYSIS.md`** | Performance comparison: RAG vs No-RAG    |
| **`EXPLANATION.md`**          | This file - detailed project explanation |
| **`requirements.txt`**        | Python dependencies                      |

### 📁 Directories

| Directory          | Contents                                 |
| ------------------ | ---------------------------------------- |
| **`logs/`**        | Application logs with timestamps         |
| **`cache/`**       | Query cache for performance optimization |
| **`schemas/`**     | Additional schema files                  |
| **`vector_db/`**   | FAISS vector database indices            |
| **`__pycache__/`** | Python compiled bytecode                 |
| **`venv/`**        | Python virtual environment               |
| **`.streamlit/`**  | Streamlit configuration                  |

---

## 🔧 How It Works

### Step 1: User Input

User enters a natural language query like: _"Show me all contacts who registered for events this month"_

### Step 2: RAG Context Retrieval

```python
# rag_engine.py performs semantic search
relevant_context = rag_engine.search(
    query="contacts who registered for events",
    top_k_tables=5,
    top_k_columns=10,
    top_k_examples=3
)
```

The RAG engine searches FAISS indices to find:

- **Relevant tables**: CONTACT, EVENT_DELEGATE, EVENT
- **Relevant columns**: FIRST_NAME, LAST_NAME, REGISTRATION_DATE
- **Similar examples**: Pre-existing NL-SQL pairs

### Step 3: LLM Query Generation

```python
# agent.py constructs the prompt
prompt = f"""
You are a SQL expert. Generate a SQL Server query for:
"{user_query}"

Database Schema Context:
{relevant_context}

Rules:
- Use proper JOINs
- Add WHERE DELETED = 0
- Limit to TOP 15 results
"""

# Mistral generates SQL via Ollama
sql_query = ollama_llm.generate(prompt)
```

### Step 4: Query Validation & Execution

```python
# tools.py validates and executes
validated_query = validate_sql(sql_query)
results = execute_query(validated_query)
```

### Step 5: Response Display

Results are formatted and displayed in the Streamlit UI with syntax highlighting.

---

## 🧠 RAG (Retrieval-Augmented Generation) Explained

### Why RAG?

| Issue Without RAG               | Solution With RAG                |
| ------------------------------- | -------------------------------- |
| LLM hallucinates table names    | FAISS retrieves actual schema    |
| Wrong column references         | Context includes real columns    |
| Inefficient full-schema prompts | Only relevant tables sent        |
| No query pattern learning       | Example queries guide generation |

### FAISS Vector Indices

1. **Table Index** (`table_index.faiss`)
   - Stores table names + descriptions as vectors
   - Used to find relevant tables for a query

2. **Column Index** (`column_index.faiss`)
   - Stores column names + data types as vectors
   - Helps identify correct column references

3. **Example Index** (`example_index.faiss`)
   - Stores NL-SQL pairs from `query_examples.json`
   - Provides similar query patterns to the LLM

---

## ⚡ Performance Improvements

Based on testing (see `COMPARATIVE_ANALYSIS.md`):

| Metric                | Without RAG | With RAG | Improvement |
| --------------------- | ----------- | -------- | ----------- |
| Exact-Match Accuracy  | 67%         | 89%      | **+22%**    |
| Execution Accuracy    | 72%         | 95%      | **+23%**    |
| Schema Compliance     | 78%         | 97%      | **+19%**    |
| Complex Query Success | 45%         | 82%      | **+37%**    |

---

## 🔐 Security Features

- **Windows Authentication**: Uses trusted connections (no password storage)
- **Local Processing**: All LLM inference runs locally via Ollama
- **No External APIs**: Data never leaves your machine
- **SQL Injection Prevention**: Query validation before execution
- **Soft Delete Support**: Respects DELETED flags in queries

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/MOHSINBINRAMZAN/QueryMancer--FYP.git
cd QueryMancer--FYP

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama + Mistral
# Download from https://ollama.ai/download
ollama pull mistral

# 5. Configure environment
# Copy .env.example to .env and set database credentials

# 6. Build FAISS indices
python rag_engine.py --build-index

# 7. Launch application
streamlit run app.py
```

---

## 📊 Technology Stack Summary

| Layer               | Technology            |
| ------------------- | --------------------- |
| **Frontend**        | Streamlit (Python)    |
| **Backend**         | Python 3.10+          |
| **LLM**             | Mistral 7B via Ollama |
| **Vector DB**       | FAISS (Facebook AI)   |
| **Embeddings**      | Ollama/MiniLM-L6      |
| **Database**        | Microsoft SQL Server  |
| **ORM**             | SQLAlchemy            |
| **Agent Framework** | LangChain             |

---

## 👨‍💻 Author

**Mohsin Bin Ramzan**  
Final Year Project - Natural Language to SQL Translation System

---

## 📄 License

MIT License - See LICENSE file for details.
