# MiniMe Documentation

Welcome to the MiniMe documentation! This folder contains detailed explanations of the codebase, organized by topic.

## 📚 Documentation Index

### Core Concepts

1. **[ML_CONCEPTS.md](ML_CONCEPTS.md)** - Machine Learning Concepts
   - What is Machine Learning?
   - Text Embeddings explained
   - Vector Similarity
   - Sentence Transformers
   - How MiniMe uses ML concepts
   - **Start here if you're new to ML!**

### File Explanations

2. **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)** - `minime/schemas.py`
   - All Pydantic data models
   - Identity, Memory, Agent, Tool schemas
   - How data validation works
   - **The foundation of all data structures**

3. **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)** - `minime/config.py`
   - Configuration loading
   - Default values
   - YAML file structure
   - **How the system is configured**

4. **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)** - `minime/memory/embeddings.py`
   - EmbeddingModel class
   - How text becomes vectors
   - Lazy loading pattern
   - **The ML engine of MiniMe**

5. **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)** - Identity Layer
   - `minime/identity/principles.py` - IdentityManager
   - `minime/identity/loader.py` - Loading from YAML
   - How principles are stored and used
   - **How MiniMe personalizes responses**

## 🎯 Quick Start Guide

### If you're new to ML:
1. Read **[ML_CONCEPTS.md](ML_CONCEPTS.md)** first
2. Then read **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)**

### If you want to understand the code structure:
1. Start with **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)** (data structures)
2. Then **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)** (configuration)
3. Then **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)** (identity layer)
4. Finally **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)** (ML implementation)

### If you want to understand a specific file:
- Check the corresponding `EXPLANATION_*.md` file
- Each file has detailed explanations, examples, and code flow

## 📖 Document Structure

Each explanation document follows this structure:

1. **Overview** - What the file/module does
2. **File Structure** - High-level organization
3. **Detailed Explanation** - Line-by-line or function-by-function
4. **Examples** - Code examples showing usage
5. **Integration** - How it works with other modules
6. **Key Concepts** - Important ideas explained
7. **Summary** - Takeaways

## 🔍 Finding Information

### Looking for ML concepts?
→ **[ML_CONCEPTS.md](ML_CONCEPTS.md)**

### Want to understand data structures?
→ **[EXPLANATION_SCHEMAS.md](EXPLANATION_SCHEMAS.md)**

### Need to configure the system?
→ **[EXPLANATION_CONFIG.md](EXPLANATION_CONFIG.md)**

### Curious about embeddings?
→ **[EXPLANATION_EMBEDDINGS.md](EXPLANATION_EMBEDDINGS.md)**

### Want to understand personalization?
→ **[EXPLANATION_IDENTITY.md](EXPLANATION_IDENTITY.md)**

## 💡 Tips for Reading

1. **Start with concepts** - Understanding ML concepts makes the code easier
2. **Read in order** - The documents build on each other
3. **Try the examples** - Code examples help solidify understanding
4. **Check integration sections** - Shows how pieces fit together

## 🚀 Next Steps

After reading these documents, you should understand:
- ✅ How MiniMe uses ML for semantic understanding
- ✅ How data is structured and validated
- ✅ How configuration works
- ✅ How identity principles are stored and used
- ✅ How embeddings enable similarity search

Happy learning! 🎓

