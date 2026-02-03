# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project performs hierarchical taxonomy classification of technologies using **Amazon Bedrock**. It takes a CSV of technology names and descriptions, generates embeddings, and recursively clusters them into a multi-level taxonomy using Ward hierarchical clustering with coherence-aware splitting.

## Running the Project

```bash
python hierarchical_taxonomy.py
```

**Prerequisites:**
1. Copy `config.example.py` to `config.py`
2. Fill in your AWS credentials in `config.py`
3. Input file: `list-main-tech.csv` with columns `technology_name` and `Technology_Description`
4. Output: `hierarchical_taxonomy_results.xlsx`

**Required Python packages:**
- pandas, numpy, scipy, scikit-learn, openpyxl
- boto3

## Architecture

The system implements a coherence-aware recursive clustering approach:

1. **Embedding Generation** (`get_batch_embeddings`): Sends texts to Amazon Titan Embed Text v2 model (1024-dim vectors)

2. **Hierarchical Clustering** (`hierarchical_cluster`): Recursive function that:
   - Checks cluster coherence via silhouette score on a 2-cluster split
   - Decides whether to split based on size, depth, and coherence (scatter mode for low-coherence groups)
   - Uses Ward linkage with distance-based threshold to find optimal cluster count
   - Calls Claude 3 Sonnet (via Bedrock) to name each cluster based on member technologies

3. **Split Decision Logic** (`should_split_cluster`): Controls recursion with `MIN_LEAF_SIZE=3`, `MAX_LEAF_SIZE=12`, `MAX_DEPTH=4`, and `COHERENCE_THRESHOLD=0.05`

4. **Hierarchical ID Assignment** (`assign_hierarchical_ids`): Assigns numeric IDs (1, 1.1, 1.1.1, etc.) sorted by group size (largest first)

5. **Summary Generation** (`create_summary_sheet`): Creates a cluster summary with hierarchy structure, tech counts, and average silhouette scores

6. **Output Builder** (`build_taxonomy_dataframe`): Preserves all original CSV columns and adds classification columns

## Output Structure

The output Excel file (`hierarchical_taxonomy_results.xlsx`) contains two sheets:

### Sheet 1: Technologies
- All original columns from input CSV
- `category_level_1` through `category_level_4`: Category names at each level
- `full_path`: Complete hierarchy path (e.g., "Bio-Energy/DNA-Based/Proton Storage")
- `leaf_category`: Most specific category name
- `depth`, `cluster_size`: Cluster metadata
- `silhouette_score`: Coherence score (-1 to 1, higher = more coherent)
- `group_id`, `subgroup_id`, `subsubgroup_id`, `subsubsubgroup_id`: Hierarchical numbering
- `full_numeric_id`: Complete numeric ID to deepest level

### Sheet 2: Cluster Summary
| Column | Description |
|--------|-------------|
| `numeric_id` | Hierarchical ID (1, 1.1, 1.1.1, etc.) |
| `category_name` | Category name |
| `parent_name` | Parent category ("-" for Level 1) |
| `level` | Hierarchy level (1-4) |
| `tech_count` | Number of technologies |
| `avg_silhouette` | Average silhouette score |

## Key Configuration Parameters

Located at the top of `hierarchical_taxonomy.py`:
- `EMBEDDING_MODEL_ID`: Amazon Titan Embed Text v2 (1024 dimensions)
- `TEXT_MODEL_ID`: Claude 3 Sonnet for cluster naming
- `MIN_LEAF_SIZE`, `MAX_LEAF_SIZE`: Cluster size bounds
- `MAX_DEPTH`: Maximum hierarchy depth
- `COHERENCE_THRESHOLD`: Silhouette score threshold for scatter mode activation

## Credentials

AWS credentials are stored in `config.py` (not tracked in git). Use `config.example.py` as template.
