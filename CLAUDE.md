# CLAUDE.md

## Project Context — The Big Picture

### What is this project?
**Horizon Scanning Technology Classification** — Part of the Israeli national technology foresight process.

### The Pipeline
1. **Horizon Scanning AI** generates ~1000+ emerging technologies
2. **Filtering** by maturity level (TRL) → ~200 technologies remain
3. **OUR STEP: Taxonomy Classification** → Group technologies into hierarchical clusters
4. **Expert Review** → Domain experts validate each group:
   - Does this technology exist?
   - What's the correct name? (Horizon AI sometimes invents names)
   - Do you know labs/research/industry in Israel working on this?
5. **Final Report** → Published national emerging tech report

### Why Clustering Matters
- **Saves expert time**: If you know a laser lab, you probably know about related laser technologies
- **Logical grouping**: Experts review coherent clusters, not random lists
- Last year's "half-strength" classification still significantly shortened the process

### Key Requirements
1. **Year-agnostic**: Must work on any year's technology list without manual tuning
2. **Flexible**: Different trends each year → different cluster sizes are OK
3. **No catch-all**: Avoid generic categories like "Dual-Use", "Other", "Miscellaneous"
4. **Hierarchical**: 3-4 levels (Group → Sub-group → Sub-sub-group)
5. **Deliverable**: Excel file that non-technical users can run yearly

### Why Embeddings (Not Fixed Rules)
- Each year has different emerging topics (2024: quantum, 2025: bio-energy, etc.)
- Fixed categories would become outdated
- Embeddings capture semantic similarity without hardcoded domains

---

## Current Status (2026-02-20)

**Branch:** `main`
**Status:** Production-ready pipeline — clustering + bottom-up naming refactor complete

### What's Fixed (from experiment results)
- **Embedding model**: Cohere Embed v3 (`cohere.embed-english-v3`) — beat Titan on all metrics
- **Text composition**: v4 = `technology_name` + `Web_of_Science_Category` + `OECD_Research_Area` + `Technology_Description`
- **Scoring weights**: Stage 1 (tag pre-filter) and Final (with LLM coherence) — calibrated from 516-config experiment

### What's Flexible (auto-selected each run)
- **Clustering algorithm**: Tries KMeans, Agglomerative-Ward, GMM, BIRCH, Spectral, HDBSCAN
- **Number of clusters (k)**: Searched across k=4-12 for L1, k=2-6 for L2, k=2-4 for L3
- **Selection**: Top 10 by tag metrics → LLM coherence check → final score → best 3 go to Stage 2

### Pipeline Architecture
```
Phase 1: All candidates × tag metrics (0 API calls)
    ↓ top 10
Phase 2: LLM coherence check (~10×k Claude calls)
    ↓ re-rank by 40% LLM + 60% tags
    ↓ top 3
Phase 3: Stage 2 sub-clustering (placeholder names — no LLM)
    ↓ best result
Stage 3: Optional L3 splits (placeholder names — no LLM)
    ↓
Orphan reassignment (misplaced techs)
    ↓
Small leaf merging (≤2 tech leaves, no LLM rename)
    ↓
Bottom-Up Naming Phase  ← NEW (all LLM naming happens here)
    Step 1: Leaf nodes (L3 / unsplit L2) — raw tech data → {name, summary}
    Step 2: L2 parents with L3 children — L3 summaries → {name, summary}
    Step 3: L1 nodes — L2 summaries → {name, summary}
    ↓
Excel output
```

---

## Systematic Experiment (testing/)

### What Was Tested
A full grid search over **516 configurations**:
- **2 embedding models**: Cohere Embed v3, Titan Embed v2
- **5 text compositions**: v1 (name+desc), v2 (+WoS), v3 (+concepts+domain), v4 (+WoS+OECD), v5 (all fields)
- **56 clustering configs**: KMeans/Agglomerative(ward,complete,average)/Spectral × k=3-12, HDBSCAN × min_size=5-25

### Two-Stage Evaluation
1. **Stage 1 (all 516)**: Tag-based metrics (WoS Jaccard, concept coherence, domain overlap, OECD coherence, silhouette) — no API cost beyond embeddings
2. **Stage 2 (top 20)**: LLM coherence check — Claude rates each cluster 1-5 for domain coherence

### Key Findings
| Finding | Detail |
|---------|--------|
| **Best model** | Cohere — all top 20 configs are Cohere, no Titan |
| **Best text** | v4 (name + WoS + OECD + desc) won final ranking |
| **Best algorithm** | KMeans k=11 (final score 0.637, LLM coherence 4.75/5) |
| **Stage 1 ≠ Final** | Stage 1 top-1 (agg-ward-k12) dropped to #15 after LLM check |
| **Max cluster %** | 15.3% (winner) — no catch-all |
| **Titan always fails** | Creates linguistic catch-alls (46%+ in one group) |

### Scoring Weights (from experiment)

**Stage 1 — Pre-filter (tag metrics only, cheap):**
| Metric | Weight |
|--------|--------|
| WoS Jaccard | 30% |
| Concept coherence | 25% |
| Domain overlap | 20% |
| OECD coherence | 15% |
| Silhouette | 10% |

**Final — With LLM coherence:**
| Metric | Weight |
|--------|--------|
| LLM coherence | 40% |
| WoS Jaccard | 20% |
| Concept coherence | 15% |
| Domain overlap | 10% |
| OECD coherence | 10% |
| Silhouette | 5% |

---

## Experiment History (Chronological)

### Baseline: Last Year's Manual Classification
- **Method:** Semi-manual with some automation
- **Result:** 7-8 L1 groups, 37 L2 sub-groups, 61 L3 sub-sub-groups
- **Status:** "Half-strength" but useful — significantly shortened expert review

### Baseline: GCP (text-embedding-004)
- **Result:** 66 categories, 5 L1, largest L1 = 58 (29%)
- **Status:** Was BEST REFERENCE until systematic experiment

### Experiment 1: AWS Titan Embed v2 (pure embedding)
- **Result:** 93 categories, "Dual-Use Technologies" catch-all with 93 techs (46%)
- **Status:** FAILED — massive catch-all cluster

### Experiment 2: AWS Cohere Embed v3 (pure embedding)
- **Result:** ~83 categories, "Unconventional Military Technologies" with 52 techs (26%)
- **Status:** PARTIAL — smaller catch-all than Titan, but still exists

### Experiment 3: Cohere + Parameter Tweaks
- Tried: `MAX_LEAF_SIZE=10`, `COHERENCE_THRESHOLD=0.08`, heterogeneity detection
- **Result:** Over-fragmented (127 categories) or still had catch-all
- **Status:** FAILED — parameter tuning doesn't solve the core problem

### Experiment 4: Two-Stage LLM + Embedding Hybrid
- **Approach:** Claude classifies L1 dynamically, then embeddings cluster L2-L4
- **Result:** Reported as working well (78 categories, 7 L1, no catch-all)
- **Status:** CODE NOT SAVED TO GIT — was lost, led to systematic experiment approach

### Experiment 5: Pure LLM Recursive
- **Approach:** Claude handles ALL classification recursively, no embeddings
- **Result:** 52 techs in "Other", 85 clusters with 1-2 techs
- **Status:** FAILED — LLM can't reliably match technology names back, over-fragmentation

### Experiment 6: Titan + WoS Field
- **Result:** 94 techs (46.5%) in one group
- **Status:** FAILED — WoS didn't help Titan at all

### Experiment 7: Cohere + WoS (manual config)
- **Result:** 31 leaf clusters, 5 L1, largest 37%, catch-all 20%
- **Status:** BETTER THAN TITAN but worse than GCP baseline

### Experiment 8: Systematic Grid Search (testing/)
- **516 configs** tested: 2 models × 5 text variants × 56 clustering configs
- **Winner:** `cohere_v4_kmeans-k11` (final score 0.637, LLM 4.75/5, max cluster 15.3%)
- **Status:** BEST RESULT — beats GCP baseline, no catch-all

### Current: Production Pipeline with Refinements
- Integrated experiment findings into production code
- Added LLM coherence check to algorithm selection (Phase 2)
- Added post-clustering refinements (L1 consolidation, orphan reassignment, small leaf merging)
- Added duplicate name prevention (existing names passed to Claude)

---

## Key Insights

### What We Learned
1. **Cohere >> Titan** for this domain — Titan creates linguistic catch-alls
2. **v4 text (name+WoS+OECD+desc)** is optimal — v5 (all fields) adds noise
3. **Stage 1 metrics ≠ actual quality** — LLM coherence check is essential
4. **KMeans k=11 won** but this may change yearly — keep algorithm flexible
5. **Pure LLM doesn't work** — can't reliably match names back to data
6. **Generic names are a real problem** — "Frontier Tech" is meaningless when all techs are emerging

### Silhouette Scores Are Misleading
- Low score (0.07) can mean GOOD cluster (very similar items)
- High score (0.25) can mean BAD cluster (mixed unrelated items)
- Use as one signal among many, never as sole criterion

### Naming Architecture (Bottom-Up — current approach)
- **Old approach (Top-Down):** `name_cluster()` called during clustering stages with only tech names → hallucinations, over-generalization, parent-child name collision
- **New approach (Bottom-Up):** All LLM naming runs in `name_all_clusters_bottom_up()` AFTER the tree is fully finalized
  - `name_cluster_bottom_up(items, client, *, is_leaf, parent_context, existing_names)` — two modes:
    - **Leaf mode**: receives `[{name, description}]` → LLM extracts shared engineering mechanism → returns `{cluster_name, summary}`
    - **Parent mode**: receives `[{cluster_name, summary}]` of children → LLM abstracts upward → returns `{cluster_name, summary}`
  - Intermediate stages use placeholder names (`Cluster_N`) — zero wasted API calls
- **Prompt safeguards:**
  - Forbidden generic words list (Frontier, Emerging, Advanced, Novel, etc.)
  - `existing_names` passed at each level for sibling deduplication
  - Leaf with parent: name must NOT reuse words from parent name (prevents L2 echoing L1)
  - Leaf with mixed storage: explicit instruction to name dominant storage type (supercapacitor vs battery)
  - Parent mode: name must be meaningfully different from any sub-group name

---

## Input Data Fields

The input file (`list-main-tech.csv`) contains:

| Field | Description | Used in Embedding? |
|-------|-------------|-------------------|
| `technology_name` | Technology name | YES — primary |
| `Technology_Description` | Long description | YES — primary |
| `Web_of_Science_Category` | Scientific categories (semicolon-separated) | YES — v4 winner |
| `OECD_Research_Area` | Research areas | YES — v4 winner |
| `emerging_tech_concepts` | Concepts like ['DNA-based', 'Quantum'] | NO (used in scoring only) |
| `applied_domain` | Application domain | NO (used in scoring only) |
| `Tech_Level` | Technology level | NO |
| `TRL_*` columns | Technology Readiness Level | NO |

---

## Running the Project

```bash
python hierarchical_taxonomy.py
```

**Prerequisites:**
1. Copy `config.example.py` to `config.py`
2. Fill in AWS credentials in `config.py`
3. Input: `list-main-tech.csv` with required columns
4. Output: `hierarchical_taxonomy_results.xlsx`

**Required packages:** pandas, numpy, scipy, scikit-learn, openpyxl, boto3

**Run the experiment framework (optional):**
```bash
cd testing/
python clustering_experiment.py
```
Results saved to `testing/results/experiment_results.xlsx`. Embeddings cached in `testing/cache/`.

---

## Output Structure

### Sheet 1: Technologies
- All original columns preserved
- `category_level_1` through `category_level_3`: Hierarchy levels
- `full_path`: Complete path (e.g., "Bio-Energy / DNA-Based / Batteries")
- `leaf_category`: Most specific category
- `full_numeric_id`: Hierarchical ID (1.2.3)

### Sheet 2: Cluster Summary
- `id`: Hierarchical numeric ID (1, 1.1, 1.1.1) — derived from assign_ids
- `category`: Category name
- `parent`: Parent category name (`-` for L1)
- `level`: Hierarchy level (1-3)
- `tech_count`: Technologies in category
- `summary`: 1-2 sentence description of the common mechanism (generated by bottom-up naming)

---

## Key Files

| File | Purpose |
|------|---------|
| `hierarchical_taxonomy.py` | Production pipeline — Cohere v4 + multi-algo + LLM coherence + bottom-up naming |
| `app.py` | Streamlit UI — **in progress, not production-ready** |
| `run.command` | Double-click launcher for macOS (runs the pipeline) |
| `setup.command` | One-time setup script for macOS (installs dependencies) |
| `config.py` | AWS credentials (gitignored) |
| `config.example.py` | Template for credentials |
| `list-main-tech.csv` | Input data (202 technologies, 2025 cycle) |
| `hierarchical_taxonomy_results.xlsx` | Latest output |
| `testing/clustering_experiment.py` | Systematic experiment framework (516 configs) |
| `testing/cache/` | Cached embeddings (10 files: 2 models × 5 variants) |
| `testing/results/experiment_results.xlsx` | Experiment results (all configs + top 20 LLM) |

---

## Post-Clustering Refinements

Three refinement steps run after the main clustering pipeline (before naming):

### 1. L1 Consolidation (`consolidate_l1_groups`)
- **Status:** Defined but NOT called — removed from pipeline (caused snowball merge into single group)
- **What it did:** Merged L1 groups with cosine similarity > 0.80

### 2. Orphan Reassignment (`reassign_orphans`)
- **When:** After Stage 3
- **What:** Detects techs in small clusters (≤3) that are far from their centroid (>1.5× median distance), reassigns if another cluster is significantly closer (ratio < 0.7)
- **Why:** Fixes misplaced techs (e.g., Metal 3D printing stuck in "Plasma")

### 3. Small Leaf Merging (`merge_small_leaves`)
- **When:** After orphan reassignment
- **What:** Merges leaf clusters with ≤2 techs into their most similar sibling under the same parent (no LLM rename — placeholder names stay until bottom-up naming)
- **Why:** Reduces over-fragmentation (23 leaves with 2 techs → consolidated)

---

## Success Criteria

1. **No catch-all categories** — No "Other", "Dual-Use", "Miscellaneous"
2. **Coherent groupings** — Technologies in each category are genuinely similar
3. **Reasonable cluster sizes** — No 85 clusters with 1-2 items
4. **Year-agnostic** — Same code works on next year's data without changes
5. **LLM coherence ≥ 4.0/5** — Claude judges clusters as genuinely related
6. **Max L1 ≤ 30%** — No single group dominates
