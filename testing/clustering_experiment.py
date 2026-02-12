"""
Clustering Experiment Framework
================================
Systematic experiment to find the best combination of text composition,
embedding model, and clustering algorithm for grouping 202 emerging
technologies into L1 categories.

Stage 1: Fast tag-based metrics on all ~560 configs (no API cost beyond embeddings)
Stage 2: LLM coherence check on top 20 configs (Claude API)
"""

import os
import time
import json
import warnings
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, HDBSCAN
import boto3

from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

INPUT_FILE = "list-main-tech.csv"
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# --- Text Composition Variants ---
TEXT_VARIANTS = {
    "v1": {
        "fields": ["technology_name", "Technology_Description"],
        "template": "Technology: {technology_name}. Description: {Technology_Description}",
    },
    "v2": {
        "fields": ["technology_name", "Technology_Description", "Web_of_Science_Category"],
        "template": "Technology: {technology_name}. Scientific domains: {Web_of_Science_Category}. Description: {Technology_Description}",
    },
    "v3": {
        "fields": ["technology_name", "Technology_Description", "emerging_tech_concepts", "applied_domain"],
        "template": "Technology: {technology_name}. Concepts: {emerging_tech_concepts}. Domains: {applied_domain}. Description: {Technology_Description}",
    },
    "v4": {
        "fields": ["technology_name", "Technology_Description", "Web_of_Science_Category", "OECD_Research_Area"],
        "template": "Technology: {technology_name}. Scientific domains: {Web_of_Science_Category}. Research areas: {OECD_Research_Area}. Description: {Technology_Description}",
    },
    "v5": {
        "fields": ["technology_name", "Technology_Description", "Web_of_Science_Category",
                    "OECD_Research_Area", "emerging_tech_concepts", "applied_domain"],
        "template": "Technology: {technology_name}. Scientific domains: {Web_of_Science_Category}. Research areas: {OECD_Research_Area}. Concepts: {emerging_tech_concepts}. Domains: {applied_domain}. Description: {Technology_Description}",
    },
}

# --- Embedding Models ---
EMBEDDING_MODELS = {
    "cohere": {
        "model_id": "cohere.embed-english-v3",
        "dimensions": 1024,
        "batch_size": 10,
        "max_text_len": 2048,
    },
    "titan": {
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "batch_size": 1,
        "max_text_len": 8192,
    },
}

# --- Clustering Configurations ---
K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
HDBSCAN_MIN_SIZES = [5, 8, 10, 15, 20, 25]

CLUSTERING_CONFIGS = []
for method in ["ward", "complete", "average"]:
    for k in K_VALUES:
        CLUSTERING_CONFIGS.append({"algo": "agglomerative", "method": method, "k": k})
for k in K_VALUES:
    CLUSTERING_CONFIGS.append({"algo": "kmeans", "k": k})
for k in K_VALUES:
    CLUSTERING_CONFIGS.append({"algo": "spectral", "k": k})
for ms in HDBSCAN_MIN_SIZES:
    CLUSTERING_CONFIGS.append({"algo": "hdbscan", "min_cluster_size": ms})

# --- Stage 1 Weights ---
STAGE1_WEIGHTS = {
    "wos_jaccard": 0.30,
    "concept_coherence": 0.25,
    "domain_overlap": 0.20,
    "oecd_coherence": 0.15,
    "silhouette": 0.10,
}

# --- Final Score Weights (Stage 2) ---
FINAL_WEIGHTS = {
    "llm_coherence": 0.40,
    "wos_jaccard": 0.20,
    "concept_coherence": 0.15,
    "domain_overlap": 0.10,
    "oecd_coherence": 0.10,
    "silhouette": 0.05,
}

TOP_N_FOR_LLM = 20
TEXT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


# ============================================================================
# SECTION 2: EMBEDDING GENERATION
# ============================================================================

def compose_embedding_text(row, variant_key):
    """Build the text string for a given row and text variant."""
    variant = TEXT_VARIANTS[variant_key]
    values = {}
    for field in variant["fields"]:
        val = str(row.get(field, ""))
        if val == "nan" or val == "None":
            val = ""
        values[field] = val
    return variant["template"].format(**values)


def _embed_cohere(texts, client, model_cfg):
    """Call Cohere Embed v3 on Bedrock in batches."""
    embeddings = []
    batch_size = model_cfg["batch_size"]
    max_len = model_cfg["max_text_len"]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = json.dumps({
            "texts": [t[:max_len] for t in batch],
            "input_type": "clustering",
            "truncate": "END",
        })
        response = client.invoke_model(
            modelId=model_cfg["model_id"],
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.extend(result["embeddings"])
        time.sleep(0.1)

    return np.array(embeddings)


def _embed_titan(texts, client, model_cfg):
    """Call Titan Embed v2 on Bedrock one at a time."""
    embeddings = []
    max_len = model_cfg["max_text_len"]

    for i, text in enumerate(texts):
        body = json.dumps({
            "inputText": text[:max_len],
            "dimensions": model_cfg["dimensions"],
            "normalize": True,
        })
        response = client.invoke_model(
            modelId=model_cfg["model_id"],
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])
        time.sleep(0.05)

        if (i + 1) % 50 == 0:
            print(f"      ... {i + 1}/{len(texts)}")

    return np.array(embeddings)


def generate_embeddings(df, model_key, variant_key, client):
    """Generate (or load cached) embeddings for a model+variant combination."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"embeddings_{model_key}_{variant_key}.npy")

    if os.path.exists(cache_path):
        print(f"   [cache] Loading {model_key}/{variant_key}")
        return np.load(cache_path)

    print(f"   [API] Generating {model_key}/{variant_key} ...")
    texts = [compose_embedding_text(row, variant_key) for _, row in df.iterrows()]

    model_cfg = EMBEDDING_MODELS[model_key]
    if model_key == "cohere":
        emb = _embed_cohere(texts, client, model_cfg)
    else:
        emb = _embed_titan(texts, client, model_cfg)

    np.save(cache_path, emb)
    print(f"   [saved] {cache_path}  shape={emb.shape}")
    return emb


# ============================================================================
# SECTION 3: CLUSTERING
# ============================================================================

def run_clustering(embeddings, config):
    """Run a single clustering configuration. Returns labels array (or None on failure)."""
    algo = config["algo"]
    try:
        if algo == "agglomerative":
            metric = "euclidean" if config["method"] == "ward" else "cosine"
            model = AgglomerativeClustering(
                n_clusters=config["k"],
                linkage=config["method"],
                metric=metric,
            )
            return model.fit_predict(embeddings)

        elif algo == "kmeans":
            model = KMeans(n_clusters=config["k"], n_init=10, random_state=42)
            return model.fit_predict(embeddings)

        elif algo == "spectral":
            model = SpectralClustering(
                n_clusters=config["k"],
                affinity="rbf",
                random_state=42,
                n_init=10,
            )
            return model.fit_predict(embeddings)

        elif algo == "hdbscan":
            model = HDBSCAN(
                min_cluster_size=config["min_cluster_size"],
                metric="euclidean",
            )
            labels = model.fit_predict(embeddings)
            # HDBSCAN uses -1 for noise; fold noise into nearest cluster
            if -1 in labels:
                from sklearn.metrics import pairwise_distances
                noise_mask = labels == -1
                if noise_mask.all():
                    return None  # everything is noise
                cluster_centers = {}
                for c in set(labels):
                    if c == -1:
                        continue
                    cluster_centers[c] = embeddings[labels == c].mean(axis=0)
                for idx in np.where(noise_mask)[0]:
                    dists = {c: np.linalg.norm(embeddings[idx] - center)
                             for c, center in cluster_centers.items()}
                    labels[idx] = min(dists, key=dists.get)
            return labels

    except Exception as e:
        return None


def config_label(config):
    """Human-readable label for a clustering config."""
    if config["algo"] == "agglomerative":
        return f"agg-{config['method']}-k{config['k']}"
    elif config["algo"] == "kmeans":
        return f"kmeans-k{config['k']}"
    elif config["algo"] == "spectral":
        return f"spectral-k{config['k']}"
    elif config["algo"] == "hdbscan":
        return f"hdbscan-ms{config['min_cluster_size']}"


# ============================================================================
# SECTION 4: TAG-BASED METRICS
# ============================================================================

def parse_semicolon_field(s):
    """Parse 'Cat1; Cat2; Cat3' -> set of stripped strings."""
    if pd.isna(s) or str(s).strip() in ("", "nan", "None"):
        return set()
    return {x.strip() for x in str(s).split(";") if x.strip()}


def parse_list_field(s):
    """Parse \"['tag1', 'tag2']\" -> set of strings."""
    if pd.isna(s) or str(s).strip() in ("", "nan", "None", "[]"):
        return set()
    s = str(s).strip()
    # Remove brackets and quotes
    s = s.strip("[]")
    items = set()
    for part in s.split(","):
        part = part.strip().strip("'\"")
        if part:
            items.add(part)
    return items


def pairwise_jaccard(sets_list):
    """Mean Jaccard similarity across all pairs. Returns 0 if < 2 non-empty sets."""
    non_empty = [s for s in sets_list if s]
    if len(non_empty) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            union = non_empty[i] | non_empty[j]
            if union:
                total += len(non_empty[i] & non_empty[j]) / len(union)
            count += 1
    return total / count if count > 0 else 0.0


def concept_coherence(sets_list):
    """Fraction of items sharing the single most common tag."""
    all_tags = []
    for s in sets_list:
        all_tags.extend(s)
    if not all_tags:
        return 0.0
    most_common_tag, most_common_count = Counter(all_tags).most_common(1)[0]
    non_empty = [s for s in sets_list if s]
    if not non_empty:
        return 0.0
    has_dominant = sum(1 for s in non_empty if most_common_tag in s)
    return has_dominant / len(sets_list)


def compute_tag_metrics(df, labels):
    """Compute all tag-based metrics for a clustering result."""
    df = df.copy()
    df["_label"] = labels

    wos_sets = df["Web_of_Science_Category"].apply(parse_semicolon_field)
    oecd_sets = df["OECD_Research_Area"].apply(parse_semicolon_field)
    concept_sets = df["emerging_tech_concepts"].apply(parse_list_field)
    domain_sets = df["applied_domain"].apply(parse_list_field)

    cluster_ids = sorted(set(labels))
    n_total = len(df)

    wos_scores, concept_scores, domain_scores, oecd_scores = [], [], [], []
    weights = []

    for cid in cluster_ids:
        mask = df["_label"] == cid
        size = mask.sum()
        weights.append(size)

        wos_scores.append(pairwise_jaccard(wos_sets[mask].tolist()))
        oecd_scores.append(pairwise_jaccard(oecd_sets[mask].tolist()))
        domain_scores.append(pairwise_jaccard(domain_sets[mask].tolist()))
        concept_scores.append(concept_coherence(concept_sets[mask].tolist()))

    weights = np.array(weights, dtype=float)
    w_norm = weights / weights.sum()

    return {
        "wos_jaccard": float(np.dot(wos_scores, w_norm)),
        "concept_coherence": float(np.dot(concept_scores, w_norm)),
        "domain_overlap": float(np.dot(domain_scores, w_norm)),
        "oecd_coherence": float(np.dot(oecd_scores, w_norm)),
    }


def compute_internal_metrics(embeddings, labels):
    """Compute silhouette, Calinski-Harabasz, and Davies-Bouldin scores."""
    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(embeddings):
        return {"silhouette": -1, "calinski_harabasz": 0, "davies_bouldin": 999}

    return {
        "silhouette": float(silhouette_score(embeddings, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
    }


def compute_stage1_score(metrics):
    """Weighted composite score for Stage 1 ranking."""
    # Normalize silhouette from [-1,1] to [0,1]
    sil_norm = (metrics["silhouette"] + 1) / 2
    return (
        STAGE1_WEIGHTS["wos_jaccard"] * metrics["wos_jaccard"]
        + STAGE1_WEIGHTS["concept_coherence"] * metrics["concept_coherence"]
        + STAGE1_WEIGHTS["domain_overlap"] * metrics["domain_overlap"]
        + STAGE1_WEIGHTS["oecd_coherence"] * metrics["oecd_coherence"]
        + STAGE1_WEIGHTS["silhouette"] * sil_norm
    )


def cluster_distribution_info(labels):
    """Return distribution stats (not scored, just reported)."""
    counts = Counter(labels)
    sizes = sorted(counts.values(), reverse=True)
    total = sum(sizes)
    return {
        "n_clusters": len(counts),
        "max_cluster_pct": round(sizes[0] / total * 100, 1),
        "sizes": sizes,
    }


# ============================================================================
# SECTION 5: LLM COHERENCE CHECK
# ============================================================================

def check_cluster_coherence_llm(tech_names, tech_descriptions, client):
    """Send a cluster to Claude and get a 1-5 coherence rating."""
    items = []
    for name, desc in zip(tech_names, tech_descriptions):
        short_desc = str(desc)[:150]
        items.append(f"- {name}: {short_desc}")
    items_text = "\n".join(items[:30])  # cap at 30 to stay within context

    prompt = f"""Here are {len(tech_names)} technologies grouped together in a cluster:

{items_text}

Rate from 1 to 5 how coherent this group is as a scientific/technical domain:
1 = Completely unrelated technologies, random mix
2 = Weak connection, mostly different fields
3 = Some shared themes but mixed domains
4 = Clearly related, same broad technical field
5 = Highly coherent, same specific scientific domain

Are these technologies genuinely from the same field?

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=TEXT_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()
        # Parse JSON from response
        parsed = json.loads(text)
        return parsed.get("score", 3), parsed.get("reason", "")
    except Exception as e:
        print(f"      LLM error: {e}")
        return 3, "error"


def evaluate_top_configs_with_llm(df, top_configs, embeddings_cache, client):
    """Run LLM coherence check on the top N configurations from Stage 1."""
    results = []

    for rank, row in enumerate(top_configs, 1):
        model_key = row["model"]
        variant_key = row["variant"]
        cluster_cfg = row["cluster_config"]
        emb = embeddings_cache[(model_key, variant_key)]
        labels = run_clustering(emb, cluster_cfg)

        if labels is None:
            row["llm_coherence"] = 0
            results.append(row)
            continue

        cluster_ids = sorted(set(labels))
        total_score = 0
        total_weight = 0

        print(f"   [{rank}/{len(top_configs)}] {row['config_id']} — {len(cluster_ids)} clusters")

        for cid in cluster_ids:
            mask = labels == cid
            names = df.loc[mask, "technology_name"].tolist()
            descs = df.loc[mask, "Technology_Description"].tolist()
            score, reason = check_cluster_coherence_llm(names, descs, client)
            weight = mask.sum()
            total_score += score * weight
            total_weight += weight
            time.sleep(0.3)

        row["llm_coherence"] = total_score / total_weight if total_weight else 0
        results.append(row)

    return results


def compute_final_score(row):
    """Compute final weighted score incorporating LLM coherence."""
    sil_norm = (row["silhouette"] + 1) / 2
    llm_norm = row["llm_coherence"] / 5.0  # scale 1-5 to 0-1
    return (
        FINAL_WEIGHTS["llm_coherence"] * llm_norm
        + FINAL_WEIGHTS["wos_jaccard"] * row["wos_jaccard"]
        + FINAL_WEIGHTS["concept_coherence"] * row["concept_coherence"]
        + FINAL_WEIGHTS["domain_overlap"] * row["domain_overlap"]
        + FINAL_WEIGHTS["oecd_coherence"] * row["oecd_coherence"]
        + FINAL_WEIGHTS["silhouette"] * sil_norm
    )


# ============================================================================
# SECTION 6: MAIN LOOP
# ============================================================================

def main():
    # --- 1. Load data ---
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} technologies from {INPUT_FILE}")
    print(f"Text variants: {len(TEXT_VARIANTS)}, Embedding models: {len(EMBEDDING_MODELS)}, Clustering configs: {len(CLUSTERING_CONFIGS)}")
    total_combos = len(TEXT_VARIANTS) * len(EMBEDDING_MODELS) * len(CLUSTERING_CONFIGS)
    print(f"Total configurations: {total_combos}")
    print()

    # --- 2. Connect to Bedrock ---
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    print("Connected to Amazon Bedrock")
    print()

    # --- 3. Generate all embeddings (with caching) ---
    print("=" * 70)
    print("PHASE 1: Generating Embeddings")
    print("=" * 70)

    embeddings_cache = {}
    for model_key in EMBEDDING_MODELS:
        for variant_key in TEXT_VARIANTS:
            emb = generate_embeddings(df, model_key, variant_key, client)
            embeddings_cache[(model_key, variant_key)] = emb

    print(f"\n   Total embedding sets: {len(embeddings_cache)}")
    print()

    # --- 4. Run all clustering + Stage 1 metrics ---
    print("=" * 70)
    print("PHASE 2: Stage 1 — Clustering + Tag-Based Metrics")
    print("=" * 70)

    all_results = []
    done = 0

    for model_key in EMBEDDING_MODELS:
        for variant_key in TEXT_VARIANTS:
            emb = embeddings_cache[(model_key, variant_key)]

            for cluster_cfg in CLUSTERING_CONFIGS:
                labels = run_clustering(emb, cluster_cfg)
                if labels is None:
                    continue

                n_clusters = len(set(labels))
                if n_clusters < 2:
                    continue

                tag_metrics = compute_tag_metrics(df, labels)
                internal = compute_internal_metrics(emb, labels)
                dist_info = cluster_distribution_info(labels)
                stage1 = compute_stage1_score({**tag_metrics, **internal})

                cfg_id = f"{model_key}_{variant_key}_{config_label(cluster_cfg)}"

                all_results.append({
                    "config_id": cfg_id,
                    "model": model_key,
                    "variant": variant_key,
                    "cluster_config": cluster_cfg,
                    "cluster_algo": config_label(cluster_cfg),
                    **tag_metrics,
                    **internal,
                    "n_clusters": dist_info["n_clusters"],
                    "max_cluster_pct": dist_info["max_cluster_pct"],
                    "stage1_score": stage1,
                })

                done += 1
                if done % 100 == 0:
                    print(f"   ... {done} configs evaluated")

    print(f"   Total valid configs: {len(all_results)}")
    print()

    # --- 5. Rank by Stage 1, take top N ---
    all_results.sort(key=lambda r: r["stage1_score"], reverse=True)

    print("=" * 70)
    print(f"PHASE 3: Stage 2 — LLM Coherence Check (top {TOP_N_FOR_LLM})")
    print("=" * 70)

    top_configs = all_results[:TOP_N_FOR_LLM]

    print(f"\n   Top {TOP_N_FOR_LLM} Stage 1 scores:")
    for i, r in enumerate(top_configs, 1):
        print(f"   {i:3d}. {r['config_id']:<45s}  S1={r['stage1_score']:.4f}  k={r['n_clusters']}  max%={r['max_cluster_pct']}")
    print()

    top_with_llm = evaluate_top_configs_with_llm(df, top_configs, embeddings_cache, client)

    # --- 6. Re-rank by final score ---
    for row in top_with_llm:
        row["final_score"] = compute_final_score(row)

    top_with_llm.sort(key=lambda r: r["final_score"], reverse=True)

    # --- 7. Output ---
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Rank':<5} {'Config':<45} {'Final':<7} {'LLM':<5} {'S1':<7} {'WoS':<6} {'Conc':<6} {'Dom':<6} {'OECD':<6} {'Sil':<6} {'k':<4} {'Max%':<6}")
    print("-" * 120)
    for i, r in enumerate(top_with_llm[:10], 1):
        print(
            f"{i:<5d} {r['config_id']:<45s} {r['final_score']:.4f} "
            f"{r['llm_coherence']:.2f} {r['stage1_score']:.4f} "
            f"{r['wos_jaccard']:.3f} {r['concept_coherence']:.3f} "
            f"{r['domain_overlap']:.3f} {r['oecd_coherence']:.3f} "
            f"{r['silhouette']:.3f} {r['n_clusters']:<4d} {r['max_cluster_pct']:.1f}"
        )

    # --- Save to Excel ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "experiment_results.xlsx")

    # Sheet 1: All results
    export_cols = [
        "config_id", "model", "variant", "cluster_algo",
        "wos_jaccard", "concept_coherence", "domain_overlap", "oecd_coherence",
        "silhouette", "calinski_harabasz", "davies_bouldin",
        "n_clusters", "max_cluster_pct", "stage1_score",
    ]
    all_df = pd.DataFrame([{k: r[k] for k in export_cols} for r in all_results])

    # Sheet 2: Top 20 + LLM
    top_cols = export_cols + ["llm_coherence", "final_score"]
    top_df = pd.DataFrame([{k: r.get(k, "") for k in top_cols} for r in top_with_llm])

    # Sheet 3: Winner details — show cluster contents
    winner = top_with_llm[0]
    w_emb = embeddings_cache[(winner["model"], winner["variant"])]
    w_labels = run_clustering(w_emb, winner["cluster_config"])
    detail_df = df[["technology_name", "Web_of_Science_Category"]].copy()
    detail_df["cluster"] = w_labels
    detail_df = detail_df.sort_values("cluster")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="All Results", index=False)
        top_df.to_excel(writer, sheet_name="Top 20 + LLM", index=False)
        detail_df.to_excel(writer, sheet_name="Winner Details", index=False)

    print(f"\nResults saved to {output_path}")
    print(f"  - All Results: {len(all_results)} rows")
    print(f"  - Top 20 + LLM: {len(top_with_llm)} rows")
    print(f"  - Winner: {winner['config_id']} (final_score={winner['final_score']:.4f})")


if __name__ == "__main__":
    main()
