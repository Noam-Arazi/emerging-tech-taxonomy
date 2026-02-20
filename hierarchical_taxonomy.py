"""
Hierarchical Taxonomy Tool — Production Version
=================================================
Year-agnostic tool for classifying emerging technologies into a hierarchical
taxonomy. Designed to run on any year's technology list without manual tuning.

Based on experiment results:
  - Embedding: Cohere Embed v3 (cohere.embed-english-v3)
  - Text: technology_name + Technology_Description + WoS + OECD
  - Clustering: Auto-selects best algorithm and k at each level
  - Hierarchy: Stage 1 (L1 groups) → Stage 2 (L2 sub-groups) → optional L3

Algorithms tested at each level (per advisor guidance):
  KMeans, GMM, HDBSCAN, BIRCH, Agglomerative

Usage:
    python hierarchical_taxonomy.py
"""

import os
import time
import json
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    HDBSCAN,
    Birch,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import boto3

from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "list-main-tech.csv"
OUTPUT_FILE = "hierarchical_taxonomy_results.xlsx"

# Bedrock models
EMBEDDING_MODEL_ID = "cohere.embed-english-v3"
TEXT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Clustering search ranges
L1_K_RANGE = range(4, 13)        # Stage 1: try 4-12 L1 groups
L2_K_RANGE = range(2, 7)         # Stage 2: try 2-6 sub-groups within each L1
L3_K_RANGE = range(2, 5)         # Stage 3: try 2-4 sub-sub-groups if needed

# Thresholds
MIN_CLUSTER_FOR_SPLIT = 8        # Don't sub-cluster groups smaller than this
L2_QUALITY_THRESHOLD = 0.10      # Min avg silhouette to accept L2 split
L3_QUALITY_THRESHOLD = 0.05      # Min avg silhouette to accept L3 split
MAX_L1_RETRIES = 3               # Correction loop: retry L1 if L2 quality is bad
LLM_TOP_N = 10                   # How many L1 candidates to LLM-check

# Stage 1 scoring weights (tag metrics only — cheap pre-filter)
METRIC_WEIGHTS = {
    "wos_jaccard": 0.30,
    "concept_coherence": 0.25,
    "domain_overlap": 0.20,
    "oecd_coherence": 0.15,
    "silhouette": 0.10,
}

# Final scoring weights (from experiment — includes LLM coherence)
FINAL_WEIGHTS = {
    "llm_coherence": 0.40,
    "wos_jaccard": 0.20,
    "concept_coherence": 0.15,
    "domain_overlap": 0.10,
    "oecd_coherence": 0.10,
    "silhouette": 0.05,
}


# ============================================================================
# BEDROCK CLIENT
# ============================================================================

def create_bedrock_client():
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    print("Connected to Amazon Bedrock")
    return client


# ============================================================================
# EMBEDDING
# ============================================================================

def compose_text(row):
    """Build embedding text from row fields (v4 template — best from experiment)."""
    name = str(row["technology_name"])
    desc = str(row["Technology_Description"])
    wos = str(row.get("Web_of_Science_Category", ""))
    oecd = str(row.get("OECD_Research_Area", ""))

    parts = [f"Technology: {name}"]
    if wos and wos != "nan":
        parts.append(f"Scientific domains: {wos}")
    if oecd and oecd != "nan":
        parts.append(f"Research areas: {oecd}")
    parts.append(f"Description: {desc}")
    return ". ".join(parts)


def get_embeddings(texts, client):
    """Generate embeddings using Cohere Embed v3 on Bedrock."""
    embeddings = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = json.dumps({
            "texts": [t[:2048] for t in batch],
            "input_type": "clustering",
            "truncate": "END",
        })
        response = client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.extend(result["embeddings"])
        time.sleep(0.1)

        if (i + batch_size) % 50 == 0:
            print(f"   ... {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(embeddings)


# ============================================================================
# TAG PARSING & METRICS
# ============================================================================

def parse_semicolon(s):
    """'Cat1; Cat2' -> {'Cat1', 'Cat2'}"""
    if pd.isna(s) or str(s).strip() in ("", "nan"):
        return set()
    return {x.strip() for x in str(s).split(";") if x.strip()}


def parse_list(s):
    """\"['tag1', 'tag2']\" -> {'tag1', 'tag2'}"""
    if pd.isna(s) or str(s).strip() in ("", "nan", "[]"):
        return set()
    return {x.strip().strip("'\"") for x in str(s).strip("[]").split(",") if x.strip().strip("'\"") }


def pairwise_jaccard(sets_list):
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
    return total / count if count else 0.0


def concept_coherence(sets_list):
    all_tags = []
    for s in sets_list:
        all_tags.extend(s)
    if not all_tags:
        return 0.0
    top_tag, _ = Counter(all_tags).most_common(1)[0]
    return sum(1 for s in sets_list if top_tag in s) / len(sets_list)


def compute_tag_metrics(df, labels):
    """Compute weighted-average tag metrics across clusters."""
    wos = df["Web_of_Science_Category"].reset_index(drop=True).apply(parse_semicolon)
    oecd = df["OECD_Research_Area"].reset_index(drop=True).apply(parse_semicolon)
    concepts = df["emerging_tech_concepts"].reset_index(drop=True).apply(parse_list)
    domains = df["applied_domain"].reset_index(drop=True).apply(parse_list)
    labels = np.asarray(labels)

    cluster_ids = sorted(set(labels))
    weights = []
    wos_s, concept_s, domain_s, oecd_s = [], [], [], []

    for cid in cluster_ids:
        mask = labels == cid
        w = mask.sum()
        weights.append(w)
        wos_s.append(pairwise_jaccard(wos[mask].tolist()))
        oecd_s.append(pairwise_jaccard(oecd[mask].tolist()))
        domain_s.append(pairwise_jaccard(domains[mask].tolist()))
        concept_s.append(concept_coherence(concepts[mask].tolist()))

    w = np.array(weights, dtype=float)
    w /= w.sum()

    return {
        "wos_jaccard": float(np.dot(wos_s, w)),
        "concept_coherence": float(np.dot(concept_s, w)),
        "domain_overlap": float(np.dot(domain_s, w)),
        "oecd_coherence": float(np.dot(oecd_s, w)),
    }


def composite_score(tag_metrics, sil):
    """Weighted composite score combining tag metrics and silhouette (Stage 1 pre-filter)."""
    sil_norm = (sil + 1) / 2  # [-1,1] -> [0,1]
    return (
        METRIC_WEIGHTS["wos_jaccard"] * tag_metrics["wos_jaccard"]
        + METRIC_WEIGHTS["concept_coherence"] * tag_metrics["concept_coherence"]
        + METRIC_WEIGHTS["domain_overlap"] * tag_metrics["domain_overlap"]
        + METRIC_WEIGHTS["oecd_coherence"] * tag_metrics["oecd_coherence"]
        + METRIC_WEIGHTS["silhouette"] * sil_norm
    )


def check_cluster_coherence_llm(tech_names, tech_descriptions, client):
    """
    Send a cluster to Claude and get a 1-5 coherence rating.
    Copied from the experiment framework (testing/clustering_experiment.py).
    """
    items = []
    for name, desc in zip(tech_names, tech_descriptions):
        short_desc = str(desc)[:150]
        items.append(f"- {name}: {short_desc}")
    items_text = "\n".join(items[:30])  # cap at 30

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
            "temperature": 0,
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
        parsed = json.loads(text)
        return parsed.get("score", 3), parsed.get("reason", "")
    except Exception as e:
        print(f"      LLM coherence error: {e}")
        return 3, "error"


def compute_final_score(tag_metrics, sil, llm_coherence):
    """
    Final weighted score incorporating LLM coherence (from experiment).
    Used to re-rank L1 candidates after LLM check.
    """
    sil_norm = (sil + 1) / 2
    llm_norm = llm_coherence / 5.0  # scale 1-5 to 0-1
    return (
        FINAL_WEIGHTS["llm_coherence"] * llm_norm
        + FINAL_WEIGHTS["wos_jaccard"] * tag_metrics["wos_jaccard"]
        + FINAL_WEIGHTS["concept_coherence"] * tag_metrics["concept_coherence"]
        + FINAL_WEIGHTS["domain_overlap"] * tag_metrics["domain_overlap"]
        + FINAL_WEIGHTS["oecd_coherence"] * tag_metrics["oecd_coherence"]
        + FINAL_WEIGHTS["silhouette"] * sil_norm
    )


def evaluate_l1_with_llm(df, labels, client):
    """
    Run LLM coherence check on all clusters of an L1 candidate.
    Returns weighted-average coherence score (1-5 scale).
    """
    cluster_ids = sorted(set(labels))
    total_score = 0
    total_weight = 0

    for cid in cluster_ids:
        mask = labels == cid
        names = df.loc[mask, "technology_name"].tolist()
        descs = df.loc[mask, "Technology_Description"].tolist()
        score, reason = check_cluster_coherence_llm(names, descs, client)
        weight = mask.sum()
        total_score += score * weight
        total_weight += weight
        time.sleep(0.3)

    return total_score / total_weight if total_weight else 0


# ============================================================================
# CLUSTERING — ALGORITHM SUITE
# ============================================================================

def try_all_algorithms(embeddings, k_range, n_items):
    """
    Try all 5 algorithms across the k range. Return list of (labels, score, description).
    Automatically skips configurations that don't make sense for the data size.
    """
    candidates = []

    for k in k_range:
        if k >= n_items:
            continue

        # 1. KMeans
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
            sil = silhouette_score(embeddings, labels)
            candidates.append((labels, sil, f"KMeans k={k}"))
        except Exception:
            pass

        # 2. Agglomerative (Ward)
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)
            sil = silhouette_score(embeddings, labels)
            candidates.append((labels, sil, f"Agglomerative-Ward k={k}"))
        except Exception:
            pass

        # 3. GMM
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
            labels = gmm.fit_predict(embeddings)
            if len(set(labels)) >= 2:
                sil = silhouette_score(embeddings, labels)
                candidates.append((labels, sil, f"GMM k={k}"))
        except Exception:
            pass

        # 4. BIRCH
        try:
            labels = Birch(n_clusters=k).fit_predict(embeddings)
            if len(set(labels)) >= 2:
                sil = silhouette_score(embeddings, labels)
                candidates.append((labels, sil, f"BIRCH k={k}"))
        except Exception:
            pass

        # 5. Spectral (only for smaller groups — slow on large data)
        if n_items <= 200:
            try:
                from sklearn.cluster import SpectralClustering
                labels = SpectralClustering(
                    n_clusters=k, affinity="rbf", random_state=42, n_init=10,
                ).fit_predict(embeddings)
                sil = silhouette_score(embeddings, labels)
                candidates.append((labels, sil, f"Spectral k={k}"))
            except Exception:
                pass

    # 6. HDBSCAN (parameter-free for k, try a few min_cluster_size values)
    for ms in [5, 8, 10, 15, 20]:
        if ms >= n_items // 2:
            continue
        try:
            hdb = HDBSCAN(min_cluster_size=ms, metric="euclidean")
            labels = hdb.fit_predict(embeddings)
            # Assign noise to nearest cluster
            if -1 in labels:
                valid_clusters = set(labels) - {-1}
                if not valid_clusters:
                    continue
                centers = {c: embeddings[labels == c].mean(axis=0) for c in valid_clusters}
                for idx in np.where(labels == -1)[0]:
                    dists = {c: np.linalg.norm(embeddings[idx] - ctr) for c, ctr in centers.items()}
                    labels[idx] = min(dists, key=dists.get)
            if len(set(labels)) >= 2:
                sil = silhouette_score(embeddings, labels)
                candidates.append((labels, sil, f"HDBSCAN ms={ms}"))
        except Exception:
            pass

    return candidates


def pick_best_clustering(df, embeddings, k_range):
    """
    Pick the best clustering from all algorithms, using the composite score
    (tag metrics + silhouette). Returns (labels, description, score).
    """
    n = len(df)
    candidates = try_all_algorithms(embeddings, k_range, n)

    if not candidates:
        return None, "no valid clustering", 0.0

    best_labels = None
    best_desc = ""
    best_score = -1

    for labels, sil, desc in candidates:
        tag_m = compute_tag_metrics(df, labels)
        score = composite_score(tag_m, sil)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_desc = desc

    return best_labels, best_desc, best_score


# ============================================================================
# CLUSTER NAMING (Claude)
# ============================================================================

def name_cluster(tech_names, parent_name, client, existing_names=None):
    """Name a cluster using Claude on Bedrock, avoiding existing names."""
    sample = tech_names[:30]
    list_text = "\n".join(f"- {t}" for t in sample)

    forbidden = (
        "Do NOT use generic words like: Frontier, Emerging, Exotic, Advanced, "
        "Novel, Breakthrough, Cutting-edge, Next-gen, Innovative, Future, "
        "Dual-Use, Other, Miscellaneous, Unconventional, Various, General."
    )

    avoid = ""
    if existing_names:
        names_list = ", ".join(f'"{n}"' for n in existing_names)
        avoid = f"\nThese names are ALREADY TAKEN by other groups — do NOT reuse them or use very similar names: {names_list}"

    if parent_name:
        prompt = f"""Here is a sub-group of technologies under "{parent_name}":
{list_text}

Task: Provide a short, SPECIFIC Category Name (2-5 words).
{forbidden}{avoid}
The name must describe the TECHNICAL DOMAIN.
Output ONLY the category name."""
    else:
        prompt = f"""Here is a list of technologies grouped together:
{list_text}

Task: Provide a short, SPECIFIC Category Name (2-4 words).
{forbidden}{avoid}
Focus on the TECHNICAL DOMAIN (e.g., "DNA Energy Storage", "Quantum Sensors").
Output ONLY the category name."""

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
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
        return text.replace('"', '').replace("Category Name:", "").replace("Category:", "").strip()
    except Exception as e:
        print(f"   Naming error: {e}")
        return "Unnamed Group"


def name_cluster_bottom_up(items, client, *, is_leaf=True, parent_context=None, existing_names=None):
    """
    Bottom-up cluster naming using Claude.

    For leaf nodes (is_leaf=True):
        items = [{"name": str, "description": str}, ...]
        LLM receives raw technology data and extracts the shared engineering mechanism.

    For parent nodes (is_leaf=False):
        items = [{"cluster_name": str, "summary": str}, ...]
        LLM receives child summaries and abstracts upward to a broader concept.

    Returns: {"cluster_name": str, "summary": str}
    """
    if not items:
        return {"cluster_name": "Unnamed Group", "summary": ""}

    forbidden = (
        "Do NOT use generic words like: Frontier, Emerging, Exotic, Advanced, "
        "Novel, Breakthrough, Cutting-edge, Next-gen, Innovative, Future, "
        "Dual-Use, Other, Miscellaneous, Unconventional, Various, General."
    )

    avoid = ""
    if existing_names:
        names_list = ", ".join(f'"{n}"' for n in existing_names)
        avoid = f"\nThese names are ALREADY TAKEN — do NOT reuse or produce very similar names: {names_list}"

    if is_leaf:
        sample = items[:30]
        lines = [f"- {item['name']}: {str(item['description'])[:200]}" for item in sample]
        items_text = "\n".join(lines)
        context_clause = f' under the parent group "{parent_context}"' if parent_context else ""

        prompt = f"""You are classifying a group of emerging technologies{context_clause}.

Technologies in this group:
{items_text}

Task: Identify the shared engineering or scientific mechanism that unites these technologies.
{forbidden}{avoid}

Return ONLY a valid JSON object — no extra text, no markdown:
{{
  "cluster_name": "<specific 2-5 word name describing the shared technical mechanism or domain>",
  "summary": "<1-2 sentences explaining the common thread that unites these technologies>"
}}"""

    else:
        lines = [f"- {item['cluster_name']}: {item['summary']}" for item in items]
        items_text = "\n".join(lines)
        context_clause = f' under the parent group "{parent_context}"' if parent_context else ""

        prompt = f"""You are classifying sub-groups of technologies{context_clause}.

Sub-groups and their summaries:
{items_text}

Task: Find the higher-level concept that abstracts over ALL these sub-groups. The name must be broader than any individual sub-group but still technically specific — not generic.
{forbidden}{avoid}

Return ONLY a valid JSON object — no extra text, no markdown:
{{
  "cluster_name": "<specific 2-5 word name for the overarching technical domain>",
  "summary": "<1-2 sentences explaining what unifies these sub-groups at a higher level of abstraction>"
}}"""

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 250,
            "temperature": 0,
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

        # Strip markdown code fences if Claude wraps the JSON
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        cluster_name = str(parsed.get("cluster_name", "Unnamed Group")).strip().strip('"')
        summary = str(parsed.get("summary", "")).strip()
        return {"cluster_name": cluster_name, "summary": summary}

    except Exception as e:
        print(f"   Naming error: {e}")
        return {"cluster_name": "Unnamed Group", "summary": ""}


def name_all_clusters_bottom_up(df, l1_labels, l1_names, l2_structure, l3_structure, client):
    """
    Bottom-up naming phase: names all clusters from leaves up to root.

    Called once, after the tree is fully finalized (post merge_small_leaves).
    Overwrites the "name" field in all structures with bottom-up names and
    adds a "summary" field to every cluster.

    Naming order:
      1a. L3 leaf nodes       — named from raw tech data (name + description)
      1b. L2 leaf nodes       — named from raw tech data (no L3 children)
      2.  L2 parent nodes     — named by abstracting over L3 summaries
      3.  L1 nodes            — named by abstracting over L2 summaries
                                (or from raw tech data if L1 was not split)

    Returns: (l1_names, l1_summaries, l2_structure, l3_structure)
      l1_names     — {l1_id: str}  updated L1 names
      l1_summaries — {l1_id: str}  new per-L1 summaries
    """
    print("\n" + "=" * 70)
    print("BOTTOM-UP NAMING PHASE")
    print("=" * 70)

    # Track names accumulated per L1 parent (for intra-L1 deduplication)
    l2_names_by_l1 = {l1_id: [] for l1_id in l2_structure}

    # ── Step 1a: Name L3 leaf nodes ────────────────────────────────────────
    print("\n   Step 1: Naming leaf nodes with raw technology data...")

    for (l1_id, l2_id), l3_groups in l3_structure.items():
        l2_name = l2_structure[l1_id][l2_id].get("name", "")
        existing = []

        for l3_id in sorted(l3_groups.keys()):
            l3_info = l3_groups[l3_id]
            tech_items = [
                {"name": str(df.loc[idx]["technology_name"]),
                 "description": str(df.loc[idx]["Technology_Description"])}
                for idx in l3_info["indices"]
            ]
            result = name_cluster_bottom_up(
                tech_items, client,
                is_leaf=True,
                parent_context=l2_name,
                existing_names=existing,
            )
            l3_info["name"] = result["cluster_name"]
            l3_info["summary"] = result["summary"]
            existing.append(result["cluster_name"])
            print(f"      L3 [{l1_id}.{l2_id}.{l3_id}]: \"{result['cluster_name']}\""
                  f"  ({len(l3_info['indices'])} techs)")
            time.sleep(0.3)

    # ── Step 1b: Name L2 leaf nodes (no L3 children) ───────────────────────
    for l1_id, l2_groups in l2_structure.items():
        l1_parent_name = l1_names.get(l1_id, "")
        has_split = len(l2_groups) > 1

        for l2_id in sorted(l2_groups.keys()):
            if (l1_id, l2_id) in l3_structure:
                continue  # Has L3 children — handled in step 2
            l2_info = l2_groups[l2_id]
            tech_items = [
                {"name": str(df.loc[idx]["technology_name"]),
                 "description": str(df.loc[idx]["Technology_Description"])}
                for idx in l2_info["indices"]
            ]
            result = name_cluster_bottom_up(
                tech_items, client,
                is_leaf=True,
                parent_context=l1_parent_name if has_split else None,
                existing_names=l2_names_by_l1[l1_id],
            )
            l2_info["name"] = result["cluster_name"]
            l2_info["summary"] = result["summary"]
            l2_names_by_l1[l1_id].append(result["cluster_name"])
            label = "L2 leaf" if has_split else "L2 (unsplit L1)"
            print(f"      {label} [{l1_id}.{l2_id}]: \"{result['cluster_name']}\""
                  f"  ({len(l2_info['indices'])} techs)")
            time.sleep(0.3)

    # ── Step 2: Name L2 parent nodes using L3 summaries ────────────────────
    print("\n   Step 2: Naming L2 parent nodes from L3 summaries...")

    for l1_id, l2_groups in l2_structure.items():
        l1_parent_name = l1_names.get(l1_id, "")

        for l2_id in sorted(l2_groups.keys()):
            if (l1_id, l2_id) not in l3_structure:
                continue  # Leaf — already named in step 1
            l2_info = l2_groups[l2_id]
            l3_groups = l3_structure[(l1_id, l2_id)]
            child_items = [
                {"cluster_name": l3_groups[l3_id]["name"],
                 "summary": l3_groups[l3_id].get("summary", "")}
                for l3_id in sorted(l3_groups.keys())
            ]
            result = name_cluster_bottom_up(
                child_items, client,
                is_leaf=False,
                parent_context=l1_parent_name,
                existing_names=l2_names_by_l1[l1_id],
            )
            l2_info["name"] = result["cluster_name"]
            l2_info["summary"] = result["summary"]
            l2_names_by_l1[l1_id].append(result["cluster_name"])
            print(f"      L2 parent [{l1_id}.{l2_id}]: \"{result['cluster_name']}\""
                  f"  ({len(l2_info['indices'])} techs)")
            time.sleep(0.3)

    # ── Step 3: Name L1 nodes ──────────────────────────────────────────────
    print("\n   Step 3: Naming L1 nodes from L2 summaries...")

    l1_summaries = {}
    l1_existing = []

    # Largest L1 group first (consistent ordering)
    l1_ids_sorted = sorted(
        set(l1_labels),
        key=lambda cid: int((l1_labels == cid).sum()),
        reverse=True,
    )

    for l1_id in l1_ids_sorted:
        l2_groups = l2_structure.get(l1_id, {})

        if len(l2_groups) > 1:
            # L1 was split — abstract over L2 summaries
            child_items = [
                {"cluster_name": l2_groups[l2_id]["name"],
                 "summary": l2_groups[l2_id].get("summary", "")}
                for l2_id in sorted(l2_groups.keys())
            ]
            result = name_cluster_bottom_up(
                child_items, client,
                is_leaf=False,
                existing_names=l1_existing,
            )
        else:
            # L1 not split — name directly from raw tech data
            indices = list(l2_groups.values())[0]["indices"] if l2_groups else []
            tech_items = [
                {"name": str(df.loc[idx]["technology_name"]),
                 "description": str(df.loc[idx]["Technology_Description"])}
                for idx in indices[:30]
            ]
            result = name_cluster_bottom_up(
                tech_items, client,
                is_leaf=True,
                existing_names=l1_existing,
            )

        l1_names[l1_id] = result["cluster_name"]
        l1_summaries[l1_id] = result["summary"]
        l1_existing.append(result["cluster_name"])
        n = int((l1_labels == l1_id).sum())
        print(f"      L1 [{l1_id}]: \"{result['cluster_name']}\"  ({n} techs)")
        time.sleep(0.3)

    print("\n   Bottom-up naming complete.")
    return l1_names, l1_summaries, l2_structure, l3_structure


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_stage1(df, embeddings, client):
    """
    Stage 1: Find the best L1 grouping.
    Tries all algorithms across L1_K_RANGE, picks the best by composite score.
    """
    print("=" * 70)
    print("STAGE 1: Finding best L1 grouping")
    print(f"   Testing {len(list(L1_K_RANGE))} k values x 5 algorithms + HDBSCAN")
    print("=" * 70)

    labels, desc, score = pick_best_clustering(df, embeddings, L1_K_RANGE)

    if labels is None:
        print("   ERROR: No valid L1 clustering found!")
        return None, None, 0

    n_clusters = len(set(labels))
    print(f"\n   Best L1: {desc} -> {n_clusters} groups (score={score:.4f})")

    # Show distribution
    counts = Counter(labels)
    for cid in sorted(counts):
        pct = counts[cid] / len(df) * 100
        print(f"      Cluster {cid}: {counts[cid]} techs ({pct:.1f}%)")

    # Name L1 clusters
    l1_names = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        names = df.loc[mask, "technology_name"].tolist()
        l1_names[cid] = name_cluster(names, None, client)
        print(f"      Cluster {cid} -> \"{l1_names[cid]}\"")
        time.sleep(0.3)

    return labels, l1_names, score


def run_stage2(df, embeddings, l1_labels, l1_names, client):
    """
    Stage 2: Sub-cluster each L1 group into L2 sub-groups.
    Only splits groups >= MIN_CLUSTER_FOR_SPLIT.
    Returns dict: {l1_id: {l2_id: {"name": str, "indices": list}}}
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Sub-clustering L1 groups into L2")
    print("=" * 70)

    l2_structure = {}

    for l1_id in sorted(set(l1_labels)):
        mask = l1_labels == l1_id
        l1_name = l1_names[l1_id]
        l1_df = df[mask].reset_index(drop=False)  # keep original index
        l1_emb = embeddings[mask]
        n = len(l1_df)

        print(f"\n   L1: \"{l1_name}\" ({n} techs)")

        if n < MIN_CLUSTER_FOR_SPLIT:
            print(f"      -> Too small to split, keeping as leaf")
            l2_structure[l1_id] = {
                0: {"name": l1_name, "indices": l1_df["index"].tolist()}
            }
            continue

        labels, desc, score = pick_best_clustering(l1_df, l1_emb, L2_K_RANGE)

        if labels is None or score < L2_QUALITY_THRESHOLD:
            print(f"      -> No good L2 split (score={score:.4f}), keeping as leaf")
            l2_structure[l1_id] = {
                0: {"name": l1_name, "indices": l1_df["index"].tolist()}
            }
            continue

        n_sub = len(set(labels))
        print(f"      -> {desc}: {n_sub} sub-groups (score={score:.4f})")

        l2_structure[l1_id] = {}
        for l2_id in sorted(set(labels)):
            sub_mask = labels == l2_id
            sub_indices = l1_df.loc[sub_mask, "index"].tolist()
            l2_name = f"Cluster_{l2_id}"
            print(f"         L2.{l2_id}: {len(sub_indices)} techs")
            l2_structure[l1_id][l2_id] = {
                "name": l2_name,
                "indices": sub_indices,
            }

    return l2_structure


def run_stage3(df, embeddings, l1_labels, l1_names, l2_structure, client):
    """
    Stage 3 (optional): Further split large L2 groups into L3.
    Only splits L2 groups >= MIN_CLUSTER_FOR_SPLIT.
    Returns updated structure with optional L3 level.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: Checking if any L2 group needs further splitting")
    print("=" * 70)

    l3_structure = {}  # {(l1_id, l2_id): {l3_id: {"name", "indices"}}}

    for l1_id, l2_groups in l2_structure.items():
        for l2_id, l2_info in l2_groups.items():
            indices = l2_info["indices"]
            n = len(indices)

            if n < MIN_CLUSTER_FOR_SPLIT:
                continue

            l2_name = l2_info["name"]
            l2_df = df.loc[indices].reset_index(drop=False)
            l2_emb = embeddings[indices]

            labels, desc, score = pick_best_clustering(l2_df, l2_emb, L3_K_RANGE)

            if labels is None or score < L3_QUALITY_THRESHOLD:
                continue

            n_sub = len(set(labels))
            print(f"   L2 \"{l2_name}\" ({n} techs) -> split into {n_sub} L3 groups (score={score:.4f})")

            l3_groups = {}
            for l3_id in sorted(set(labels)):
                sub_mask = labels == l3_id
                sub_indices = l2_df.loc[sub_mask, "index"].tolist()
                l3_name = f"Cluster_{l3_id}"
                print(f"      L3.{l3_id}: {len(sub_indices)} techs")
                l3_groups[l3_id] = {"name": l3_name, "indices": sub_indices}

            l3_structure[(l1_id, l2_id)] = l3_groups

    if not l3_structure:
        print("   No L2 groups needed further splitting.")

    return l3_structure


def evaluate_overall_quality(df, l1_labels, l2_structure, embeddings):
    """Compute overall quality of the two-stage clustering for the correction loop."""
    # Collect all leaf-level labels
    leaf_labels = np.full(len(df), -1, dtype=int)
    label_counter = 0

    for l1_id, l2_groups in l2_structure.items():
        for l2_id, l2_info in l2_groups.items():
            for idx in l2_info["indices"]:
                leaf_labels[idx] = label_counter
            label_counter += 1

    if len(set(leaf_labels)) < 2:
        return 0.0

    tag_m = compute_tag_metrics(df, leaf_labels)
    sil = silhouette_score(embeddings, leaf_labels)
    return composite_score(tag_m, sil)


def consolidate_l1_groups(df, embeddings, l1_labels, l1_names, l2_structure, client,
                          similarity_threshold=0.80):
    """
    Post-Stage-2 refinement: merge L1 groups whose mean embeddings are too similar.
    This prevents fragmentation like 4 separate bio-energy groups when 2 would suffice.

    Returns updated (l1_labels, l1_names, l2_structure).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n" + "=" * 70)
    print("L1 CONSOLIDATION: Checking for similar L1 groups to merge")
    print(f"   Similarity threshold: {similarity_threshold}")
    print("=" * 70)

    l1_labels = np.array(l1_labels, copy=True)
    l1_names = dict(l1_names)
    l2_structure = {k: dict(v) for k, v in l2_structure.items()}  # deep copy

    merged_any = True
    merge_round = 0
    while merged_any:
        merged_any = False
        merge_round += 1
        active_ids = sorted(set(l1_labels))

        if len(active_ids) < 2:
            break

        # Compute centroids for each active L1
        centroids = {}
        for cid in active_ids:
            mask = l1_labels == cid
            centroids[cid] = embeddings[mask].mean(axis=0)

        # Find the most similar pair
        best_sim = -1
        best_pair = None
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                sim = cosine_similarity(
                    centroids[a].reshape(1, -1),
                    centroids[b].reshape(1, -1),
                )[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (a, b)

        if best_sim < similarity_threshold or best_pair is None:
            print(f"   No pair above threshold (best: {best_sim:.4f}). Done.")
            break

        a, b = best_pair
        n_a = (l1_labels == a).sum()
        n_b = (l1_labels == b).sum()
        print(f"\n   Round {merge_round}: Merging L1 \"{l1_names[a]}\" ({n_a} techs) "
              f"+ \"{l1_names[b]}\" ({n_b} techs)  [similarity={best_sim:.4f}]")

        # Merge b into a: update labels
        old_name_a = l1_names[a]
        old_name_b = l1_names[b]
        l1_labels[l1_labels == b] = a

        # Merge L2 structures: append b's sub-groups to a's
        if b in l2_structure:
            next_l2_id = max(l2_structure.get(a, {}).keys(), default=-1) + 1
            for l2_id, l2_info in l2_structure[b].items():
                l2_structure.setdefault(a, {})[next_l2_id] = l2_info
                next_l2_id += 1
            del l2_structure[b]

        # Remove old name for b
        del l1_names[b]

        # Re-name the merged group with Claude (avoid other L1 names)
        other_names = [n for cid, n in l1_names.items() if cid != a]
        merged_techs = df.loc[l1_labels == a, "technology_name"].tolist()
        new_name = name_cluster(merged_techs, None, client, existing_names=other_names)
        l1_names[a] = new_name
        print(f"   -> Merged name: \"{new_name}\" ({len(merged_techs)} techs)")
        print(f"      (was \"{old_name_a}\" + \"{old_name_b}\")")
        time.sleep(0.3)
        merged_any = True

    # Summary
    active_ids = sorted(set(l1_labels))
    print(f"\n   After consolidation: {len(active_ids)} L1 groups")
    for cid in active_ids:
        n = (l1_labels == cid).sum()
        pct = n / len(df) * 100
        print(f"      \"{l1_names[cid]}\": {n} techs ({pct:.1f}%)")

    return l1_labels, l1_names, l2_structure


def correction_loop(df, embeddings, client):
    """
    Correction loop (matches experiment framework):
      1. Try all algorithms → score by tag metrics (cheap, ~50 candidates)
      2. Take top LLM_TOP_N → run LLM coherence check (costs API calls)
      3. Re-rank by final score (40% LLM + 60% tag metrics)
      4. Take top MAX_L1_RETRIES → run Stage 2 + L1 consolidation on each
      5. Pick the best overall
    """
    # ── Phase 1: Score all L1 candidates by tag metrics (cheap) ──
    print("=" * 70)
    print("PHASE 1: Scoring all L1 candidates by tag metrics")
    print("=" * 70)

    all_candidates = try_all_algorithms(embeddings, L1_K_RANGE, len(df))
    print(f"   Generated {len(all_candidates)} clustering candidates")

    scored = []
    for labels, sil, desc in all_candidates:
        tag_m = compute_tag_metrics(df, labels)
        s1_score = composite_score(tag_m, sil)
        scored.append({
            "labels": labels, "sil": sil, "desc": desc,
            "tag_metrics": tag_m, "s1_score": s1_score,
        })
    scored.sort(key=lambda x: x["s1_score"], reverse=True)

    print(f"\n   Top 5 by Stage 1 score (tag metrics only):")
    for i, c in enumerate(scored[:5], 1):
        k = len(set(c["labels"]))
        print(f"      {i}. {c['desc']:<30s}  S1={c['s1_score']:.4f}  k={k}")

    # ── Phase 2: LLM coherence check on top N (from experiment) ──
    top_n = scored[:LLM_TOP_N]
    print(f"\n{'='*70}")
    print(f"PHASE 2: LLM Coherence Check on top {len(top_n)} candidates")
    print("=" * 70)

    for i, c in enumerate(top_n, 1):
        k = len(set(c["labels"]))
        print(f"\n   [{i}/{len(top_n)}] {c['desc']} ({k} clusters)")
        llm_score = evaluate_l1_with_llm(df, c["labels"], client)
        c["llm_coherence"] = llm_score
        c["final_score"] = compute_final_score(c["tag_metrics"], c["sil"], llm_score)
        print(f"      LLM coherence: {llm_score:.2f}/5  |  Final score: {c['final_score']:.4f}")

    # Re-rank by final score
    top_n.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"\n   Re-ranked by final score (40% LLM + 60% tag metrics):")
    for i, c in enumerate(top_n[:5], 1):
        k = len(set(c["labels"]))
        print(f"      {i}. {c['desc']:<30s}  Final={c['final_score']:.4f}  "
              f"LLM={c['llm_coherence']:.2f}  S1={c['s1_score']:.4f}  k={k}")

    # ── Phase 3: Stage 2 + consolidation on top candidates ──
    best_result = None
    best_quality = -1

    for attempt, candidate in enumerate(top_n[:MAX_L1_RETRIES], 1):
        l1_labels = candidate["labels"]
        l1_desc = candidate["desc"]
        n_clusters = len(set(l1_labels))
        print(f"\n{'='*70}")
        print(f"ATTEMPT {attempt}/{MAX_L1_RETRIES}: {l1_desc} ({n_clusters} groups, "
              f"final={candidate['final_score']:.4f}, LLM={candidate['llm_coherence']:.2f})")
        print(f"{'='*70}")

        # Assign placeholder names (final names set in bottom-up naming phase)
        l1_names = {}
        for cid in sorted(set(l1_labels)):
            mask = l1_labels == cid
            count = mask.sum()
            l1_names[cid] = f"Cluster_{cid}"
            print(f"   L1.{cid}: {count} techs")

        # Run Stage 2
        l2_structure = run_stage2(df, embeddings, l1_labels, l1_names, client)

        # Evaluate overall quality
        quality = evaluate_overall_quality(df, l1_labels, l2_structure, embeddings)
        print(f"\n   Overall quality: {quality:.4f}")

        if quality > best_quality:
            best_quality = quality
            best_result = (l1_labels, l1_names, l2_structure, l1_desc)

        if quality >= 0.40:
            print(f"   Quality above threshold — accepting this configuration.")
            break
        else:
            print(f"   Quality below threshold — will try next candidate...")

    return best_result, best_quality


# ============================================================================
# POST-CLUSTERING REFINEMENT
# ============================================================================

def _get_all_leaves(l2_structure, l3_structure):
    """
    Return a list of (key, indices) for every leaf-level group.
    key is either ("l2", l1_id, l2_id) or ("l3", l1_id, l2_id, l3_id).
    """
    leaves = []
    for l1_id, l2_groups in l2_structure.items():
        for l2_id, l2_info in l2_groups.items():
            l3_groups = l3_structure.get((l1_id, l2_id), {})
            if l3_groups:
                for l3_id, l3_info in l3_groups.items():
                    leaves.append((("l3", l1_id, l2_id, l3_id), l3_info["indices"]))
            else:
                leaves.append((("l2", l1_id, l2_id), l2_info["indices"]))
    return leaves


def reassign_orphans(df, embeddings, l1_labels, l2_structure, l3_structure,
                     distance_factor=1.5, improvement_ratio=0.7):
    """
    Post-Stage-3 refinement: detect misplaced techs in small leaf clusters
    and reassign them to a better-fitting cluster.

    A tech is flagged as an orphan if:
      - Its cluster has ≤ 3 members
      - Its distance to cluster centroid > distance_factor * median distance in that cluster

    An orphan is reassigned if:
      - The nearest other cluster's centroid is significantly closer (ratio < improvement_ratio)

    Returns updated (l2_structure, l3_structure).
    """
    from sklearn.metrics.pairwise import cosine_distances

    print("\n" + "=" * 70)
    print("ORPHAN REASSIGNMENT: Detecting misplaced technologies")
    print(f"   Distance factor: {distance_factor}x median | Improvement ratio: {improvement_ratio}")
    print("=" * 70)

    # Deep copy structures
    l2_structure = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in l2_structure.items()}
    l3_structure = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in l3_structure.items()}

    # Collect all leaves with their centroids
    leaves = _get_all_leaves(l2_structure, l3_structure)

    centroids = {}
    for key, indices in leaves:
        if indices:
            centroids[key] = embeddings[indices].mean(axis=0)

    reassigned = 0

    for key, indices in leaves:
        if len(indices) > 3 or len(indices) == 0:
            continue

        centroid = centroids[key]
        # Compute distances of each tech to its centroid
        embs = embeddings[indices]
        dists = cosine_distances(embs, centroid.reshape(1, -1)).flatten()
        median_dist = np.median(dists)

        for local_i, global_idx in enumerate(indices):
            if dists[local_i] <= distance_factor * median_dist:
                continue
            if median_dist == 0 and dists[local_i] == 0:
                continue

            tech_name = df.iloc[global_idx]["technology_name"]
            tech_emb = embeddings[global_idx].reshape(1, -1)

            # Find nearest other leaf
            best_other_key = None
            best_other_dist = float("inf")
            for other_key, other_centroid in centroids.items():
                if other_key == key:
                    continue
                d = cosine_distances(tech_emb, other_centroid.reshape(1, -1))[0, 0]
                if d < best_other_dist:
                    best_other_dist = d
                    best_other_key = other_key

            if best_other_key is None:
                continue

            current_dist = dists[local_i]
            ratio = best_other_dist / current_dist if current_dist > 0 else 1.0

            if ratio >= improvement_ratio:
                continue

            # Perform reassignment
            # Get target leaf name for logging
            if best_other_key[0] == "l3":
                _, tgt_l1, tgt_l2, tgt_l3 = best_other_key
                target_name = l3_structure[(tgt_l1, tgt_l2)][tgt_l3]["name"]
            else:
                _, tgt_l1, tgt_l2 = best_other_key
                target_name = l2_structure[tgt_l1][tgt_l2]["name"]

            # Get source name
            if key[0] == "l3":
                _, src_l1, src_l2, src_l3 = key
                source_name = l3_structure[(src_l1, src_l2)][src_l3]["name"]
            else:
                _, src_l1, src_l2 = key
                source_name = l2_structure[src_l1][src_l2]["name"]

            print(f"   REASSIGN: \"{tech_name}\"")
            print(f"      FROM: \"{source_name}\" (dist={current_dist:.4f})")
            print(f"      TO:   \"{target_name}\" (dist={best_other_dist:.4f}, ratio={ratio:.3f})")

            # Remove from source
            if key[0] == "l3":
                _, src_l1, src_l2, src_l3 = key
                l3_structure[(src_l1, src_l2)][src_l3]["indices"].remove(global_idx)
            else:
                _, src_l1, src_l2 = key
                l2_structure[src_l1][src_l2]["indices"].remove(global_idx)

            # Add to target
            if best_other_key[0] == "l3":
                _, tgt_l1, tgt_l2, tgt_l3 = best_other_key
                l3_structure[(tgt_l1, tgt_l2)][tgt_l3]["indices"].append(global_idx)
            else:
                _, tgt_l1, tgt_l2 = best_other_key
                l2_structure[tgt_l1][tgt_l2]["indices"].append(global_idx)

            # Update l1_labels if the tech moved across L1 groups
            if key[0] == "l3":
                src_l1_id = key[1]
            else:
                src_l1_id = key[1]
            if best_other_key[0] == "l3":
                tgt_l1_id = best_other_key[1]
            else:
                tgt_l1_id = best_other_key[1]
            if src_l1_id != tgt_l1_id:
                l1_labels[global_idx] = tgt_l1_id

            reassigned += 1

    # Clean up empty leaves
    for l1_id in list(l2_structure.keys()):
        for l2_id in list(l2_structure[l1_id].keys()):
            if not l2_structure[l1_id][l2_id]["indices"]:
                name = l2_structure[l1_id][l2_id]["name"]
                print(f"   Removed empty L2 leaf: \"{name}\"")
                del l2_structure[l1_id][l2_id]
                if (l1_id, l2_id) in l3_structure:
                    del l3_structure[(l1_id, l2_id)]

    for key in list(l3_structure.keys()):
        for l3_id in list(l3_structure[key].keys()):
            if not l3_structure[key][l3_id]["indices"]:
                name = l3_structure[key][l3_id]["name"]
                print(f"   Removed empty L3 leaf: \"{name}\"")
                del l3_structure[key][l3_id]
        if not l3_structure[key]:
            del l3_structure[key]

    print(f"\n   Total reassignments: {reassigned}")
    return l2_structure, l3_structure


def merge_small_leaves(df, embeddings, l1_labels, l1_names, l2_structure, l3_structure,
                       client, max_leaf_size=2):
    """
    Post-refinement: merge leaf clusters with ≤ max_leaf_size techs into their
    most similar sibling under the same parent.

    Only merges within the same parent — never across L1 groups.
    Re-names merged clusters via Claude.

    Returns updated (l2_structure, l3_structure).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n" + "=" * 70)
    print(f"SMALL LEAF MERGING: Consolidating leaves with ≤ {max_leaf_size} techs")
    print("=" * 70)

    # Deep copy
    l2_structure = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in l2_structure.items()}
    l3_structure = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in l3_structure.items()}

    total_merges = 0

    # --- Pass 1: Merge small L3 leaves within the same L2 parent ---
    for (l1_id, l2_id) in list(l3_structure.keys()):
        l3_groups = l3_structure[(l1_id, l2_id)]
        if len(l3_groups) < 2:
            continue

        merged_any = True
        while merged_any:
            merged_any = False
            # Find a small leaf
            small_ids = [l3_id for l3_id, info in l3_groups.items()
                         if len(info["indices"]) <= max_leaf_size]
            if not small_ids:
                break

            for small_id in small_ids:
                if small_id not in l3_groups:
                    continue
                small_info = l3_groups[small_id]
                small_indices = small_info["indices"]
                if not small_indices:
                    continue

                # Find most similar sibling
                small_centroid = embeddings[small_indices].mean(axis=0).reshape(1, -1)
                best_sib_id = None
                best_sim = -1
                for sib_id, sib_info in l3_groups.items():
                    if sib_id == small_id or not sib_info["indices"]:
                        continue
                    sib_centroid = embeddings[sib_info["indices"]].mean(axis=0).reshape(1, -1)
                    sim = cosine_similarity(small_centroid, sib_centroid)[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_sib_id = sib_id

                if best_sib_id is None:
                    continue

                sib_info = l3_groups[best_sib_id]
                print(f"   MERGE L3: \"{small_info['name']}\" ({len(small_indices)} techs) "
                      f"-> \"{sib_info['name']}\" ({len(sib_info['indices'])} techs) "
                      f"[sim={best_sim:.4f}]")

                # Merge
                sib_info["indices"].extend(small_indices)
                del l3_groups[small_id]
                print(f"      -> Merged into \"{sib_info['name']}\" ({len(sib_info['indices'])} techs)")
                total_merges += 1
                merged_any = True

        # If only one L3 left, collapse it back into L2
        if len(l3_groups) == 1:
            remaining_id = list(l3_groups.keys())[0]
            remaining = l3_groups[remaining_id]
            l2_structure[l1_id][l2_id]["indices"] = remaining["indices"]
            l2_structure[l1_id][l2_id]["name"] = remaining["name"]
            del l3_structure[(l1_id, l2_id)]
            print(f"      Collapsed single L3 back into L2: \"{remaining['name']}\"")

    # --- Pass 2: Merge small L2 leaves within the same L1 parent ---
    for l1_id in sorted(l2_structure.keys()):
        l2_groups = l2_structure[l1_id]
        if len(l2_groups) < 2:
            continue

        merged_any = True
        while merged_any:
            merged_any = False
            small_ids = [l2_id for l2_id, info in l2_groups.items()
                         if len(info["indices"]) <= max_leaf_size
                         and (l1_id, l2_id) not in l3_structure]  # only leaf L2s
            if not small_ids:
                break

            for small_id in small_ids:
                if small_id not in l2_groups:
                    continue
                small_info = l2_groups[small_id]
                small_indices = small_info["indices"]
                if not small_indices:
                    continue

                # Find most similar sibling L2 leaf
                small_centroid = embeddings[small_indices].mean(axis=0).reshape(1, -1)
                best_sib_id = None
                best_sim = -1
                for sib_id, sib_info in l2_groups.items():
                    if sib_id == small_id or not sib_info["indices"]:
                        continue
                    # Only merge into leaf siblings (no L3 children)
                    sib_centroid = embeddings[sib_info["indices"]].mean(axis=0).reshape(1, -1)
                    sim = cosine_similarity(small_centroid, sib_centroid)[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_sib_id = sib_id

                if best_sib_id is None:
                    continue

                sib_info = l2_groups[best_sib_id]
                print(f"   MERGE L2: \"{small_info['name']}\" ({len(small_indices)} techs) "
                      f"-> \"{sib_info['name']}\" ({len(sib_info['indices'])} techs) "
                      f"[sim={best_sim:.4f}]")

                # Merge indices into sibling
                sib_info["indices"].extend(small_indices)

                # If sibling has L3 children, add the merged techs as a new L3 group
                if (l1_id, best_sib_id) in l3_structure:
                    l3_groups = l3_structure[(l1_id, best_sib_id)]
                    new_l3_id = max(l3_groups.keys(), default=-1) + 1
                    l3_groups[new_l3_id] = {
                        "name": small_info["name"],
                        "indices": list(small_indices),
                    }

                del l2_groups[small_id]
                # Clean up any L3 structure for the deleted L2
                if (l1_id, small_id) in l3_structure:
                    del l3_structure[(l1_id, small_id)]

                print(f"      -> Merged into \"{sib_info['name']}\" ({len(sib_info['indices'])} techs)")
                total_merges += 1
                merged_any = True

    # Summary
    leaves = _get_all_leaves(l2_structure, l3_structure)
    size_dist = Counter(len(indices) for _, indices in leaves)
    print(f"\n   Total merges: {total_merges}")
    print(f"   Remaining leaves: {len(leaves)}")
    print(f"   Leaf size distribution: {dict(sorted(size_dist.items()))}")

    return l2_structure, l3_structure


# ============================================================================
# OUTPUT
# ============================================================================

def build_result_dataframe(df, l1_labels, l1_names, l2_structure, l3_structure):
    """Build the final DataFrame with hierarchy columns."""
    rows = []
    for idx, row in df.iterrows():
        r = row.to_dict()
        l1_id = l1_labels[idx]
        r["category_level_1"] = l1_names[l1_id]

        # Find L2
        r["category_level_2"] = ""
        r["category_level_3"] = ""

        for l2_id, l2_info in l2_structure.get(l1_id, {}).items():
            if idx in l2_info["indices"]:
                # If this L1 wasn't split, don't repeat the name as L2
                if len(l2_structure[l1_id]) > 1:
                    r["category_level_2"] = l2_info["name"]

                # Find L3
                l3_groups = l3_structure.get((l1_id, l2_id), {})
                for l3_id, l3_info in l3_groups.items():
                    if idx in l3_info["indices"]:
                        r["category_level_3"] = l3_info["name"]
                        break
                break

        # Build full path
        parts = [r["category_level_1"]]
        if r["category_level_2"]:
            parts.append(r["category_level_2"])
        if r["category_level_3"]:
            parts.append(r["category_level_3"])
        r["full_path"] = " / ".join(parts)
        r["leaf_category"] = parts[-1]

        rows.append(r)

    return pd.DataFrame(rows)


def assign_ids(result_df):
    """Assign hierarchical numeric IDs (1, 1.1, 1.1.1)."""
    # L1 IDs by size
    l1_sizes = result_df.groupby("category_level_1").size().sort_values(ascending=False)
    l1_map = {cat: str(i + 1) for i, cat in enumerate(l1_sizes.index)}

    ids = []
    for _, row in result_df.iterrows():
        l1 = row["category_level_1"]
        l2 = row["category_level_2"]
        l3 = row["category_level_3"]
        fid = l1_map[l1]

        if l2:
            # Compute L2 id on the fly
            l2_cats = result_df[result_df["category_level_1"] == l1]["category_level_2"].value_counts()
            l2_cats = l2_cats[l2_cats.index != ""]
            l2_map = {cat: f"{fid}.{j+1}" for j, cat in enumerate(l2_cats.index)}
            fid = l2_map.get(l2, fid)

            if l3:
                l3_cats = result_df[
                    (result_df["category_level_1"] == l1) &
                    (result_df["category_level_2"] == l2)
                ]["category_level_3"].value_counts()
                l3_cats = l3_cats[l3_cats.index != ""]
                l3_map = {cat: f"{l2_map.get(l2, fid)}.{k+1}" for k, cat in enumerate(l3_cats.index)}
                fid = l3_map.get(l3, fid)

        ids.append(fid)

    result_df["full_numeric_id"] = ids
    return result_df


def build_summary(result_df, summaries=None):
    """Build cluster summary sheet with hierarchical IDs and per-cluster summaries."""
    if summaries is None:
        summaries = {}

    # Build category name → hierarchical numeric ID mapping from result_df
    id_map = {}
    for _, row in result_df.drop_duplicates(["category_level_1"]).iterrows():
        id_map[row["category_level_1"]] = row["full_numeric_id"].split(".")[0]
    l2_rows = result_df[result_df["category_level_2"] != ""].drop_duplicates(["category_level_2"])
    for _, row in l2_rows.iterrows():
        parts = row["full_numeric_id"].split(".")
        id_map[row["category_level_2"]] = ".".join(parts[:2])
    l3_rows = result_df[result_df["category_level_3"] != ""].drop_duplicates(["category_level_3"])
    for _, row in l3_rows.iterrows():
        id_map[row["category_level_3"]] = row["full_numeric_id"]

    rows = []
    for l1 in result_df["category_level_1"].unique():
        l1_data = result_df[result_df["category_level_1"] == l1]
        rows.append({
            "id": id_map.get(l1, ""), "category": l1, "parent": "-",
            "level": 1, "tech_count": len(l1_data), "summary": summaries.get(l1, ""),
        })

        for l2 in l1_data["category_level_2"].unique():
            if not l2:
                continue
            l2_data = l1_data[l1_data["category_level_2"] == l2]
            rows.append({
                "id": id_map.get(l2, ""), "category": l2, "parent": l1,
                "level": 2, "tech_count": len(l2_data), "summary": summaries.get(l2, ""),
            })

            for l3 in l2_data["category_level_3"].unique():
                if not l3:
                    continue
                l3_data = l2_data[l2_data["category_level_3"] == l3]
                rows.append({
                    "id": id_map.get(l3, ""), "category": l3, "parent": l2,
                    "level": 3, "tech_count": len(l3_data), "summary": summaries.get(l3, ""),
                })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} technologies from {INPUT_FILE}\n")

    client = create_bedrock_client()

    # --- Embeddings ---
    print("\nGenerating embeddings with Cohere Embed v3...")
    texts = [compose_text(row) for _, row in df.iterrows()]
    embeddings = get_embeddings(texts, client)
    print(f"   Embeddings shape: {embeddings.shape}\n")

    # --- Correction Loop: Stage 1 + Stage 2 ---
    result, quality = correction_loop(df, embeddings, client)

    if result is None:
        print("ERROR: No valid clustering found!")
        return

    l1_labels, l1_names, l2_structure, l1_desc = result

    # --- Stage 3 (optional) ---
    l3_structure = run_stage3(df, embeddings, l1_labels, l1_names, l2_structure, client)

    # --- Post-clustering refinement ---
    l2_structure, l3_structure = reassign_orphans(
        df, embeddings, l1_labels, l2_structure, l3_structure
    )
    l2_structure, l3_structure = merge_small_leaves(
        df, embeddings, l1_labels, l1_names, l2_structure, l3_structure, client
    )

    # --- Bottom-up naming phase (runs after tree is fully finalized) ---
    l1_names, l1_summaries, l2_structure, l3_structure = name_all_clusters_bottom_up(
        df, l1_labels, l1_names, l2_structure, l3_structure, client
    )

    # Build name → summary lookup for the summary sheet
    summaries = {}
    for l1_id, name in l1_names.items():
        summaries[name] = l1_summaries.get(l1_id, "")
    for l1_id, l2_groups in l2_structure.items():
        for l2_id, l2_info in l2_groups.items():
            summaries[l2_info["name"]] = l2_info.get("summary", "")
    for (l1_id, l2_id), l3_groups in l3_structure.items():
        for l3_id, l3_info in l3_groups.items():
            summaries[l3_info["name"]] = l3_info.get("summary", "")

    # --- Build output ---
    print("\n" + "=" * 70)
    print("BUILDING OUTPUT")
    print("=" * 70)

    result_df = build_result_dataframe(df, l1_labels, l1_names, l2_structure, l3_structure)
    result_df = assign_ids(result_df)
    summary_df = build_summary(result_df, summaries=summaries)

    # Save
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Technologies", index=False)
        summary_df.to_excel(writer, sheet_name="Cluster Summary", index=False)

    # --- Print summary ---
    print(f"\nL1 algorithm: {l1_desc}")
    print(f"Overall quality: {quality:.4f}\n")

    print("L1 Groups:")
    l1_counts = result_df["category_level_1"].value_counts()
    for cat, count in l1_counts.items():
        pct = count / len(df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    total_leaves = result_df["leaf_category"].nunique()
    print(f"\nTotal leaf categories: {total_leaves}")
    print(f"Hierarchy depth: {'3 levels' if l3_structure else '2 levels'}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
