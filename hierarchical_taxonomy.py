import os
import time
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
CREDENTIALS_PATH = 'credentials.json'
PROJECT_ID = 'noamarazi'
LOCATION = 'us-central1'
INPUT_FILE = 'list-main-tech.csv'
OUTPUT_FILE = 'hierarchical_taxonomy_results.xlsx'

# פרמטרים - Coherence-Aware Splitting
MIN_LEAF_SIZE = 3           # גודל מינימלי לעלה
MAX_LEAF_SIZE = 12          # הורדה מ-20 ל-12!
MAX_DEPTH = 4               # עומק מקסימלי
COHERENCE_THRESHOLD = 0.05  # סף לזיהוי קבוצה הטרוגנית

# --- AUTHENTICATION & INIT ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    gen_model = GenerativeModel("gemini-2.5-pro") 
    print("✅ Connected to Vertex AI")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit()

def get_batch_embeddings(texts, batch_size=5):
    """מייצר אמבדינגס בקבוצות כדי לא לחרוג ממגבלות"""
    embeddings = []
    print(f"   ⏳ Embedding {len(texts)} items...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            result = embedding_model.get_embeddings(batch)
            embeddings.extend([e.values for e in result])
            time.sleep(0.2) 
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            embeddings.extend([[0.0]*768 for _ in range(len(batch))]) 
    return np.array(embeddings)

def create_clean_context(row):
    """context נקי ופשוט"""
    name = str(row['technology_name'])
    desc = str(row['Technology_Description'])
    return f"Technology: {name}. Description: {desc}"

def name_cluster_with_ai(titles_list, parent_name=None):
    """
    נותן שם לקבוצה - עם הקשר להורה אם קיים
    """
    sample_titles = titles_list[:30]
    list_text = "\n".join([f"- {t}" for t in sample_titles])

    forbidden = "Do NOT use generic words like: Frontier, Emerging, Exotic, Advanced, Novel, Breakthrough, Cutting-edge, Next-gen, Innovative, Future."

    if parent_name:
        prompt = f"""
Here is a sub-group of technologies under "{parent_name}":
{list_text}

Task: Provide a short, SPECIFIC Category Name (2-5 words).
{forbidden}
The name must describe the TECHNICAL DOMAIN, not how "new" or "advanced" it is.
Output ONLY the category name.
"""
    else:
        prompt = f"""
Here is a list of technologies grouped together:
{list_text}

Task: Provide a short, SPECIFIC Category Name (2-4 words).
{forbidden}
Focus on the TECHNICAL DOMAIN (e.g., "DNA Energy Storage", "Quantum Sensors", "Weather Modification").
Output ONLY the category name.
"""

    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip().replace('"', '').replace("Category Name:", "").replace("Category:", "")
    except Exception as e:
        print(f"Naming error: {e}")
        return "Unclassified"

def check_cluster_coherence(embeddings):
    """
    בודק את הקוהרנטיות הפנימית של קבוצה
    מחזיר silhouette score אם אפשר לחלק, אחרת None
    """
    if len(embeddings) < 4:
        return None

    try:
        # ניסיון חלוקה ל-2 כדי לבדוק קוהרנטיות
        Z = linkage(embeddings, method='ward')
        labels = fcluster(Z, t=2, criterion='maxclust')
        if len(set(labels)) < 2:
            return None
        score = silhouette_score(embeddings, labels)
        return score
    except:
        return None


def find_distance_threshold(embeddings, target_clusters_range=(5, 15), mode="normal"):
    """
    מוצא את ה-distance threshold שמייצר מספר קבוצות בטווח הרצוי
    mode="scatter" - יוצר הרבה קבוצות קטנות
    """
    Z = linkage(embeddings, method='ward')
    distances = Z[:, 2]

    # ב-scatter mode, נרצה יותר קבוצות
    if mode == "scatter":
        n = len(embeddings)
        target_clusters_range = (max(n // 4, 3), max(n // 3, 4))

    # חיפוש על distance threshold - מהגבוה לנמוך
    for dist in sorted(distances, reverse=True):
        labels = fcluster(Z, t=dist, criterion='distance')
        n_clusters = len(set(labels))
        if target_clusters_range[0] <= n_clusters <= target_clusters_range[1]:
            return Z, dist, n_clusters

    # fallback
    fallback_dist = np.percentile(distances, 85 if mode == "normal" else 70)
    labels = fcluster(Z, t=fallback_dist, criterion='distance')
    n_clusters = len(set(labels))
    return Z, fallback_dist, n_clusters


def should_split_cluster(size, depth, coherence_score=None):
    """
    החלטה משופרת - עם בדיקת קוהרנטיות
    מחזיר: (should_split, mode)
    mode: "normal" או "scatter"
    """
    if depth >= MAX_DEPTH:
        return False, "normal"

    if size < MIN_LEAF_SIZE * 2:  # קטן מדי לחלוקה (צריך לפחות 6)
        return False, "normal"

    # קבוצה גדולה מדי
    if size > MAX_LEAF_SIZE:
        # אם יש מידע על קוהרנטיות והיא נמוכה - scatter mode
        if coherence_score is not None and coherence_score < COHERENCE_THRESHOLD:
            return True, "scatter"
        return True, "normal"

    # בינוני ברמות עליונות
    if size > 10 and depth < 2:
        if coherence_score is not None and coherence_score < COHERENCE_THRESHOLD:
            return True, "scatter"
        return True, "normal"

    return False, "normal"

def hierarchical_cluster(df, embeddings, depth=0, parent_path="", parent_name=None):
    """
    חלוקה רקורסיבית באמצעות distance-based cutting עם זיהוי קוהרנטיות

    Returns: list of dicts with structure:
    {
        'path': 'Bio-Energy/DNA-Based/Proton Storage',
        'name': 'Proton Storage Systems',
        'parent': 'DNA-Based Energy',
        'depth': 3,
        'size': 8,
        'tech_indices': [1, 5, 12, ...]
    }
    """
    indent = "  " * depth
    print(f"{indent}📊 Level {depth}: Analyzing {len(df)} technologies...")

    # בדיקת קוהרנטיות
    coherence = check_cluster_coherence(embeddings)
    if coherence is not None:
        print(f"{indent}   Coherence score: {coherence:.3f}")

    # תנאי עצירה
    should_split, split_mode = should_split_cluster(len(df), depth, coherence)

    if not should_split:
        print(f"{indent}✋ Keeping as leaf (size={len(df)}, depth={depth})")
        name = name_cluster_with_ai(df['technology_name'].tolist(), parent_name)
        return [{
            'path': f"{parent_path}/{name}" if parent_path else name,
            'name': name,
            'parent': parent_name,
            'depth': depth,
            'size': len(df),
            'tech_indices': df.index.tolist(),
            'silhouette_score': coherence  # שמירת ה-silhouette score
        }]

    # קביעת טווח קבוצות לפי גודל ומצב
    if split_mode == "scatter":
        print(f"{indent}🔀 Low coherence detected - using scatter mode")
        target_range = (max(len(df) // 4, 3), max(len(df) // 3, 4))
    elif len(df) > 100:
        target_range = (5, 12)
    elif len(df) > 50:
        target_range = (4, 8)
    else:
        target_range = (2, 5)

    # חישוב linkage וחיתוך
    Z, dist_threshold, n_clusters = find_distance_threshold(embeddings, target_range, mode=split_mode)
    labels = fcluster(Z, t=dist_threshold, criterion='distance')

    print(f"{indent}✂️  Splitting into {n_clusters} groups (distance={dist_threshold:.2f})")

    df_copy = df.copy()
    df_copy['temp_cluster'] = labels

    all_results = []

    # עיבוד רקורסיבי של כל תת-קבוצה
    unique_labels = sorted(set(labels))
    for cluster_id in unique_labels:
        cluster_mask = df_copy['temp_cluster'] == cluster_id
        cluster_df = df_copy[cluster_mask].copy()
        cluster_embeddings = embeddings[cluster_mask.values]

        # מתן שם ראשוני לקבוצה
        temp_name = name_cluster_with_ai(cluster_df['technology_name'].tolist(), parent_name)
        new_path = f"{parent_path}/{temp_name}" if parent_path else temp_name

        print(f"{indent}  📁 Cluster {cluster_id}: {temp_name} ({len(cluster_df)} items)")

        # קריאה רקורסיבית
        sub_results = hierarchical_cluster(
            cluster_df,
            cluster_embeddings,
            depth + 1,
            new_path,
            temp_name
        )

        all_results.extend(sub_results)

    return all_results

def assign_hierarchical_ids(result_df):
    """
    מקצה מספור היררכי לכל רשומה לפי גודל הקבוצות (יורד)
    """
    # חישוב גודל לכל רמה
    level1_sizes = result_df.groupby('category_level_1').size().sort_values(ascending=False)
    level1_id_map = {cat: str(i+1) for i, cat in enumerate(level1_sizes.index)}

    # מספור Level 2 בתוך כל Level 1
    level2_id_map = {}
    for level1_cat in level1_sizes.index:
        level1_data = result_df[result_df['category_level_1'] == level1_cat]
        level2_cats = level1_data[level1_data['category_level_2'] != '']['category_level_2'].value_counts()
        for j, level2_cat in enumerate(level2_cats.index):
            key = (level1_cat, level2_cat)
            level2_id_map[key] = f"{level1_id_map[level1_cat]}.{j+1}"

    # מספור Level 3 בתוך כל Level 2
    level3_id_map = {}
    for (level1_cat, level2_cat), level2_id in level2_id_map.items():
        mask = (result_df['category_level_1'] == level1_cat) & (result_df['category_level_2'] == level2_cat)
        level3_cats = result_df[mask & (result_df['category_level_3'] != '')]['category_level_3'].value_counts()
        for k, level3_cat in enumerate(level3_cats.index):
            key = (level1_cat, level2_cat, level3_cat)
            level3_id_map[key] = f"{level2_id}.{k+1}"

    # מספור Level 4 בתוך כל Level 3
    level4_id_map = {}
    for (level1_cat, level2_cat, level3_cat), level3_id in level3_id_map.items():
        mask = (result_df['category_level_1'] == level1_cat) & \
               (result_df['category_level_2'] == level2_cat) & \
               (result_df['category_level_3'] == level3_cat)
        level4_cats = result_df[mask & (result_df['category_level_4'] != '')]['category_level_4'].value_counts()
        for m, level4_cat in enumerate(level4_cats.index):
            key = (level1_cat, level2_cat, level3_cat, level4_cat)
            level4_id_map[key] = f"{level3_id}.{m+1}"

    # הקצאת המספורים לכל שורה
    group_ids = []
    subgroup_ids = []
    subsubgroup_ids = []
    subsubsubgroup_ids = []
    full_numeric_ids = []

    for _, row in result_df.iterrows():
        l1 = row['category_level_1']
        l2 = row['category_level_2']
        l3 = row['category_level_3']
        l4 = row['category_level_4']

        g_id = level1_id_map.get(l1, '')
        sg_id = level2_id_map.get((l1, l2), '') if l2 else ''
        ssg_id = level3_id_map.get((l1, l2, l3), '') if l3 else ''
        sssg_id = level4_id_map.get((l1, l2, l3, l4), '') if l4 else ''

        # full_numeric_id = המספור המלא עד הרמה האחרונה
        if sssg_id:
            full_id = sssg_id
        elif ssg_id:
            full_id = ssg_id
        elif sg_id:
            full_id = sg_id
        else:
            full_id = g_id

        group_ids.append(g_id)
        subgroup_ids.append(sg_id)
        subsubgroup_ids.append(ssg_id)
        subsubsubgroup_ids.append(sssg_id)
        full_numeric_ids.append(full_id)

    result_df['group_id'] = group_ids
    result_df['subgroup_id'] = subgroup_ids
    result_df['subsubgroup_id'] = subsubgroup_ids
    result_df['subsubsubgroup_id'] = subsubsubgroup_ids
    result_df['full_numeric_id'] = full_numeric_ids

    return result_df


def build_taxonomy_dataframe(original_df, cluster_results):
    """
    בונה DataFrame סופי עם כל המידע ההיררכי
    שומר את כל העמודות המקוריות
    """
    # יצירת מיפוי מאינדקס לקבוצות
    index_to_cluster = {}
    for cluster in cluster_results:
        for idx in cluster['tech_indices']:
            index_to_cluster[idx] = cluster

    # הוספת מידע לדאטה פריים המקורי - שמירת כל העמודות
    results = []
    for idx, row in original_df.iterrows():
        cluster_info = index_to_cluster.get(idx, None)
        if cluster_info:
            # שמירת כל העמודות המקוריות
            result_row = row.to_dict()

            # הוספת עמודות הסיווג
            path_parts = cluster_info['path'].split('/')
            result_row.update({
                'category_level_1': path_parts[0],
                'category_level_2': path_parts[1] if len(path_parts) > 1 else '',
                'category_level_3': path_parts[2] if len(path_parts) > 2 else '',
                'category_level_4': path_parts[3] if len(path_parts) > 3 else '',
                'full_path': cluster_info['path'],
                'leaf_category': cluster_info['name'],
                'depth': cluster_info['depth'],
                'cluster_size': cluster_info['size'],
                'silhouette_score': cluster_info.get('silhouette_score', None)
            })
            results.append(result_row)

    return pd.DataFrame(results)


def create_summary_sheet(result_df):
    """
    יוצר גיליון סיכום עם מבנה הקלאסטרים
    """
    summary_rows = []

    # Level 1
    for level1_cat in result_df['category_level_1'].unique():
        l1_data = result_df[result_df['category_level_1'] == level1_cat]
        l1_id = l1_data['group_id'].iloc[0] if 'group_id' in l1_data.columns else ''
        l1_silhouettes = l1_data['silhouette_score'].dropna()
        avg_sil = l1_silhouettes.mean() if len(l1_silhouettes) > 0 else None

        summary_rows.append({
            'numeric_id': l1_id,
            'category_name': level1_cat,
            'parent_name': '-',
            'level': 1,
            'tech_count': len(l1_data),
            'avg_silhouette': round(avg_sil, 3) if avg_sil is not None else None
        })

        # Level 2
        level2_cats = l1_data[l1_data['category_level_2'] != '']['category_level_2'].unique()
        for level2_cat in level2_cats:
            l2_data = l1_data[l1_data['category_level_2'] == level2_cat]
            l2_id = l2_data['subgroup_id'].iloc[0] if 'subgroup_id' in l2_data.columns else ''
            l2_silhouettes = l2_data['silhouette_score'].dropna()
            avg_sil = l2_silhouettes.mean() if len(l2_silhouettes) > 0 else None

            summary_rows.append({
                'numeric_id': l2_id,
                'category_name': level2_cat,
                'parent_name': level1_cat,
                'level': 2,
                'tech_count': len(l2_data),
                'avg_silhouette': round(avg_sil, 3) if avg_sil is not None else None
            })

            # Level 3
            level3_cats = l2_data[l2_data['category_level_3'] != '']['category_level_3'].unique()
            for level3_cat in level3_cats:
                l3_data = l2_data[l2_data['category_level_3'] == level3_cat]
                l3_id = l3_data['subsubgroup_id'].iloc[0] if 'subsubgroup_id' in l3_data.columns else ''
                l3_silhouettes = l3_data['silhouette_score'].dropna()
                avg_sil = l3_silhouettes.mean() if len(l3_silhouettes) > 0 else None

                summary_rows.append({
                    'numeric_id': l3_id,
                    'category_name': level3_cat,
                    'parent_name': level2_cat,
                    'level': 3,
                    'tech_count': len(l3_data),
                    'avg_silhouette': round(avg_sil, 3) if avg_sil is not None else None
                })

                # Level 4
                level4_cats = l3_data[l3_data['category_level_4'] != '']['category_level_4'].unique()
                for level4_cat in level4_cats:
                    l4_data = l3_data[l3_data['category_level_4'] == level4_cat]
                    l4_id = l4_data['subsubsubgroup_id'].iloc[0] if 'subsubsubgroup_id' in l4_data.columns else ''
                    l4_silhouettes = l4_data['silhouette_score'].dropna()
                    avg_sil = l4_silhouettes.mean() if len(l4_silhouettes) > 0 else None

                    summary_rows.append({
                        'numeric_id': l4_id,
                        'category_name': level4_cat,
                        'parent_name': level3_cat,
                        'level': 4,
                        'tech_count': len(l4_data),
                        'avg_silhouette': round(avg_sil, 3) if avg_sil is not None else None
                    })

    summary_df = pd.DataFrame(summary_rows)
    # מיון לפי numeric_id
    summary_df = summary_df.sort_values('numeric_id', key=lambda x: x.apply(
        lambda v: [int(n) for n in str(v).split('.')] if v else [999]
    )).reset_index(drop=True)

    return summary_df

def main():
    # 1. טעינת נתונים
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"📂 Loaded {len(df)} technologies.\n")
    
    # 2. הכנת הטקסט
    df['embedding_text'] = df.apply(create_clean_context, axis=1)
    
    # 3. יצירת וקטורים (פעם אחת)
    embeddings_matrix = get_batch_embeddings(df['embedding_text'].tolist())
    
    print("\n" + "="*80)
    print("🌳 Starting Hierarchical Clustering")
    print("="*80 + "\n")
    
    # 4. קלאסטרינג היררכי
    cluster_results = hierarchical_cluster(df, embeddings_matrix)
    
    print("\n" + "="*80)
    print("📊 Clustering Complete!")
    print("="*80)
    print(f"Total leaf clusters: {len(cluster_results)}")
    
    # הצגת סטטיסטיקה
    depths = [c['depth'] for c in cluster_results]
    sizes = [c['size'] for c in cluster_results]
    print(f"Depth range: {min(depths)} to {max(depths)}")
    print(f"Cluster size range: {min(sizes)} to {max(sizes)}")
    print(f"Average cluster size: {np.mean(sizes):.1f}")
    
    # 5. בניית DataFrame סופי
    result_df = build_taxonomy_dataframe(df, cluster_results)

    # 6. הוספת מספור היררכי
    result_df = assign_hierarchical_ids(result_df)

    # 7. יצירת גיליון סיכום
    summary_df = create_summary_sheet(result_df)

    # 8. שמירה עם 2 גיליונות
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Technologies', index=False)
        summary_df.to_excel(writer, sheet_name='Cluster Summary', index=False)
    
    # הצגת סיכום לפי רמות
    print("\n" + "="*80)
    print("📋 Summary by Level")
    print("="*80)
    
    level1_counts = result_df['category_level_1'].value_counts()
    print(f"\nLevel 1: {len(level1_counts)} categories")
    for cat, count in level1_counts.head(10).items():
        print(f"  {cat}: {count} technologies")
    
    print(f"\n✅ Done! Results saved to {OUTPUT_FILE}")
    print(f"   Use category_level_1 for high-level view")
    print(f"   Use full_path for complete hierarchy")
    print(f"   Use leaf_category for most specific classification")

if __name__ == "__main__":
    main()
