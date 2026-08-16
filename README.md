# Emerging Technology Taxonomy Engine

Automatic hierarchical classification of emerging technologies, built for the Israel
Innovation Authority's national technology foresight cycle.

The engine takes a flat annual list of a few hundred emerging technologies and returns a
named, three-level taxonomy tree — so that human domain experts review coherent families
of related technologies instead of an alphabetical list.

In the 2025 cycle it organised **202 technologies into 12 parent families and 35 final
groups** in roughly ten minutes of runtime. Only 4 of those 35 groups needed a third level —
depth is decided by the data, not imposed by the schema.

```
Biocatalytic Energy Storage
   └─ Bio-enzymatic Storage
        └─ Biocatalytic Conversion
             └─ 3D Structured Sugar Biobattery
```

---

## The problem

Every year the Innovation Authority receives roughly **3,600** candidate emerging
technologies from an automated horizon-scanning system. Filtering by maturity — technologies
proven at lab scale and expected to reach product within 3–8 years — leaves about **200**
worth a serious look.

Those 200 then have to be reviewed by human experts: does this technology actually exist,
is the name real or invented by the scanning AI, and who in Israel is working on it.

Reviewing them one by one is slow, and it is slow for an avoidable reason: **most of the 200
are not independent**. They cluster into the same ecosystems — the same labs, the same
researchers, the same companies. An expert who knows one laser technology usually knows the
five next to it. Reviewing them as a random list throws that leverage away.

So the task is not "label technologies". It is: *group them so that one expert's context is
reused across a whole family.*

### Where this sits

```
automated scan  →  maturity filter  →  ┌ TAXONOMY ┐ → ┌ REVIEWER ┐ →  ecosystem  →  policy
   ~3,600            ~200              │  ENGINE  │   │ ROUTING  │      survey
                                       └──────────┘   └──────────┘
                                       ╰──────── this repo ───────╯
```

The engine runs in two stages. **Stage A** decides how technologies group.
**Stage B** decides who reads each group — and, more importantly, where it is not
confident, so that a human sees the doubt instead of inheriting it silently.

---

## The core design constraint: it has to work next year too

An earlier version of this tool was deterministic: one fixed algorithm, one fixed tree-cut
height, one fixed number of groups.

It produced good results — but only because the parameters had been hand-tuned until they
fit *that year's* data. That is an illusion of success. Emerging technology is not stable
between cycles: one year is dominated by quantum and photonics, the next by bio-energy. A
hand-tuned configuration silently degrades the moment the input distribution shifts, and
re-tuning it demands an expert and a week of work at every annual run.

**So the tool was rebuilt to tune itself at runtime.** It makes no assumption about which
algorithm is correct. It generates ~50 competing clusterings of the data it was actually
given, scores them, and lets the best one win. If next year's data collapses into three
dense topics, it will pick an algorithm suited to large groups; if it arrives noisy and
scattered, it will pick one that isolates noise.

Everything below follows from that constraint.

---

## How it works

```
 1  EMBED        each technology → a 1024-dim semantic vector
                 text = name + Web of Science category + OECD area + description,
                        truncated at 2048 chars, sent in batches of 10
                 model: Cohere Embed v3 (input_type="clustering") on AWS Bedrock

 2  COMPETE      ~50 candidate clusterings from 6 algorithms
                 KMeans · Agglomerative-Ward · GMM · BIRCH · Spectral · HDBSCAN
                 across k = 4…12   (L2: 2…6, L3: 2…4)
                 Spectral is gated to inputs under 200 items — it does not scale
                 scored on tag metrics only — zero API calls

 3  JUDGE        the top 10 candidates go to an LLM, which rates every cluster 1–5
                 for domain coherence
                 re-ranked: 40% LLM judgment + 60% tag metrics

 4  RECURSE      each surviving L1 group is treated as its own dataset and
                 re-clustered into L2, then optionally L3 — a split is kept only
                 if it improves separation past a threshold

 5  REPAIR       orphan reassignment  — a technology stranded far from its centroid
                                        moves to the group it actually belongs to
                 small-leaf merging   — groups of 1–2 items fold into their nearest sibling

 6  NAME         bottom-up: leaves first from raw technology data, then parents from
                 their children's summaries, then the top level from theirs

 7  DELIVER      a styled Excel workbook an analyst can open, sort and annotate
```

### Why the scoring is two-stage

Judging a clustering by asking an LLM is the accurate signal, and it is the expensive one —
one call per cluster, per candidate. Running it across all ~50 candidates would multiply
cost for no benefit, since most candidates are obviously poor.

So cheap statistical metrics act as a **pre-filter**, not as the verdict: they cut 50
candidates to 10, and the LLM decides among those 10. Cost scales with the shortlist, not
with the search space.

This split is not a guess. It came out of the experiment described below — which also showed
why the cheap metrics must never be trusted alone.

The competition is genuine rather than ceremonial, but it does have a usual winner: KMeans
takes roughly 70% of runs. That is itself useful information — it means the alternatives earn
their keep in the remaining 30%, on the years where the data does not behave.

### Why naming happens last, and bottom-up

The first implementation named groups *while* it built them, top-down: name L1, then name its
children, then theirs. That failed in three specific ways. Names were invented from
technology titles alone, so the model over-generalised; parent and child ended up carrying
the same words; and every discarded intermediate candidate had still burned naming calls.

Naming now runs **only after the tree is final**, from the bottom up. A leaf is named from
the raw data of the technologies inside it — the model is asked to identify the shared
engineering mechanism, not to summarise titles. A parent is then named from its children's
summaries, so abstraction is built on what is actually below it.

Three guards are enforced in the prompt:

- a **blocklist of empty words** — *Frontier, Emerging, Advanced, Novel, Breakthrough,
  Next-gen, Dual-Use, Other, Miscellaneous* and similar. In a list where every entry is by
  definition an emerging technology, "Emerging Technologies" carries no information.
- **sibling deduplication** — names already assigned at that level are passed in and
  excluded, so two groups cannot land on near-identical labels.
- **parent-distinctness** — a child may not reuse words from its parent's name, which forces
  each level to add real information rather than echo the one above.

All naming calls run at temperature 0 and return strict JSON.

### No catch-all, by design

The failure mode this domain punishes hardest is a giant "Dual-Use" or "Other" bucket that
absorbs everything hard to place. It looks like a taxonomy and is useless to a reviewer.

The engine treats it as a first-class failure: maximum-cluster share is measured on every
candidate, generic names are blocked at the prompt level, and the recursion refuses to split
groups it cannot split cleanly rather than manufacturing filler categories.

## Stage B — routing to a human, and admitting doubt

`reviewer_assignment.py` takes the finished tree and decides which expert reads which
group. Two decisions define it.

**Classification and anomaly detection are separate model calls.** The first version asked
one call to both assign a batch of groups to domains *and* point out the items that did not
belong. It assigned well and it missed the outliers almost entirely — the second task
quietly loses to the first. Splitting them into one batched call for assignment and one
focused call per group for review found substantially more. This is the finding that
justifies the extra cost, and it is the sort of thing you only learn by reading the output
rather than the accuracy number.

**An anomaly is defined relative to the routing decision, not to semantics.** The obvious
implementation flags any technology that looks different from its group. That produces a
flood of flags that cost the reviewer attention and change nothing — because two
technologies with different mechanisms that land on the *same* reviewer are not a problem;
that person reads both either way. So the definition was moved onto the decision itself:

```
group   tight     one reviewer, one shared mechanism        → nothing to see
        diverse   one reviewer, loose grouping              → routed correctly, flagged loose
        mixed     spans several reviewers, no majority      → the whole group goes to a human

item    yellow    odd one out, but the same reviewer        → minor, noted
        red       belongs to a different reviewer           → this routing is probably wrong
```

A mixed group suppresses its per-item flags entirely: it is already going to manual routing
as a whole, so marking individual rows inside it only adds noise. That rule is enforced in
code and covered by a test, because it is exactly the kind of nicety that erodes.

**Everything the model returns is treated as untrusted.** The domain name is constrained to
a closed enumeration and then folded back onto the canonical list, because a single deformed
character — a lookalike Latin letter inside a non-Latin word — would route an entire file to
the wrong person. Malformed JSON is recovered in three escalating steps, ending in
field-level regex salvage, so that a missing comma costs a formatting retry rather than a
silently dropped group.

A technology whose core sits in one domain and whose delivery sits in another appears in
**both** reviewers' workbooks — as a normal row in its home file, and as a greyed referral
row in the other, pointing back home. Neither reviewer is left assuming someone else covered
it.

### Human-in-the-loop

The engine does not decide anything final. It produces a **proposal** — a tree, with a
written justification for every node, in a format built to be argued with.

Each group carries a 1–2 sentence summary stating the mechanism its members share, so a
reviewer can reject the grouping on its stated reasoning rather than guessing at it. The
Excel output is structured for annotation, and ambiguous cases are surfaced for a domain
lead to rule on instead of being silently resolved. Downstream, expert review is a required
stage of the national process, not an optional check.

---

## Evidence: a 516-configuration experiment

The production settings were not chosen by intuition. `testing/clustering_experiment.py` runs
a full grid:

| Axis | Values |
|---|---|
| Embedding model | Cohere Embed v3, Amazon Titan Embed v2 |
| Text composition | 5 variants, from name+description to all six fields |
| Clustering | KMeans · Ward · Complete · Average · Spectral × k=3–12, HDBSCAN × min_size 5–25 |

**516 configurations**, scored in two stages: cheap tag metrics for all 516, then LLM
coherence for the top 20. The full result table ships with the repo as
`testing/results/experiment_results_public.xlsx` — every configuration with all its metrics.
The sheet mapping real technologies to their assigned clusters is withheld; the scores are
not.

### What it found

**Cohere beat Titan decisively.** Of 516 configurations, 261 used Cohere and 255 used Titan.
All 20 finalists were Cohere; not one Titan configuration survived. Titan's characteristic
failure was linguistic rather than semantic — it built a catch-all holding 46% of the corpus.

**Text composition matters, and more is not better.** The winning template combines name,
Web of Science category, OECD research area and description. Adding the remaining fields
measurably degraded results — extra fields were noise, not signal.

**The cheap metrics do not predict the final answer.** Within the 20 finalists, the rank
correlation between the pre-filter score and the final score is *negative*:

```
spearman(stage-1 score, final score)     rho = -0.20   p = 0.41
spearman(stage-1 score, LLM coherence)   rho = -0.43   p = 0.06

the configuration ranked  #1  by cheap metrics finished  #16 of 20
```

This is the finding the architecture is built around. Had the pipeline trusted its
statistical pre-filter, it would have shipped the configuration the human-facing judge liked
*least*. The pre-filter earns its place by being cheap enough to run everywhere — not by
being right.

**Honest limit on that claim:** the correlation is measured only across the 20 shortlisted
configurations, because only those have LLM scores. That is a restricted range, so this is
strong evidence that cheap metrics are insufficient — not a clean measurement of how they
behave across the whole space.

**Silhouette scores mislead in this domain.** A low score (0.07) frequently marked a genuinely
tight cluster; a high one (0.25) frequently marked a mixed bag. It is weighted at 5% in the
final score and is never used alone.

---

## Running it

```bash
pip install -r requirements.txt
cp config.example.py config.py     # add AWS credentials with Bedrock access
python hierarchical_taxonomy.py
```

A synthetic sample dataset ships with the repo so the pipeline can be run end to end without
the source data:

```bash
# data/sample_technologies.csv — 42 invented technologies across 6 domains,
# same schema as production input
```

The Stage B tests run offline, with no AWS account and no model calls:

```bash
pytest tests/ -q          # 16 tests, ~0.5s
```

Required input columns: `technology_name`, `Technology_Description`,
`Web_of_Science_Category`, `OECD_Research_Area`, `emerging_tech_concepts`, `applied_domain`.

Output is `hierarchical_taxonomy_results.xlsx` with three sheets — the technologies with
their assigned path and hierarchical ID, a cluster summary with per-node justifications, and
a styled overview.

A Streamlit interface is the intended entry point for non-technical users — it loads the
input file, runs the pipeline and lets an analyst walk the resulting tree:

```bash
streamlit run app.py
```

### What is safe to tune

The engine is meant to be handed over and kept running by someone else, so its knobs are
documented by **blast radius** rather than by location:

| Parameter | Risk of changing |
|---|---|
| Embedding text composition | **Very high** — the single largest driver of output quality. Never change without re-running the experiment grid. |
| Embedding model, final scoring weights | **High** — both were selected empirically; changing either invalidates the calibration. |
| k ranges, split thresholds, shortlist size, LLM temperature | Medium — affects granularity and reproducibility. |
| Text model, token ceilings, blocked-word list | Low — swap freely. |

The prompts themselves are the most powerful tuning surface and the most dangerous one; they
live as strings in the code and should be tested on a small sample before a full run.

**Requirements:** Python 3.10+, an AWS account with Bedrock access to Cohere Embed v3 and an
Anthropic text model. The engine queries Bedrock at startup and selects the best text model
actually available in the account, rather than hard-coding a model ID that may be
deprecated or unavailable.

---

## Scope and confidentiality

Two things in this repository are redacted, and only those two:

- **The source technology list.** It is the output of the Innovation Authority's
  horizon-scanning cycle and is not published. A synthetic sample with the same schema ships
  in its place. Results on the real data appear only as the aggregate figures already
  published in the project poster.
- **The reviewer taxonomy.** `reviewer_assignment.py` runs against a client-owned taxonomy of
  domains and sub-domains. The system prompt that carried it is blanked out with a note
  describing its structure, and the domain names are replaced by `DOMAIN_1 … DOMAIN_4`. The
  code around them is untouched.

Nothing else is withheld: the architecture, the scoring, the anomaly model, the parsing and
the workbook layout are all here as they ran. No credentials appear in this repository or
anywhere in its history.

Built as a practicum project at the Israel Innovation Authority (Infrastructure Division),
Lauder School of Government, Reichman University, 2026.

## Known limitations

- **Not fully deterministic.** Every LLM call runs at temperature 0, but the judging stage
  still makes the pipeline non-reproducible in the strict sense: re-running on the same input
  can return a different tree. In production this is handled by policy — the taxonomy is
  generated once per cycle and treated as an artefact, not regenerated on demand.
- KMeans forces every technology into some group; there is no "noise" bucket. This is
  intentional — a technology that fits nowhere is surfaced to a human rather than silently
  discarded — but it means every group can contain an outlier.
- Requires AWS Bedrock; there is no offline or local-model path.
- Stage A is a single ~1,900-line module. Its clustering and scoring are covered only by the
  experiment grid, not by unit tests; the offline suite covers Stage B's routing and flagging.
- Cluster names are generated in English by design, so that groups can later be compared
  against international technology databases.
- The LLM coherence judge is a single model at temperature 0; it has not been calibrated
  against multiple human raters.

## License

MIT — see [LICENSE](LICENSE).
