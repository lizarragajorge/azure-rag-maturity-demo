# Demo script — RAG after Build 2026

A 15–20 minute walkthrough structured around two common RAG questions, with a
"what's new since Build" punchline at the end.

## Before the meeting

```powershell
# pre-warm everything so there's no cold-start awkwardness
.\.venv\Scripts\Activate.ps1
python src/build_index.py                  # idempotent
python src/agentic_demo.py --setup         # idempotent
streamlit run src/app.py
```

Open three browser tabs:
1. The Streamlit app
2. [Azure portal → AI Search → Indexes → utility-ops → Search Explorer](https://portal.azure.com)
3. [src/build_index.py](../src/build_index.py) in VS Code (for the semantic-config code)

## Opening (1 min)

> "Two questions today: how do we make RAG retrieval *reliable*, and what
> does a *semantic configuration* actually look like. I built a demo on a
> synthetic utility-ops corpus — gas-leak procedures, transformer
> maintenance, FERC/NERC compliance notes, customer FAQs. Same shape and
> tone as a real operations corpus, but the company, sources, and phone
> numbers are all fabricated. Three tabs, ~15 minutes."

---

## Question 1 — "How do we make retrieval reliable?"  (6–8 min)

**Open Tab 1: "Reliable retrieval"**

Pick the canned query:

> *"What PPE is required for an arc flash above 25 cal/cm2?"*

Click **Run all four**. While it spins:

> "We're going to hit the same index four different ways. Watch the right-side
> column — that's the semantic ranker score. It's a 0-to-4 relevance grade
> produced by a deep model that Microsoft trains and updates. Above ~1.5 you
> can trust it. Below that, the right move is to *not* answer."

When results land, walk through them left-to-right:

| Pane | Talking point |
|---|---|
| **Keyword (BM25)** | "Finds the doc because 'arc flash' and '25 cal' literally appear. But if the user had said 'protective gear for high-energy arc events' — pure keyword would miss it." |
| **Vector** | "Now we get the conceptual match. But notice — sometimes a wildfire-mitigation doc shows up because 'protective' is semantically close. No hard signal saying which is *more* relevant." |
| **Hybrid (RRF)** | "Reciprocal Rank Fusion combines both rankings. This is the **reliable default**. It's what most production RAG should use as a floor." |
| **Hybrid + Semantic ranker** | "And here's the move. The semantic ranker rescored the top 50, returned a real relevance grade, *and* produced an extractive caption — a verbatim snippet from the chunk. That caption is what we feed the LLM, not the whole chunk. Less context, less drift." |

Then drop the reliability checklist:

> "What 'reliable retrieval' means in production:
> 1. **Hybrid as the floor.** Never ship pure vector or pure keyword.
> 2. **Semantic ranker as the precision layer.** Use the score as a confidence threshold.
> 3. **Extract — don't dump.** Captions and answers, not raw chunks.
> 4. **Refuse on low confidence.** If the top reranker score is < 1.5, return 'I don't know' before the LLM has a chance to hallucinate.
> 5. **Eval as a gate, not a one-time test.** Track precision@k and grounded-answer rate per release."

Try the abstain case so they see the refusal pattern:

> *"What's our policy on EV charging in employee parking lots?"*

(Corpus doesn't cover it. Watch the reranker scores all stay low.)

---

## Question 2 — "Show me semantic configurations"  (4–6 min)

**Switch to Tab 2: "Semantic configurations"**

Before clicking anything, switch to VS Code and show
[src/build_index.py](../src/build_index.py) — the `SemanticSearch` block:

```python
SemanticSearch(
    default_configuration_name="default-balanced",
    configurations=[
        SemanticConfiguration(name="default-balanced",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="keywords")],
            )),
        SemanticConfiguration(name="content-only",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
            )),
        SemanticConfiguration(name="title-and-keywords",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="keywords")],
                content_fields=[SemanticField(field_name="content")],
            )),
    ],
)
```

> "A semantic configuration is the index's way of telling the ranker *which
> fields carry the meaning*. Three things to know:
>
> - You can have **many configs on the same index**. Pick at query time.
> - **Order matters** within each list — it's a priority signal.
> - **No `weight` parameter.** Don't try to tune scalars. Tune *which fields
>   and in what order*."

Back to the UI. Pick the canned query:

> *"transformer oil sample interpretation"*

Click **Compare configurations**. Three columns appear.

Walk left-to-right:

- **default-balanced** — DGA doc is #1. Title says "transformer", content explains DGA. All three field types fire.
- **content-only** — Often still #1, but you'll see a sibling chunk surface higher because we threw away the title's hint. *"This is what 90% of teams build by accident — they put content into the index and forget the ranker has no idea what each chunk is *about*."*
- **title-and-keywords** — Heavier title weighting. Best for catalog-style indexes (equipment lists, FAQ banks, customer-facing search) — *worst* if your titles are auto-generated junk.

Pivot question for the room:

> "Which config would you choose for an outage-management knowledge base
> that dispatch operators use? Probably default-balanced. For a
> customer-facing FAQ search? Probably title-and-keywords. **One index, many
> configs.** That's a real cost-saver — you don't reindex per surface."

---

## "What's new since Build last week"  (4–5 min)

**Switch to Tab 3: "Agentic retrieval"**

> "OK — what changed at Build. The big one for RAG is **agentic retrieval**.
> It's GA in REST `2026-04-01`. Instead of you orchestrating
> embed → vector search → rerank → prompt, you ship a **knowledge base** that
> sits on top of the index and exposes a single `/retrieve` endpoint."

Show the canned multi-part question:

> *"If a Grade 1 gas leak is detected next to a substation that is also
> experiencing a CIP-008 reportable cyber incident, what are our concurrent
> notification obligations and field actions in the first hour?"*

> "Notice this is **two questions glued together**. A naive vector search
> sends one embedding and gets back chunks about either gas leaks or cyber
> incidents — usually not both. Watch what the knowledge base does instead."

Click **Ask the knowledge base**.

When the response lands, narrate the three sections:

1. **Synthesized answer** — concise, cites e.g. NGU-OPS-GAS-014 and NGU-SEC-IR-002.
2. **Subqueries the planner produced** — usually 2 to 4. Point at the actual
   subquery strings: *"It decomposed the question. Each subquery is a focused
   vector+keyword+rerank pass. They ran in parallel."*
3. **Grounding chunks** — the actual chunks the synthesizer used, each with a
   reranker score. *"The answer is locked to these. No grounding chunk, no
   sentence in the answer."*

Then the receipts:

> "The activity trace shows tokens-in, tokens-out, elapsed milliseconds per
> step. Two billing meters — Search for the retrieval, OpenAI for the
> synthesis. You can right-size by lowering `retrievalReasoningEffort` from
> `high` to `medium` to `low`. For most ops Q&A, `low` is fine and 5–10x
> cheaper."

Optional second turn (shows multi-turn awareness):

> *"And if there are no field crews available within 30 minutes?"*

It'll fold the prior turn's context into the planning.

---

## Close (2 min)

The summary slide to keep in your head:

| Concern | Lever |
|---|---|
| Recall on rare terms + paraphrases | **Hybrid search** (BM25 + vector via RRF) |
| Precision | **Semantic ranker** (score as confidence threshold) |
| "Which fields matter?" | **Semantic configurations** (multiple per index) |
| Compound / multi-step questions | **Agentic retrieval** (knowledge base) |
| Hallucinations | **Answer synthesis with required citations** + abstain on low reranker score |
| One index, many surfaces | **Multiple semantic configs**, pick at query time |
| Eval as a release gate | Reranker score distribution + grounded-answer rate per query set |

> "If you want, we can fork this against a slice of your real document set —
> the same code works, you only swap the corpus and re-run `build_index.py`.
> Short path to a private POC if you can share a sanitized sample."

---

## Backup queries (if a demo question lands flat)

- *"What's the threshold for a Priority A danger tree?"* (single-doc, easy)
- *"How long do we retain leak survey records?"* (specific number, exact match)
- *"What's the difference between Grade 1, 2, and 3 leaks?"* (multi-section in one doc — shows good chunk fusion)
- *"Can a single worker do their own LOTO?"* (exception path — exercises understanding)
- *"What does PTO mean for solar installs?"* (acronym disambiguation — exercises semantic ranker)

## Backup multi-part agentic queries

- *"Walk me through what happens if a customer reports a high bill that covers an estimated read period right after their AMI install, and they want to escalate to the regulator."*
- *"During a Red Flag Warning that triggers a PSPS, how do we coordinate with our gas system if the same area has interruptible customers?"*
