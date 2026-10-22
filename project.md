# RepoMind

## An NLP-Based Framework for Semantic Code Retrieval and Bug Localization

**Project type:** Natural Language Processing (NLP), Information
Retrieval, Code Intelligence, and Software Engineering\
**Primary goal:** Given a natural-language bug report and a software
repository, rank the code files and functions most likely to be involved
in the bug, explain the evidence, and optionally propose a patch that
can be tested safely.

------------------------------------------------------------------------

## 1. Project Overview

Software bug reports are usually written in natural language. They
describe symptoms, error messages, unexpected behavior, or steps to
reproduce a problem. The code responsible for the problem may be buried
in a large repository, spread across multiple functions, or described
using terminology that differs from the wording in the report.

RepoMind is designed to bridge that gap.

It accepts a bug report---initially from a pasted issue or a GitHub
issue URL---and a repository. It analyzes the report, parses the
repository, retrieves relevant code using lexical and semantic search,
ranks candidate files and functions, examines relationships between code
elements, and returns an evidence-backed localization report.

The system may then use a language model to explain the leading
candidates and suggest a patch. Any proposed patch is treated as a
hypothesis until it has been reviewed and tested. RepoMind should not
claim that it has found or fixed a bug merely because an LLM generated a
plausible answer.

### One-sentence description

RepoMind is an NLP-based code intelligence system that uses semantic
retrieval, relevance ranking, and repository context to map
natural-language bug reports to likely code locations and support
evidence-based debugging.

------------------------------------------------------------------------

## 2. The Problem We Are Solving

Consider a report such as:

> "After refreshing the dashboard, users are redirected to the login
> page even though they previously signed in. The problem occurs
> intermittently."

A developer may need to inspect authentication state restoration, token
storage, API request setup, route protection, and related tests.
Searching for the word "dashboard" alone is unlikely to be sufficient.
The relevant function may be named `restoreSession`, `hydrateUser`, or
something else that does not appear in the issue.

The main challenges are:

1.  **Vocabulary mismatch:** The report describes behavior in ordinary
    language, while the code uses identifiers and
    implementation-specific terms.
2.  **Repository scale:** A real project can contain thousands of files
    and functions.
3.  **Context dependency:** A bug in one function may be caused by
    another function, an import, a caller, a shared utility, or a
    configuration.
4.  **Noisy issue reports:** Reports may contain incomplete steps, stack
    traces, logs, duplicated text, or uncertain assumptions.
5.  **Ranking ambiguity:** Several code snippets may look relevant, but
    only a few are genuinely connected to the failure.
6.  **Evaluation difficulty:** A convincing explanation is not proof
    that the system found the right code or produced a correct fix.

RepoMind focuses first on the measurable task of **bug localization**:
finding and ranking relevant code locations. Patch generation is a
later, optional stage.

------------------------------------------------------------------------

## 3. Goals and Non-Goals

### Goals

-   Accept a bug report and a supported source-code repository.
-   Parse source files into meaningful code units such as functions,
    methods, and classes.
-   Preserve metadata for each code unit, including repository, file
    path, symbol name, and line range.
-   Retrieve candidates using lexical and semantic search.
-   Combine retrieval signals and rerank candidate functions.
-   Use code relationships to add relevant context where useful.
-   Show why each candidate was ranked highly, with links or references
    to source lines.
-   Evaluate retrieval and localization using a labeled benchmark.
-   Optionally generate a proposed diagnosis and patch.
-   Test generated patches in an isolated environment before reporting
    test outcomes.
-   Provide a simple interface and API for repeatable use.

### Non-goals for the initial version

-   Guaranteeing automatic detection of every bug.
-   Automatically merging or deploying generated patches.
-   Supporting every programming language from day one.
-   Replacing a developer's review or testing process.
-   Treating LLM confidence or fluent explanations as proof of
    correctness.
-   Training a large language model from scratch.

------------------------------------------------------------------------

## 4. Intended Users and Use Cases

Potential users include:

-   Software developers investigating unfamiliar codebases.
-   Open-source maintainers triaging incoming bug reports.
-   Engineering teams trying to shorten debugging and issue-triage time.
-   Researchers studying natural-language-to-code retrieval and
    automated bug localization.
-   Students evaluating information retrieval and code-language models.

Example use cases:

1.  A developer pastes a GitHub issue and receives a ranked list of
    likely functions.
2.  A maintainer sees related callers, callees, imports, and tests for a
    candidate function.
3.  A researcher compares BM25, dense retrieval, hybrid retrieval,
    reranking, and graph-aware ranking.
4.  A developer requests a patch proposal and reviews the generated diff
    and test results.

------------------------------------------------------------------------

## 5. End-to-End User Experience

A typical workflow is:

1.  **Select the repository.** The user provides a supported GitHub URL
    or a local repository path.
2.  **Provide the issue.** The user pastes a bug report or supplies a
    supported GitHub issue URL.
3.  **Validate the input.** RepoMind checks repository access, supported
    language, issue content, and indexing status.
4.  **Index the repository.** Source files are parsed into functions,
    methods, classes, and other useful code units.
5.  **Understand the issue.** RepoMind extracts useful terms such as
    error messages, stack-trace locations, identifiers, and reproduction
    steps.
6.  **Retrieve candidates.** Lexical and dense retrieval systems
    independently find relevant code units.
7.  **Combine and rerank candidates.** A hybrid retrieval stage combines
    the candidates; a reranker assesses issue--code relevance.
8.  **Expand relevant context.** Where useful, the system inspects
    callers, callees, imports, tests, and nearby code.
9.  **Produce a localization report.** It displays ranked files and
    functions, source locations, relevance evidence, and uncertainty.
10. **Optionally propose a patch.** An LLM receives the issue and
    selected evidence and proposes a minimal change.
11. **Validate the patch.** The patch is checked for formatting or
    syntax where possible and tests are run in an isolated environment.
12. **Review the result.** The user sees the diff, test outcomes,
    limitations, and evidence before deciding what to do next.

If repository indexing fails, the issue is incomplete, or evidence is
insufficient, RepoMind should report the limitation rather than invent a
diagnosis.

------------------------------------------------------------------------

## 6. System Architecture

The project is divided into the following logical components.

### 6.1 Input and Repository Connector

Responsibilities:

-   Accept a repository URL or local path.
-   Fetch issue title, body, labels, comments if explicitly enabled, and
    available stack traces.
-   Retrieve the relevant repository revision.
-   Record the exact commit hash used for analysis.
-   Respect repository permissions, rate limits, and API terms.

For the MVP, public GitHub repositories and pasted issue text are
sufficient. Private repository support can be added later through
secure, user-authorized authentication.

### 6.2 Repository Indexer

Responsibilities:

-   Walk the repository tree.
-   Ignore generated files, binaries, dependency folders, build output,
    and configured exclusions.
-   Detect supported languages.
-   Parse source files into functions, methods, classes, and other code
    chunks.
-   Store file paths, symbol names, line ranges, imports, and available
    call relationships.
-   Build lexical and vector indexes.

Tree-sitter is a candidate parsing technology because it supports
syntax-aware parsing across multiple programming languages. Parsing
success and call-graph completeness will vary by language and code
construct.

### 6.3 Issue Understanding Module

Responsibilities:

-   Normalize issue text without destroying technical tokens.
-   Separate narrative text, code blocks, logs, stack traces, and
    reproduction steps.
-   Extract error types, file paths, function names, API names, and
    other technical entities where possible.
-   Build one or more search queries from the issue.
-   Retain the original report for context.

This stage should preserve identifiers, version strings, punctuation
that matters to stack traces, and code snippets. Aggressive generic text
cleaning can remove precisely the information needed for retrieval.

### 6.4 Retrieval Engine

The retrieval engine finds candidate code units using more than one
method.

**Lexical retrieval:** BM25 or another text-based retrieval method
searches issue terms against code text, identifiers, comments, file
paths, and symbol names.

**Dense retrieval:** An embedding model maps issue text and code chunks
into vector representations. Similarity search finds semantically
related code even when the wording differs.

**Hybrid retrieval:** Lexical and dense results are combined, for
example with reciprocal rank fusion (RRF) or a tuned score combination.

The engine should retain each method's original score and rank so that
the methods can be evaluated independently.

### 6.5 Reranking Module

A cross-encoder or another supervised ranking model receives an
issue--code pair and estimates its relevance. It reranks a smaller
candidate set returned by the first-stage retrievers.

This two-stage design balances broad candidate recall with more
expensive relevance scoring. It also creates a clear experimental
comparison: retrieval alone versus retrieval plus reranking.

### 6.6 Repository Context and Dependency Analysis

The context module expands the initial candidates using repository
relationships, such as:

-   Function callers and callees.
-   Imports and references between modules.
-   Related classes and shared utilities.
-   Tests associated with a function or module.
-   Nearby code and relevant configuration.

Tree-sitter can help identify syntax and some structural relationships.
More complete call-graph construction may require language-specific
static analysis, build information, or language servers. The system must
record which relationships are known versus inferred.

Graph expansion should be controlled. Traversing every dependency can
add noise, so the system should limit expansion depth and evaluate
whether graph context improves ranking.

### 6.7 Ranking and Evidence Assembly

The final ranking can combine:

-   BM25 rank or score.
-   Dense similarity score.
-   Cross-encoder relevance score.
-   Exact matches to stack-trace locations or identifiers.
-   Relationship to other high-ranking candidates.
-   Test or configuration evidence.
-   Repository-specific signals, if available and validated.

Begin with transparent score fusion and establish a baseline before
training a more complex learning-to-rank model.

For every displayed candidate, preserve its file path, line range,
symbol name, and supporting signals. A ranking score is a prioritization
signal, not a calibrated probability that the function contains the bug.

### 6.8 Diagnosis and Patch Proposal

An LLM may receive:

-   The original issue.
-   The top-ranked code units.
-   Relevant surrounding code.
-   Selected dependency relationships.
-   Relevant tests and error evidence.
-   Explicit instructions to distinguish observations from hypotheses.

It can then produce:

-   A concise diagnosis hypothesis.
-   The evidence supporting that hypothesis.
-   Alternative explanations or missing evidence.
-   A proposed patch.
-   Suggested regression tests.

The LLM should not be allowed to silently fabricate file paths, line
references, test outcomes, or repository facts. References should be
validated against the indexed revision.

### 6.9 Patch Validation

Patch validation is optional and comes after localization.

The validation process should:

1.  Create an isolated worktree or disposable checkout at the recorded
    commit.
2.  Apply the proposed patch without changing the original repository.
3.  Run safe, configured checks and tests with resource and time limits.
4.  Capture stdout, stderr, exit codes, and test summaries.
5.  Compare the changed files with the proposed scope.
6.  Report which checks passed, failed, or could not run.

A patch passing existing tests is evidence, not a guarantee of
correctness. Tests may be incomplete or may not cover the reported
defect.

### 6.10 API and User Interface

The backend can expose endpoints such as:

-   `POST /repositories/index` --- index a supported repository
    revision.
-   `POST /issues/analyze` --- analyze an issue and return ranked
    candidates.
-   `GET /analyses/{analysis_id}` --- retrieve an analysis result.
-   `POST /patches/propose` --- request a patch proposal.
-   `POST /patches/validate` --- run approved validation checks in
    isolation.

These are proposed API contracts and can be adjusted during
implementation.

The UI should let users submit an issue, inspect ranked functions, open
source context, understand ranking evidence, compare alternative
candidates, and review a patch and its test results.

------------------------------------------------------------------------

## 7. NLP Topics Used in RepoMind

RepoMind is interdisciplinary, but its core NLP contribution is
issue-to-code retrieval and ranking.

### 7.1 Text Preprocessing and Information Extraction

Used to interpret bug reports and extract useful signals.

Topics:

-   Tokenization and sentence segmentation.
-   Text normalization.
-   Parsing mixed natural-language, code, and log content.
-   Named entity recognition and technical entity extraction.
-   Information extraction from stack traces and reproduction steps.
-   Query construction and expansion.

A key implementation detail is to preserve technical tokens. Generic
text preprocessing designed for ordinary prose may damage code
identifiers, exception names, and paths.

### 7.2 Information Retrieval

Used to find candidate code from the issue.

Topics:

-   Inverted indexes.
-   Term weighting and TF-IDF.
-   BM25.
-   Query formulation.
-   Top-k retrieval.
-   Precision and recall.
-   Reciprocal rank fusion for combining ranked lists.

BM25 is an important non-neural baseline. The project should show what
semantic methods improve over it and where lexical search remains
stronger.

### 7.3 Embeddings and Semantic Similarity

Used to retrieve code that is conceptually relevant even when it shares
few exact words with the issue.

Topics:

-   Vector representations of text and code.
-   Sentence and code embeddings.
-   Cosine similarity or another appropriate similarity measure.
-   Dense retrieval.
-   Vector indexes.
-   Contrastive learning, if fine-tuning is pursued.

Candidate model families include general-purpose sentence embedding
models and code-aware models. Their suitability must be established
experimentally rather than assumed.

### 7.4 Transformers and Code Language Models

Used to encode issue--code relationships and, optionally, generate
explanations or patches.

Topics:

-   Transformer encoders and self-attention.
-   Pretrained representations.
-   Code and natural-language alignment.
-   Fine-tuning.
-   Sequence-pair classification.
-   Cross-encoder relevance scoring.

Models worth investigating include CodeBERT, GraphCodeBERT, and
UniXcoder. They are candidates for experiments, not mandatory
dependencies.

### 7.5 Learning to Rank

Used to combine signals and improve the order of candidate functions.

Topics:

-   Pointwise, pairwise, and listwise ranking.
-   Supervised relevance labels.
-   Negative sampling and hard negatives.
-   Ranking loss functions.
-   Feature-based score fusion.
-   Ranking evaluation.

Start with transparent score fusion. A learned ranking model should be
introduced only when sufficient training labels and a reliable
evaluation split are available.

### 7.6 Natural Language Generation and Grounded Reasoning

Used to summarize evidence and propose a possible fix.

Topics:

-   Context selection.
-   Evidence-grounded generation.
-   Summarization.
-   Code generation.
-   Instruction following.
-   Uncertainty and abstention.
-   Citation and source-reference validation.

This is a supporting component. An articulate explanation is not a
substitute for correct localization.

------------------------------------------------------------------------

## 8. Non-NLP Technologies

Other disciplines make the system practical:

-   **Tree-sitter and compiler concepts:** Parse source code and
    identify code units.
-   **Static analysis:** Discover references, imports, and potential
    call relationships.
-   **Graph algorithms:** Expand or rank candidates using dependency
    relationships.
-   **Vector databases or vector indexes:** Store and search embeddings.
-   **Backend engineering:** Provide indexing and analysis APIs.
-   **Testing and program repair:** Validate proposed changes.
-   **Frontend engineering:** Present ranked results and source
    evidence.
-   **Sandboxing and security:** Run untrusted repositories and tests
    safely.

This distinction is useful when describing the project: the core NLP
contribution is semantic retrieval, relevance ranking, and
evidence-grounded issue-to-code matching; repository parsing and graph
analysis provide additional context.

------------------------------------------------------------------------

## 9. Proposed Technology Stack

The stack should remain modular so individual methods can be compared.

  --------------------------------------------------------------------------
  Layer                   Proposed technologies      Purpose
  ----------------------- -------------------------- -----------------------
  Primary language        Python                     NLP, retrieval,
                                                     orchestration,
                                                     evaluation

  Source parsing          Tree-sitter                Parse functions,
                                                     classes, and syntax
                                                     structure

  NLP models              Hugging Face Transformers, Embeddings and
                          Sentence Transformers      reranking

  Lexical retrieval       BM25 library or            Exact-term and lexical
                          OpenSearch/Elasticsearch   matching

  Vector search           FAISS for an MVP; a vector Dense similarity search
                          database if needed later   

  Graph analysis          NetworkX initially;        Represent dependencies
                          specialized graph tooling  and code relationships
                          if justified               

  Backend                 FastAPI, Pydantic          API and request
                                                     validation

  Data storage            SQLite or PostgreSQL       Repository metadata and
                                                     analysis records

  Frontend                React with Vite            User-facing analysis
                                                     interface

  Model/API provider      Local model or authorized  Optional diagnosis and
                          hosted API                 patch proposal

  Testing                 pytest and                 Unit tests and patch
                          repository-specific test   validation
                          commands                   

  Deployment              Docker, with isolated      Repeatable deployment
                          workers for untrusted code and execution
                                                     boundaries
  --------------------------------------------------------------------------

These are recommendations, not fixed requirements. For the first
version, avoid operating a separate service for every component. A
modular Python backend is sufficient.

------------------------------------------------------------------------

## 10. Data Sources and Dataset Strategy

A reliable evaluation dataset is essential.

### 10.1 Candidate Benchmarks

**SWE-bench:** A benchmark based on real GitHub issues and software
repositories, designed around repository-level issue resolution. It can
support experiments on retrieval and localization as well as end-to-end
patch evaluation, depending on the chosen subset and available
annotations.

Website: https://www.swebench.com/

**BugsInPy:** A benchmark of real bugs in Python projects. It is a
candidate for a manageable initial study of Python bug localization and
repair.

Repository: https://github.com/soarsmu/BugsInPy

Before using either dataset, verify its current format, license,
repository setup requirements, labels, and whether it provides the exact
function-level ground truth needed for a proposed metric.

### 10.2 Building the Initial Evaluation Set

Start with a manageable subset rather than indexing every project
immediately.

For each issue, aim to record:

-   Issue identifier and description.
-   Repository URL and exact commit.
-   Buggy or relevant file paths where available.
-   Relevant functions or source-line ranges where reliable labels
    exist.
-   Associated tests, if available.
-   Whether the issue is suitable for localization evaluation.
-   Source and annotation provenance.

Some issues have multiple valid locations or require changes across
several files. The evaluation format should support multiple relevant
code units where possible.

### 10.3 Data Leakage Controls

-   Use repository- or project-aware splits when feasible.
-   Deduplicate issues and near-identical patches.
-   Avoid using the fix patch or post-fix information as model input
    during evaluation.
-   Index the intended pre-fix revision.
-   Ensure that training labels, hard negatives, and evaluation labels
    do not leak across splits.
-   Record all preprocessing and indexing configurations.

A strong benchmark protocol is more valuable than a large but
contaminated dataset.

------------------------------------------------------------------------

## 11. Research Questions and Experimental Plan

The research goal is to determine which techniques improve bug
localization and at what cost.

### Main research question

Does hybrid lexical and semantic retrieval, combined with cross-encoder
reranking and controlled dependency-aware expansion, improve
repository-level bug localization over lexical search alone?

### Experiments

**Experiment A --- BM25 baseline**

Search issue text against functions, comments, identifiers, file paths,
and related code metadata.

**Experiment B --- Dense retrieval**

Retrieve code using semantic embeddings.

**Experiment C --- Hybrid retrieval**

Combine lexical and dense rankings using a method such as reciprocal
rank fusion or a tuned score combination.

**Experiment D --- Reranking**

Rerank the hybrid top-k results using a cross-encoder.

**Experiment E --- Dependency-aware expansion**

Expand candidates with relevant callers, callees, imports, or tests, and
measure the effect.

**Experiment F --- Optional learned ranking**

Train a ranking model using issue--function relevance labels and
retrieval features, provided the dataset is sufficient.

Each experiment should use the same evaluation cases, repository
revision, ground truth, and metric definitions. Report latency and
resource usage alongside ranking quality.

### Evaluation metrics

-   **Recall@5 and Recall@10:** Whether one or more known relevant
    locations appear in the top five or ten results.
-   **MRR (Mean Reciprocal Rank):** How highly the first relevant result
    is ranked.
-   **nDCG@10:** Ranking quality when several candidates can be relevant
    and relevance grades are available.
-   **File-level localization accuracy:** How often a relevant or known
    buggy file is retrieved within a specified cutoff.
-   **Function-level localization accuracy:** How often a relevant or
    known buggy function is retrieved within a specified cutoff.
-   **Latency:** Time taken by indexing, retrieval, reranking, graph
    expansion, and generation.
-   **Patch success rate:** For the optional repair stage, the
    proportion of evaluated proposed patches that pass the defined
    validation suite.

State the relevance definition for every metric. Do not report a single
accuracy number without explaining its denominator, cutoff, and
ground-truth assumptions. Use confidence intervals or repeated
measurements where practical.

------------------------------------------------------------------------

## 12. Success Criteria

The project should have concrete acceptance criteria rather than relying
on visual polish.

### MVP success criteria

1.  A user can submit a supported repository and a bug report.
2.  The system can parse and index supported source files.
3.  It returns a ranked list of code functions with verified file paths
    and line ranges.
4.  Each result retains the retrieval method and evidence behind its
    rank.
5.  Results can be evaluated against labeled issues using repeatable
    scripts.
6.  The BM25 baseline and dense retrieval baseline can be compared
    fairly.
7.  Failures such as unsupported languages or inaccessible repositories
    are reported clearly.

### Research success criteria

-   Hybrid retrieval and reranking are evaluated against the same
    baseline.
-   Any claimed improvement is supported by measured results.
-   The contribution of each component is examined through ablation
    experiments.
-   Performance is tested on held-out projects or repositories.
-   Failure cases and trade-offs are documented.
-   Code, configuration, dataset references, and evaluation procedures
    are reproducible.

Do not set a numerical performance improvement target until a baseline
has been measured. The goal is to discover whether the proposed method
improves localization reliably.

------------------------------------------------------------------------

## 13. Suggested Development Roadmap

A reasonable first implementation can be planned over approximately six
to eight weeks, depending on the available time, dataset setup, and
model resources.

### Phase 1 --- Scope and repository parsing (Week 1)

-   Choose one initial language, preferably Python.
-   Select a small set of repositories and issues.
-   Implement repository checkout and revision tracking.
-   Use Tree-sitter to extract functions and methods.
-   Store source paths, symbol names, and line ranges.
-   Write parser and indexing tests.

**Deliverable:** A searchable, metadata-rich code index.

### Phase 2 --- Lexical retrieval baseline (Week 2)

-   Convert each code unit into a searchable document.
-   Index code, comments, identifiers, paths, and symbol names.
-   Implement BM25 search.
-   Return top-k functions for an issue.
-   Create a script to evaluate retrieval against known relevant
    locations.

**Deliverable:** A reproducible baseline with initial metrics.

### Phase 3 --- Dense and hybrid retrieval (Week 3)

-   Choose a pretrained embedding model.
-   Generate and store code-unit embeddings.
-   Embed issue descriptions and search the vector index.
-   Combine lexical and dense rankings.
-   Compare Recall@k, MRR, and latency.

**Deliverable:** A comparison of lexical, dense, and hybrid retrieval.

### Phase 4 --- Reranking and error analysis (Week 4)

-   Add a cross-encoder or code-aware reranker.
-   Rerank a fixed candidate set.
-   Inspect false positives and false negatives.
-   Analyze issue types where each method succeeds or fails.

**Deliverable:** A measured reranking experiment and failure report.

### Phase 5 --- Dependency-aware context (Week 5)

-   Extract imports and feasible call relationships.
-   Build a simple graph of code units.
-   Expand candidates with bounded graph traversal.
-   Compare ranking quality and latency with and without graph context.

**Deliverable:** An ablation study of dependency-aware ranking.

### Phase 6 --- Diagnosis and patch proposal (Week 6)

-   Select and configure an LLM if required.
-   Supply the issue, ranked candidates, and verified source context.
-   Generate an evidence-based diagnosis and optional diff.
-   Validate references and ensure the model distinguishes hypotheses
    from facts.

**Deliverable:** An end-to-end diagnosis prototype.

### Phase 7 --- Test validation and UI (Weeks 7--8)

-   Run approved tests in an isolated disposable environment.
-   Build a simple React interface.
-   Expose the workflow through FastAPI.
-   Add clear status, error handling, and result provenance.
-   Document setup, evaluation, limitations, and safe usage.

**Deliverable:** A demonstrable prototype with reproducible evaluation.

The order can be adjusted, but research evaluation should begin early.
Do not spend most of the project building the interface before
establishing that the retrieval pipeline works.

------------------------------------------------------------------------

## 14. Proposed API Design

The API can begin with a small number of endpoints.

### `POST /repositories/index`

Purpose: Index a repository revision.

Example request:

``` json
{
  "repository_url": "https://github.com/example/project",
  "commit": "optional-commit-hash",
  "languages": ["python"]
}
```

The response should include an indexing identifier, resolved commit,
supported languages, and indexing status.

### `POST /issues/analyze`

Purpose: Retrieve and rank candidate code locations.

Example request:

``` json
{
  "repository_id": "repo_123",
  "issue_title": "Session lost after refresh",
  "issue_body": "Users are redirected to login after refreshing the dashboard.",
  "top_k": 10
}
```

The response should include ranked candidate locations, scores or ranks
from the relevant retrieval stages, source line ranges, supporting
evidence, and warnings.

### `GET /analyses/{analysis_id}`

Purpose: Retrieve the status or results of an analysis.

### `POST /patches/propose`

Purpose: Generate a proposed patch from an issue and selected evidence.

### `POST /patches/validate`

Purpose: Validate an approved proposed patch in a disposable
environment.

These schemas are illustrative and should be formalized with Pydantic
models, input limits, authorization, and error contracts during
implementation.

------------------------------------------------------------------------

## 15. Proposed Repository Structure

``` text
repomind/
├── app/
│   ├── api/
│   │   ├── routes_repositories.py
│   │   ├── routes_issues.py
│   │   └── routes_patches.py
│   ├── ingestion/
│   │   ├── github_client.py
│   │   └── repository_loader.py
│   ├── parsing/
│   │   ├── tree_sitter_parser.py
│   │   └── symbol_extractor.py
│   ├── issue_understanding/
│   │   ├── preprocessing.py
│   │   └── entity_extraction.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   ├── graph/
│   │   ├── dependency_extractor.py
│   │   └── graph_expansion.py
│   ├── diagnosis/
│   │   ├── evidence_builder.py
│   │   └── patch_generator.py
│   ├── validation/
│   │   └── test_runner.py
│   ├── storage/
│   ├── schemas/
│   └── main.py
├── frontend/
├── evaluation/
│   ├── datasets/
│   ├── run_retrieval_eval.py
│   ├── metrics.py
│   └── reports/
├── tests/
├── configs/
├── docs/
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

This is a proposed layout, not a requirement to create every module at
the beginning. Implement only the modules needed for the current phase.

------------------------------------------------------------------------

## 16. Security, Privacy, and Reliability

Repositories and issue text must be treated as untrusted input.

Important safeguards include:

-   Do not execute repository code during indexing.
-   Run tests and generated patches in disposable, isolated workers.
-   Use strict CPU, memory, time, filesystem, and network restrictions.
-   Never expose API tokens or secrets to model prompts, logs, or the
    UI.
-   Scan or exclude likely secret files before sending any source
    context to an external model provider.
-   Require explicit authorization for private repositories.
-   Avoid logging complete source files or sensitive issue contents
    unnecessarily.
-   Validate file paths and prevent path traversal.
-   Apply limits to repository size, file size, indexing duration, and
    API payloads.
-   Treat comments and repository text as data, not as trusted
    instructions to the model.
-   Validate all file references and line ranges against the exact
    indexed commit.
-   Require user review before accepting or applying a generated patch.
-   Never automatically merge or deploy a generated change in the MVP.

If a hosted LLM is used, disclose which code context is sent to it and
follow the provider's applicable data-handling terms.

------------------------------------------------------------------------

## 17. Risks and Limitations

### Incomplete ground truth

A bug can involve multiple functions, or the original fix may be
imperfect. Mitigation: support multiple relevant locations, document
annotation rules, and use several complementary metrics.

### Semantic retrieval misses exact identifiers

Embeddings may overlook rare symbols, error codes, or precise file
names. Mitigation: retain lexical search and evaluate hybrid retrieval.

### Large repositories and long context

Indexing every file and passing all code to an LLM can be expensive.
Mitigation: parse incrementally, retrieve a bounded candidate set, cache
embeddings, and send only relevant context.

### Incomplete dependency graphs

Dynamic dispatch, reflection, generated code, and language-specific
behavior can make call graphs incomplete. Mitigation: label
relationships by provenance and avoid treating inferred links as
certain.

### LLM hallucination

An LLM can produce a plausible but incorrect explanation or patch.
Mitigation: ground the model in verified source context, validate
references, run tests, and expose uncertainty.

### Dataset leakage

Duplicate issues or fix information can inflate evaluation results.
Mitigation: split carefully by project or repository, use pre-fix
revisions, and exclude fix patches from retrieval inputs.

### Limited language support

Parsing and call-graph quality vary across languages. Mitigation: begin
with Python, validate the pipeline, then add another language
deliberately.

### Test suite limitations

A patch can pass existing tests while leaving the reported bug
unresolved. Mitigation: distinguish test success from verified bug
resolution and add regression tests where appropriate.

------------------------------------------------------------------------

## 18. Final Project Deliverables

The completed project should include:

1.  A working repository indexer for the chosen language.
2.  An issue-understanding and query-preparation module.
3.  BM25, dense, and hybrid retrieval implementations.
4.  A reranking module.
5.  An optional dependency-aware candidate expansion module.
6.  A reproducible benchmark and evaluation scripts.
7.  An analysis report with metrics, ablations, and failure cases.
8.  A FastAPI backend and basic interactive UI.
9.  An optional grounded diagnosis and patch proposal workflow.
10. An isolated patch-validation workflow, if implemented.
11. Documentation for installation, configuration, evaluation, and
    limitations.
12. A demo using a public repository and a known issue at a recorded
    commit.

------------------------------------------------------------------------

## 19. What Makes RepoMind Distinctive?

The project should not be presented as merely an LLM that reads code and
suggests fixes.

Its technical contribution is a measurable pipeline for mapping
natural-language bug descriptions to relevant code locations. It
compares traditional information retrieval with neural semantic
retrieval, reranking, and repository-aware context. It also exposes
evidence behind the ranking and evaluates the methods on known issues.

A strong project outcome would be a carefully supported finding about
which retrieval and ranking components improve localization, on which
types of issues, and at what computational cost. The result may show
that one method is better overall, or that different methods work best
for different classes of bugs. Both are valuable findings when properly
measured.

## 20. Recommended Initial Scope

To keep the project achievable:

-   Support Python repositories first.
-   Accept pasted issue descriptions and public GitHub issues.
-   Index functions and methods with line-level metadata.
-   Implement BM25 before adding neural retrieval.
-   Add one embedding model and a dense vector index.
-   Compare lexical, dense, and hybrid retrieval.
-   Add a cross-encoder only after the baseline is reproducible.
-   Introduce dependency-aware expansion as an experiment, not an
    assumption.
-   Keep patch generation optional until localization has been
    evaluated.
-   Use a small, carefully checked benchmark subset initially.

**The first milestone is simple:** given a bug report, RepoMind should
return a ranked list of relevant functions with valid source locations
and a repeatable evaluation of whether those functions are actually
relevant.

Once that works, the project can grow into a broader code-intelligence
and debugging assistant.
