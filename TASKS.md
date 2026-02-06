# Tasks

Pick one task to implement. All are scoped for ~40 minutes of implementation time.

---

## 1. Conversation Memory

**Goal:** Add conversation history so follow-up questions work naturally.

**The problem:** Each question is currently independent. If you ask "how many vacation days do I get?" and then "what about sick days?", the bot doesn't understand that "what about" refers to the previous topic.

**What to do:**
- Store conversation history (user questions and assistant responses)
- Include history in the LLM context so follow-ups work
- Add a `/clear` command to reset the conversation
- Handle token limits — don't send infinite history to the LLM

**Design decisions to make:**
- Where to store history (array, class, global)
- What to include (just Q&A pairs, or also retrieved context)
- How much history to keep (last N turns, token counting, simple cap)

**Key files:** `src/rag-cli.ts` (answerQuestion, interactiveCLI)

---

## 2. Source Citations

**Goal:** When the bot answers a question, show which document(s) the answer came from.

**What to do:**
- Modify `answerQuestion()` to track which source files contributed to the answer
- Display source filenames alongside the answer (e.g., `Sources: company_handbook.md, benefits_overview.md`)
- Deduplicate sources — don't show the same file multiple times
- Only show sources that were actually used, not all files in the knowledge base

**Key files:** `src/rag-cli.ts` (answerQuestion, buildContext)

---

## 3. Hybrid Search

**Goal:** Vector search misses exact keyword matches. Add keyword/text search and combine results with vector search.

**What to do:**
- Implement a keyword/text search over the loaded documents
- Merge keyword results with vector search results sensibly (union, weighted combination, etc.)
- Deduplicate when both methods find the same chunk
- Consider making the weighting configurable or toggleable

**Test cases to try:**
- `"What is the Q3 2024 revenue?"` — keyword search helps with exact terms
- `"How do I handle expenses?"` — semantic search helps with conceptual matches
- `"OWASP security"` — both methods contribute

**Key files:** `src/embeddings-service.ts`

---

## 4. Query-Aware Retrieval

**Goal:** Search queries don't use conversation history. A follow-up like "what about contractors?" only searches for "contractors" without the previous topic's context.

**What to do:**
- Pass recent conversation history into the query generation step
- Rewrite follow-up queries to include context from the previous topic
- `"what about contractors?"` should become something like `"contractor policy for PTO and time off"`
- Show rewritten queries in `--debug` mode
- Be judicious — don't always dump full history into query generation

**Key files:** `src/rag-cli.ts` (proposeSearchQueries)

---

## 5. Semantic Chunking

**Goal:** The current fixed-size chunking (400 chars) breaks mid-sentence and splits related content. Make chunking respect document structure.

**What to do:**
- Chunk at sentence or paragraph boundaries instead of fixed character counts
- Start new chunks at markdown headings
- Keep code blocks and lists together in one chunk
- Keep chunk size flexible but bounded (roughly 200–600 chars)
- Handle edge cases: very small sections, very large paragraphs

**Things to consider:**
- How to handle sections that exceed the max size
- Whether headings should be included in the chunk text or stored as metadata
- Whether chunks should overlap slightly for better context continuity

**Key files:** `src/embeddings-service.ts` (chunkFile)
