#!/usr/bin/env node
/*
 * Simple RAG CLI (backend-only) implemented in TypeScript.
 *
 * Features:
 *  - Load & chunk markdown docs from data/company-data/*.md
 *  - Cache embeddings in .cache/embeddings.json on first run
 *  - Interactive ask/answer loop (vector search + LLM)
 *  - Deterministic mock mode when OPENAI_API_KEY is missing
 *  - TODO stub for future `train` command
 *  - Contains an intentionally buggy similarity implementation
 */

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import crypto from 'crypto';

const DOCS_DIR = path.resolve('data', 'company-data');
const CACHE_PATH = path.resolve('.cache', 'embeddings.json');
const CHUNK_SIZE = 400; // characters
const EMBEDDING_DIM = 1536;

type Chunk = { source: string; content: string };

interface CacheFile {
  chunks: Chunk[];
  embeddings: number[][];
}

// ---------------- Embeddings ------------------

function openaiAvailable(): boolean {
  return Boolean(process.env.OPENAI_API_KEY);
}

async function embedTexts(texts: string[]): Promise<number[][]> {
  if (openaiAvailable()) {
    // Dynamically import to avoid failure in mock mode
    const { default: OpenAI } = await import('openai');
    const openai = new OpenAI();
    const resp = await openai.embeddings.create({
      input: texts,
      model: 'text-embedding-ada-002'
    });
    // @ts-ignore
    return resp.data.map((d: any) => d.embedding as number[]);
  }
  // Mock deterministic sha256 → floats
  return texts.map((t) => shaEmbedding(t));
}

function shaEmbedding(text: string): number[] {
  const hash = crypto.createHash('sha256').update(text).digest();
  const repeats = Math.ceil((EMBEDDING_DIM * 4) / hash.length);
  const bytes = Buffer.alloc(repeats * hash.length);
  for (let i = 0; i < repeats; i++) {
    hash.copy(bytes, i * hash.length);
  }
  const embedding: number[] = [];
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    embedding.push(bytes[i * 4] / 255);
  }
  return embedding;
}

// --------------- Chunking ---------------------

function loadMarkdownChunks(): Chunk[] {
  const chunks: Chunk[] = [];
  if (!fs.existsSync(DOCS_DIR)) return chunks;
  const files = walkFiles(DOCS_DIR, '.md');
  for (const file of files) {
    const text = fs.readFileSync(file, 'utf8');
    for (let i = 0; i < text.length; i += CHUNK_SIZE) {
      chunks.push({ source: path.relative('.', file), content: text.slice(i, i + CHUNK_SIZE) });
    }
  }
  return chunks;
}

function walkFiles(dir: string, ext: string): string[] {
  let results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const res = path.join(dir, entry.name);
    if (entry.isDirectory()) results = results.concat(walkFiles(res, ext));
    else if (entry.isFile() && res.endsWith(ext)) results.push(res);
  }
  return results;
}

// --------------- Cache / Ensure --------------

async function ensureEmbeddings(): Promise<CacheFile> {
  const dir = path.dirname(CACHE_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  if (fs.existsSync(CACHE_PATH)) {
    const data = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf8')) as CacheFile;
    return data;
  }

  const chunks = loadMarkdownChunks();
  const embeddings = await embedTexts(chunks.map((c) => c.content));
  const data: CacheFile = { chunks, embeddings };
  fs.writeFileSync(CACHE_PATH, JSON.stringify(data));
  return data;
}

// -------------- Retrieval (buggy) ------------

function buggySimilarity(a: number[], b: number[]): number {
  // BUG: misses sqrt in Euclidean distance and 0-division edge cases
  let dist = 0;
  for (let i = 0; i < a.length; i++) dist += (a[i] - b[i]) ** 2;
  if (dist === 0) return 1;
  return 1 / dist;
}

function retrieve(queryEmbed: number[], embeddings: number[][], k = 4): number[] {
  const scores = embeddings.map((emb, idx) => ({ idx, score: buggySimilarity(queryEmbed, emb) }));
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k).map((s) => s.idx);
}

// -------------- LLM Wrapper ------------------

type Message = { role: 'system' | 'user' | 'assistant'; content: string };

async function chatCompletion(messages: Message[], temperature = 0): Promise<string> {
  if (openaiAvailable()) {
    const { default: OpenAI } = await import('openai');
    const openai = new OpenAI();
    const resp = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages,
      temperature
    });
    // @ts-ignore
    return resp.choices[0].message.content as string;
  }
  // mock deterministic concat lengths
  const joined = messages.map((m) => `${m.role}-${m.content.length}`).join('|');
  return `MOCK_RESPONSE[${joined}]`;
}

// -------------- Prompt helpers ---------------

const SYSTEM_QUERY_GEN = `You are an assistant that suggests concise search queries (2–4) users might run in a vector search over company documentation. Return ONLY a JSON list of strings.`;

const SYSTEM_ANSWER = `Answer the question strictly based on the provided context. If the answer is not contained, say you don't know.`;

async function proposeSearchQueries(question: string): Promise<string[]> {
  const raw = await chatCompletion([
    { role: 'system', content: SYSTEM_QUERY_GEN },
    { role: 'user', content: question }
  ]);
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed.slice(0, 4).map(String);
  } catch {
    /* ignore */
  }
  return question.split(' ').slice(0, 4);
}

function buildContext(indices: number[], chunks: Chunk[]): string {
  return indices
    .map((i) => `### Source: ${chunks[i].source}\n${chunks[i].content}`)
    .join('\n');
}

async function answerQuestion(question: string, state: CacheFile): Promise<string> {
  const queries = await proposeSearchQueries(question);
  const queryEmbeds = await embedTexts(queries);

  const idxSet = new Set<number>();
  for (const qe of queryEmbeds) {
    retrieve(qe, state.embeddings, 3).forEach((i) => idxSet.add(i));
  }

  const context = buildContext(Array.from(idxSet), state.chunks);

  const response = await chatCompletion(
    [
      { role: 'system', content: SYSTEM_ANSWER },
      { role: 'system', content: `Context:\n${context}` },
      { role: 'user', content: question }
    ],
    0.2
  );
  return response;
}

// --------------- CLI -------------------------

async function interactiveCLI() {
  console.log('Simple RAG CLI. Type "exit" to quit.');
  const state = await ensureEmbeddings();

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = (): void => {
    rl.question('\n> ', async (line) => {
      const q = line.trim();
      if (!q || q.toLowerCase() === 'exit' || q.toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      const ans = await answerQuestion(q, state);
      console.log('\n' + wrap(ans, 80));
      ask();
    });
  };

  ask();
}

function wrap(text: string, width: number): string {
  const lines: string[] = [];
  for (const word of text.split(/\s+/)) {
    const last = lines[lines.length - 1];
    if (!last || last.length + word.length + 1 > width) lines.push(word);
    else lines[lines.length - 1] = last + ' ' + word;
  }
  return lines.join('\n');
}

// --------------- Main ------------------------

(async () => {
  const cmd = process.argv[2];
  if (cmd === 'train') {
    console.log('TODO: implement train command to ingest new data.');
    process.exit(0);
  }
  await interactiveCLI();
})();

