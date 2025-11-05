#!/usr/bin/env node
/*
 * Simple RAG CLI (backend-only) implemented in TypeScript.
 *
 * Features:
 *  - Load & chunk markdown docs from data/company-data/*.md
 *  - Cache embeddings in .cache/embeddings.json on first run
 *  - Interactive ask/answer loop (vector search + LLM)
 *  - Requires OPENAI_API_KEY environment variable
 *  - TODO stub for future `train` command
 */

import path from 'path';
import readline from 'readline';
import OpenAI from 'openai';
import { EmbeddingsService, type Chunk } from './embeddings-service.js';

const DOCS_DIR = path.resolve('data', 'company-data');
const CACHE_PATH = path.resolve('.cache', 'embeddings.json');

let openai: OpenAI;
let embeddingsService: EmbeddingsService;

// -------------- LLM Wrapper ------------------

type Message = { role: 'system' | 'user' | 'assistant'; content: string };

async function chatCompletion(messages: Message[], temperature = 0): Promise<string> {
  const resp = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages,
    temperature
  });
  return resp.choices[0].message.content || '';
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
    if (Array.isArray(parsed)) {
      const queries = parsed.slice(0, 4).map(String).filter(q => q.trim().length > 0);
      if (queries.length > 0) return queries;
    }
  } catch {
    /* ignore */
  }
  const fallback = question.split(' ').filter(q => q.trim().length > 0).slice(0, 4);
  return fallback.length > 0 ? fallback : [question];
}

function buildContext(chunks: Chunk[]): string {
  return chunks
    .map((chunk) => `### Source: ${chunk.source}\n${chunk.content}`)
    .join('\n');
}

async function answerQuestion(question: string): Promise<string> {
  const queries = await proposeSearchQueries(question);
  const chunks = await embeddingsService.multiSearch(queries, 3);

  const context = buildContext(chunks);

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

  console.log('Initializing embeddings cache...');
  await embeddingsService.initialize();

  const stats = embeddingsService.getCacheStats();
  console.log(`Loaded ${stats.chunkCount} chunks (embedding dim: ${stats.embeddingDim})`);

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  let closed = false;

  rl.on('close', () => {
    closed = true;
  });

  const ask = (): void => {
    if (closed) return;
    rl.question('\n> ', async (line) => {
      const q = line.trim();
      if (!q || q.toLowerCase() === 'exit' || q.toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      const ans = await answerQuestion(q);
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

async function promptForAPIKey(): Promise<string> {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question('Enter your OpenAI API key: ', (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

(async () => {
  let apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    apiKey = await promptForAPIKey();
    if (!apiKey) {
      console.error('Error: API key is required to run this CLI.');
      process.exit(1);
    }
  }

  openai = new OpenAI({ apiKey });
  embeddingsService = new EmbeddingsService(openai, DOCS_DIR, CACHE_PATH);

  const cmd = process.argv[2];
  if (cmd === 'train') {
    console.log('Rebuilding embeddings cache...');
    await embeddingsService.rebuildCache();
    console.log('Cache rebuilt successfully.');
    process.exit(0);
  }
  await interactiveCLI();
})();

