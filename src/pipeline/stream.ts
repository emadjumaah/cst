/**
 * Streaming pipeline: download parquet → tokenize → write .jsonl.
 * Downloads the parquet file once, then processes locally (no rate limits).
 *
 * Usage:
 *   npx tsx src/pipeline/stream.ts [count] [split]
 *   npx tsx src/pipeline/stream.ts 100000          # first 100K from train
 *   npx tsx src/pipeline/stream.ts 1530000 train   # full train split
 *   npx tsx src/pipeline/stream.ts 171000 test     # test split
 */

import {
  createWriteStream,
  mkdirSync,
  existsSync,
  readFileSync,
} from "node:fs";
import { writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { CSTTokenizer } from "../tokenizer/index.ts";
import { parquetRead } from "hyparquet";

const PARQUET_API =
  "https://huggingface.co/api/datasets/agentlans/high-quality-english-sentences/parquet/default";

interface TrainingExample {
  ids: number[];
  tokens: string[];
  text: string;
}

// ── Download parquet file ──────────────────────────

async function downloadParquet(split: string): Promise<string> {
  const localPath = `data/parquet/${split}.parquet`;

  if (existsSync(localPath)) {
    console.log(`  Parquet cached: ${localPath}`);
    return localPath;
  }

  // Get parquet URL
  console.log(`  Fetching parquet URL for split="${split}"...`);
  const listRes = await fetch(`${PARQUET_API}/${split}`);
  if (!listRes.ok) throw new Error(`Failed to list parquet: ${listRes.status}`);
  const urls = (await listRes.json()) as string[];
  if (urls.length === 0) throw new Error("No parquet files found");

  console.log(`  Downloading ${urls[0]} (~150 MB)...`);
  const dataRes = await fetch(urls[0], { redirect: "follow" });
  if (!dataRes.ok) throw new Error(`Download failed: ${dataRes.status}`);

  mkdirSync(dirname(localPath), { recursive: true });
  const buffer = Buffer.from(await dataRes.arrayBuffer());
  await writeFile(localPath, buffer);
  console.log(
    `  Saved to ${localPath} (${(buffer.length / 1e6).toFixed(1)} MB)`,
  );

  return localPath;
}

// ── Read sentences from parquet ────────────────────

async function readParquetSentences(
  filePath: string,
  maxCount: number,
): Promise<string[]> {
  const buffer = readFileSync(filePath);
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  );

  const sentences: string[] = [];

  await parquetRead({
    file: {
      byteLength: arrayBuffer.byteLength,
      slice: (start: number, end: number) => arrayBuffer.slice(start, end),
    },
    columns: ["text"],
    onComplete: (rows: any[][]) => {
      for (const row of rows) {
        const text = (row[0] as string)?.trim();
        if (text && text.length > 10) {
          sentences.push(text);
        }
      }
    },
  });

  return maxCount < sentences.length ? sentences.slice(0, maxCount) : sentences;
}

// ── BPE baseline tokenizer ────────────────────────

function buildBPETokens(
  sentence: string,
  vocab: Map<string, number>,
): { ids: number[]; tokens: string[] } | null {
  const words = sentence
    .toLowerCase()
    .replace(/[.,!?;:'"()\-]/g, " $& ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
  if (words.length < 3) return null;

  const ids: number[] = [];
  for (const w of words) {
    let id = vocab.get(w);
    if (id === undefined) {
      id = vocab.size;
      vocab.set(w, id);
    }
    ids.push(id);
  }
  return { ids, tokens: words };
}

// ── Main ──────────────────────────────────────────

async function main() {
  const targetCount = parseInt(process.argv[2] ?? "100000", 10);
  const split = process.argv[3] ?? "train";

  console.log(`\n═══ CST Stream Pipeline ═══`);
  console.log(
    `Target: ${targetCount.toLocaleString()} sentences, split=${split}\n`,
  );

  // Step 1: Download parquet
  console.log("Step 1: Download parquet...");
  const parquetPath = await downloadParquet(split);

  // Step 2: Read sentences
  console.log("\nStep 2: Reading sentences from parquet...");
  const sentences = await readParquetSentences(parquetPath, targetCount);
  console.log(`  Loaded ${sentences.length.toLocaleString()} sentences`);

  // Step 3: Tokenize and write
  console.log("\nStep 3: Tokenizing...");
  const cstPath = `data/tokenized/cst/${split}-${sentences.length}.jsonl`;
  const bpePath = `data/tokenized/bpe/${split}-${sentences.length}.jsonl`;

  mkdirSync(dirname(cstPath), { recursive: true });
  mkdirSync(dirname(bpePath), { recursive: true });

  const cstStream = createWriteStream(cstPath);
  const bpeStream = createWriteStream(bpePath);

  const tokenizer = new CSTTokenizer();
  const bpeVocab = new Map<string, number>();
  bpeVocab.set("[PAD]", 0);
  bpeVocab.set("[UNK]", 1);
  bpeVocab.set("[BOS]", 2);
  bpeVocab.set("[EOS]", 3);

  let cstExamples = 0;
  let bpeExamples = 0;
  const startTime = Date.now();
  let lastLogTime = startTime;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];

    // CST
    const output = tokenizer.tokenize(sentence);
    if (output.ids.length >= 3) {
      const example: TrainingExample = {
        ids: output.ids,
        tokens: output.tokens.map((t) => t.value),
        text: sentence,
      };
      cstStream.write(JSON.stringify(example) + "\n");
      cstExamples++;
    }

    // BPE
    const bpe = buildBPETokens(sentence, bpeVocab);
    if (bpe) {
      const example: TrainingExample = {
        ids: bpe.ids,
        tokens: bpe.tokens,
        text: sentence,
      };
      bpeStream.write(JSON.stringify(example) + "\n");
      bpeExamples++;
    }

    // Progress every 5 seconds
    const now = Date.now();
    if (now - lastLogTime > 5000) {
      const elapsed = (now - startTime) / 1000;
      const rate = (i + 1) / elapsed;
      const eta = (sentences.length - i - 1) / rate;
      console.log(
        `  ${(i + 1).toLocaleString()} / ${sentences.length.toLocaleString()} ` +
          `(${(((i + 1) / sentences.length) * 100).toFixed(1)}%) ` +
          `${rate.toFixed(0)} sent/s  ` +
          `ETA: ${formatTime(eta)}  ` +
          `cache: ${(tokenizer as any).lemmaCache?.size ?? "?"} lemmas`,
      );
      lastLogTime = now;
    }
  }

  // Flush
  await new Promise<void>((resolve) => cstStream.end(resolve));
  await new Promise<void>((resolve) => bpeStream.end(resolve));

  // Save vocabularies
  tokenizer.saveVocab(cstPath.replace(/\.jsonl$/, "-vocab.json"));
  const { writeFileSync } = await import("node:fs");
  writeFileSync(
    bpePath.replace(/\.jsonl$/, "-vocab.json"),
    JSON.stringify(Object.fromEntries(bpeVocab), null, 2),
  );

  const totalTime = (Date.now() - startTime) / 1000;

  console.log(`\n═══ COMPLETE ═══`);
  console.log(`  Time:        ${formatTime(totalTime)}`);
  console.log(`  Processed:   ${sentences.length.toLocaleString()} sentences`);
  console.log(
    `  CST:         ${cstExamples.toLocaleString()} examples, vocab ${tokenizer.getVocabSize().toLocaleString()}`,
  );
  console.log(
    `  BPE:         ${bpeExamples.toLocaleString()} examples, vocab ${bpeVocab.size.toLocaleString()}`,
  );
  console.log(
    `  Lemma cache: ${(tokenizer as any).lemmaCache?.size ?? "?"} unique words`,
  );
  console.log(
    `  Compression: ${((1 - tokenizer.getVocabSize() / bpeVocab.size) * 100).toFixed(1)}% vocab reduction`,
  );
  console.log(`\nFiles:`);
  console.log(`  ${cstPath}`);
  console.log(`  ${bpePath}`);
}

function formatTime(secs: number): string {
  if (secs < 60) return `${secs.toFixed(0)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
