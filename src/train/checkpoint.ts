import fs from 'fs'
import path from 'path'
import type { Transformer } from '../transformer'

export interface CheckpointMeta {
  step: number
  loss: number
  savedAt: string
}

// ── ekstrak semua bobot model ke plain object ─────────────────────────────────

function extractWeights(model: Transformer) {
  return {
    projection: {
      weights: model.projection.weights,
      bias: model.projection.bias,
    },
    blocks: model.blocks.map(block => ({
      attention: {
        wq: { weights: block.attention.wq.weights, bias: block.attention.wq.bias },
        wk: { weights: block.attention.wk.weights, bias: block.attention.wk.bias },
        wv: { weights: block.attention.wv.weights, bias: block.attention.wv.bias },
        wo: { weights: block.attention.wo.weights, bias: block.attention.wo.bias },
      },
      feedforward: {
        linear1: { weights: block.feedforward.linear1.weights, bias: block.feedforward.linear1.bias },
        linear2: { weights: block.feedforward.linear2.weights, bias: block.feedforward.linear2.bias },
      },
      norm1: { gamma: block.norm1.gamma, beta: block.norm1.beta },
      norm2: { gamma: block.norm2.gamma, beta: block.norm2.beta },
    })),
    embedding: {
      tokenWeights: model.embedding.tokenWeights,
      positionWeights: model.embedding.positionWeights,
    },
  }
}

// ── terapkan bobot dari plain object ke model ─────────────────────────────────

function applyWeights(model: Transformer, data: ReturnType<typeof extractWeights>) {
  model.projection.weights = data.projection.weights
  model.projection.bias = data.projection.bias

  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i]!
    const src = data.blocks[i]!

    block.attention.wq.weights = src.attention.wq.weights
    block.attention.wq.bias    = src.attention.wq.bias
    block.attention.wk.weights = src.attention.wk.weights
    block.attention.wk.bias    = src.attention.wk.bias
    block.attention.wv.weights = src.attention.wv.weights
    block.attention.wv.bias    = src.attention.wv.bias
    block.attention.wo.weights = src.attention.wo.weights
    block.attention.wo.bias    = src.attention.wo.bias

    block.feedforward.linear1.weights = src.feedforward.linear1.weights
    block.feedforward.linear1.bias    = src.feedforward.linear1.bias
    block.feedforward.linear2.weights = src.feedforward.linear2.weights
    block.feedforward.linear2.bias    = src.feedforward.linear2.bias

    block.norm1.gamma = src.norm1.gamma
    block.norm1.beta  = src.norm1.beta
    block.norm2.gamma = src.norm2.gamma
    block.norm2.beta  = src.norm2.beta
  }

  model.embedding.tokenWeights    = data.embedding.tokenWeights
  model.embedding.positionWeights = data.embedding.positionWeights
}

// ── public API ────────────────────────────────────────────────────────────────

export function saveCheckpoint(
  model: Transformer,
  meta: CheckpointMeta,
  checkpointDir: string,
  perStep: boolean = false
): string {
  if (!fs.existsSync(checkpointDir)) {
    fs.mkdirSync(checkpointDir, { recursive: true })
  }

  const data = { meta, weights: extractWeights(model) }
  const json = JSON.stringify(data)

  // simpan per-step dan overwrite latest
  const latestFile = path.join(checkpointDir, 'latest.json')

  if (perStep) {
    const stepFile  = path.join(checkpointDir, `checkpoint-step${meta.step}.json`)
    fs.writeFileSync(stepFile, json, 'utf-8')
  }
  fs.writeFileSync(latestFile, json, 'utf-8')

  return latestFile
}

export function loadCheckpoint(model: Transformer, checkpointPath: string): CheckpointMeta {
  if (!fs.existsSync(checkpointPath)) {
    throw new Error(`[checkpoint] File not found: ${checkpointPath}`)
  }

  const data = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8'))
  applyWeights(model, data.weights)

  console.log(`[checkpoint] Loaded step=${data.meta.step}, loss=${data.meta.loss.toFixed(4)}`)
  return data.meta as CheckpointMeta
}

// kembalikan path latest.json jika ada, null jika belum pernah save
export function latestCheckpoint(checkpointDir: string): string | null {
  const p = path.join(checkpointDir, 'latest.json')
  return fs.existsSync(p) ? p : null
}
