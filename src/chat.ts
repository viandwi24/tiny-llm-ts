import readline from 'readline'
import { resolve } from 'path'
import { existsSync } from 'fs'
import { loadConfig, resolveTokenizerConfig, resolvePretrainTokenizeConfig, resolveTrainConfig } from './config'
import { Transformer } from './transformer'
import { loadTokenizer } from './tokenizer'
import { loadCheckpoint, latestCheckpoint } from './train/checkpoint'
import { softmax } from './neural-network/activation'

function applyRepetitionPenalty(logits: number[], generatedIds: number[], penalty: number): number[] {
  if (penalty === 1.0 || generatedIds.length === 0) return logits
  const penalized = [...logits]
  for (const id of generatedIds) {
    const v = penalized[id]!
    penalized[id] = v > 0 ? v / penalty : v * penalty
  }
  return penalized
}

function generate(
  model: Transformer,
  tokenizer: { tokenToId: (t: string) => number; idToToken: (id: number) => string },
  tokenIds: number[],
  maxNewTokens: number,
  maxSeqLen: number,
  temperature: number,
  repetitionPenalty: number,
): number[] {
  const generated: number[] = []
  let context = [...tokenIds]
  const endId = tokenizer.tokenToId('<|end|>')
  const unkId = tokenizer.tokenToId('<|unk|>')

  for (let i = 0; i < maxNewTokens; i++) {
    // trim context to maxSeqLen
    const input = context.length > maxSeqLen ? context.slice(context.length - maxSeqLen) : context

    const logits = model.forward(input)
    const rawLogits = logits[logits.length - 1]!

    // terapkan repetition penalty pada token yang sudah di-generate
    const penalizedLogits = applyRepetitionPenalty(rawLogits, generated, repetitionPenalty)

    let nextId: number
    if (temperature <= 0) {
      // greedy
      let maxVal = -Infinity
      nextId = 0
      for (let k = 0; k < penalizedLogits.length; k++) {
        if (penalizedLogits[k]! > maxVal) { maxVal = penalizedLogits[k]!; nextId = k }
      }
    } else {
      // temperature sampling
      const scaled = penalizedLogits.map(v => v / temperature)
      const probs = softmax(scaled)
      const rand = Math.random()
      let cumulative = 0
      nextId = probs.length - 1
      for (let k = 0; k < probs.length; k++) {
        cumulative += probs[k]!
        if (rand < cumulative) { nextId = k; break }
      }
    }

    const decoded = tokenizer.idToToken(nextId)
    const decodedClean = decoded.replace(/^Ġ/, '')
    if (nextId === endId || decodedClean.startsWith('<|')) break
    if (nextId === unkId) break
    generated.push(nextId)
    context.push(nextId)
  }

  return generated
}

export async function runChat(opts: { temperature?: number; maxTokens?: number; repetitionPenalty?: number } = {}) {
  const config = loadConfig()
  const trainCfg = resolveTrainConfig(config)
  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)

  const vocabDir = resolve(process.cwd(), tokenizeCfg.outputDir)
  const checkpointDir = resolve(process.cwd(), 'data/checkpoints')
  const temperature = opts.temperature ?? 0.8
  const maxNewTokens = opts.maxTokens ?? 50
  const repetitionPenalty = opts.repetitionPenalty ?? 1.3

  // load tokenizer
  const tokenizer = loadTokenizer(vocabDir, tokenizerCfg.charMarker)

  // build model
  const model = new Transformer({
    vocabSize: trainCfg.vocabSize,
    embedSize: trainCfg.embedSize,
    numHeads: trainCfg.numHeads,
    numLayers: trainCfg.numLayers,
    ffnDim: trainCfg.ffnDim,
    maxSeqLen: trainCfg.maxSeqLen,
  })

  // load checkpoint
  const ckpt = latestCheckpoint(checkpointDir)
  if (!ckpt) {
    console.error('[chat] No checkpoint found in data/checkpoints/')
    console.error('[chat] Run "train" first.')
    process.exit(1)
  }
  const meta = loadCheckpoint(model, ckpt)
  console.log(`[chat] Model loaded (step=${meta.step}, loss=${meta.loss.toFixed(4)})`)
  console.log(`[chat] temperature=${temperature}, maxNewTokens=${maxNewTokens}, repetitionPenalty=${repetitionPenalty}`)
  console.log(`[chat] Type your prompt and press Enter. Ctrl+C to exit.\n`)

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout })

  const ask = () => {
    rl.question('You: ', (input) => {
      const prompt = input.trim()
      if (!prompt) { ask(); return }

      const formatted = `<|user|> ${prompt} <|end|>\n<|assistant|>`
      const inputIds = tokenizer.encode(formatted)
      if (inputIds.length === 0) {
        console.log('[chat] Could not encode input.')
        ask()
        return
      }

      const newIds = generate(model, tokenizer, inputIds, maxNewTokens, trainCfg.maxSeqLen, temperature, repetitionPenalty)
      const reply = tokenizer.decode(newIds)
      console.log(`Assistant: ${reply}\n`)

      ask()
    })
  }

  rl.on('close', () => process.exit(0))
  ask()
}
