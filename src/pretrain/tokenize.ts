import fs from 'fs'
import path from 'path'
import { resolve } from 'path'
import React from 'react'
import { renderTUI } from '../tui'
import { TokenizeProgress, type TokenizeCallbacks } from '../tui/components/TokenizeProgress'
import { loadConfig, resolveTokenizerConfig, resolvePretrainTokenizeConfig } from '../config'
import { SPECIAL_TOKENS } from '../constants'

interface TokenizeOptions {
  inputDir: string
  outputDir: string
  vocabSize: number
  charMarker: string
}

const yield_ = () => new Promise<void>(resolve => setImmediate(resolve))

async function runTokenizeWithCallbacks(opts: TokenizeOptions, cb: TokenizeCallbacks) {
  const start = Date.now()

  // step 1: read all text files from inputDir
  cb.onPhase('Reading files...')
  await yield_()
  const corpus_texts: string[] = []
  const files = fs.readdirSync(opts.inputDir)
  for (const file of files) {
    if (file.endsWith('.txt')) {
      const content = fs.readFileSync(path.join(opts.inputDir, file), 'utf-8')
      corpus_texts.push(content)
    }
  }

  // step 2: word frequency count
  cb.onPhase(`Building word frequencies (${corpus_texts.length} files)...`)
  await yield_()
  const wordFreq: Record<string, number> = {}
  const words = corpus_texts.join('\n\n').toLowerCase().split(/\s+/)
  for (const word of words) {
    const token = (opts.charMarker + word).split('').join(' ')
    wordFreq[token] = (wordFreq[token] || 0) + 1
  }

  // step 3: initialize first vocab
  cb.onPhase('Initializing base vocabulary...')
  await yield_()
  const vocab: Record<string, number> = {}
  let idCount = 0
  for (const word of Object.keys(wordFreq)) {
    const chars = word.split(' ')
    for (const char of chars) {
      if (!(char in vocab)) {
        vocab[char] = idCount
        idCount++
      }
    }
  }

  // step 4: BPE merge
  cb.onPhase('BPE merge...')
  const merges: [string, string][] = []
  const totalMerges = opts.vocabSize - Object.keys(vocab).length
  let iteration = 0
  const logEvery = Math.max(1, Math.floor(totalMerges / 20))
  const bpeStart = Date.now()

  while (Object.keys(vocab).length < opts.vocabSize) {
    const pairFreq: Record<string, number> = {}

    for (const [word, freq] of Object.entries(wordFreq)) {
      const symbols = word.split(' ')
      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = `${symbols[i]} ${symbols[i + 1]}`
        pairFreq[pair] = (pairFreq[pair] || 0) + freq
      }
    }

    if (Object.keys(pairFreq).length === 0) break

    const bestPair = Object.entries(pairFreq).reduce((a, b) => b[1] > a[1] ? b : a)[0]!
    const [first, second] = bestPair.split(' ') as [string, string]
    const newToken = first + second

    for (const word of Object.keys(wordFreq)) {
      const updated = word.replaceAll(`${first} ${second}`, newToken)
      if (updated !== word) {
        wordFreq[updated] = (wordFreq[updated] || 0) + (wordFreq[word] as number)
        delete wordFreq[word]
      }
    }

    vocab[newToken] = idCount++
    merges.push([first, second])
    iteration++

    if (iteration % logEvery === 0) {
      const elapsed = (Date.now() - bpeStart) / 1000
      const msPerIter = (Date.now() - bpeStart) / iteration
      const eta = (msPerIter * Math.max(0, totalMerges - iteration)) / 1000
      cb.onBpeTick(iteration, totalMerges, elapsed, eta, Object.keys(vocab).length)
      await yield_()
    }
  }

  // step 5: special tokens
  cb.onPhase('Adding special tokens...')
  await yield_()
  for (const token of SPECIAL_TOKENS) {
    vocab[token] = idCount++
  }

  // step 6: save
  cb.onPhase('Saving vocab and merges...')
  await yield_()
  if (!fs.existsSync(opts.outputDir)) {
    fs.mkdirSync(opts.outputDir, { recursive: true })
  }
  fs.writeFileSync(path.join(opts.outputDir, 'vocab.json'), JSON.stringify(vocab), 'utf-8')
  fs.writeFileSync(path.join(opts.outputDir, 'merges.json'), JSON.stringify(merges), 'utf-8')

  const totalElapsed = (Date.now() - start) / 1000
  cb.onDone(Object.keys(vocab).length, merges.length, totalElapsed)
}

export async function runTokenize() {
  const config = loadConfig()
  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)

  const opts: TokenizeOptions = {
    inputDir: resolve(process.cwd(), tokenizeCfg.inputDir),
    outputDir: resolve(process.cwd(), tokenizeCfg.outputDir),
    vocabSize: tokenizerCfg.vocabSize,
    charMarker: tokenizerCfg.charMarker,
  }

  const waitUntilExit = renderTUI(
    React.createElement(TokenizeProgress, {
      vocabSize: opts.vocabSize,
      onRun: (cb) => runTokenizeWithCallbacks(opts, cb),
    })
  )
  await waitUntilExit()
}