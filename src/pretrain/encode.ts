import fs from 'fs'
import path from 'path'
import React from 'react'
import { renderTUI } from '../tui'
import { EncodeProgress, type EncodeCallbacks } from '../tui/components/EncodeProgress'

export interface EncodeOptions {
  inputDir: string
  vocabDir: string
  outputFile: string
}

function encode(merges: [string, string][], vocab: Record<string, number>, text: string): number[] {
  const tokenIds: number[] = []
  const words = text.toLowerCase().split(/\s+/).filter(Boolean)
  const charMarker = 'Ġ'
  const unkId = vocab['<|unk|>'] ?? 0

  for (const word of words) {
    let symbols = (charMarker + word).split('')

    for (const [left, right] of merges) {
      let i = 0
      while (i < symbols.length - 1) {
        if (symbols[i] === left && symbols[i + 1] === right) {
          symbols = [...symbols.slice(0, i), left + right, ...symbols.slice(i + 2)]
        } else {
          i++
        }
      }
    }

    for (const symbol of symbols) {
      tokenIds.push(vocab[symbol] ?? unkId)
    }
  }
  return tokenIds
}

async function runEncodeWithCallbacks(opts: EncodeOptions, cb: EncodeCallbacks) {
  const start = Date.now()

  const vocabPath = path.join(opts.vocabDir, 'vocab.json')
  const mergesPath = path.join(opts.vocabDir, 'merges.json')
  const vocab: Record<string, number> = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'))
  const merges: [string, string][] = JSON.parse(fs.readFileSync(mergesPath, 'utf-8'))

  const files = fs.readdirSync(opts.inputDir).filter(f => f.endsWith('.txt'))
  const allTokenIds: number[] = []

  for (let i = 0; i < files.length; i++) {
    const file = files[i]!
    const text = fs.readFileSync(path.join(opts.inputDir, file), 'utf-8')
    const ids = encode(merges, vocab, text)
    allTokenIds.push(...ids)
    cb.onFile(file, i + 1, files.length, ids.length)
  }

  const buffer = new Uint16Array(allTokenIds)
  if (!fs.existsSync(path.dirname(opts.outputFile))) {
    fs.mkdirSync(path.dirname(opts.outputFile), { recursive: true })
  }
  fs.writeFileSync(opts.outputFile, Buffer.from(buffer.buffer))

  const elapsed = (Date.now() - start) / 1000
  cb.onDone(allTokenIds.length, buffer.byteLength, elapsed)
}

export async function runEncode(opts: EncodeOptions) {
  if (!fs.existsSync(path.join(opts.vocabDir, 'vocab.json')) || !fs.existsSync(path.join(opts.vocabDir, 'merges.json'))) {
    console.error(`[encode] vocab.json or merges.json not found in ${opts.vocabDir}`)
    process.exit(1)
  }

  const waitUntilExit = renderTUI(
    React.createElement(EncodeProgress, {
      onRun: (cb) => runEncodeWithCallbacks(opts, cb),
    })
  )
  await waitUntilExit()
}
