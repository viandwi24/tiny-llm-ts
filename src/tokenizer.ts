import fs from 'fs'
import path from 'path'
import { UNK_TOKEN } from './constants'

export interface Tokenizer {
  vocab: Record<string, number>
  merges: [string, string][]
  charMarker: string
  unkId: number
  encode(text: string): number[]
  decode(ids: number[]): string
  tokenToId(token: string): number
  idToToken(id: number): string
}

export function loadTokenizer(vocabDir: string, charMarker: string): Tokenizer {
  const vocabPath = path.join(vocabDir, 'vocab.json')
  const mergesPath = path.join(vocabDir, 'merges.json')

  if (!fs.existsSync(vocabPath) || !fs.existsSync(mergesPath)) {
    throw new Error(`[tokenizer] vocab.json or merges.json not found in: ${vocabDir}\nRun "pretrain tokenize" first.`)
  }

  const vocab: Record<string, number> = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'))
  const merges: [string, string][] = JSON.parse(fs.readFileSync(mergesPath, 'utf-8'))
  const unkId = vocab[UNK_TOKEN] ?? 0
  const invertedVocab: Record<number, string> = Object.fromEntries(
    Object.entries(vocab).map(([token, id]) => [id, token])
  )

  return {
    vocab,
    merges,
    charMarker,
    unkId,

    encode(text: string): number[] {
      const tokenIds: number[] = []
      const words = text.toLowerCase().split(/\s+/).filter(Boolean)

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
    },

    decode(ids: number[]): string {
      return ids
        .map(id => invertedVocab[id] ?? UNK_TOKEN)
        .join('')
        .replaceAll(charMarker, ' ')
        .trim()
    },

    tokenToId(token: string): number {
      return vocab[token] ?? unkId
    },

    idToToken(id: number): string {
      return invertedVocab[id] ?? UNK_TOKEN
    },
  }
}
