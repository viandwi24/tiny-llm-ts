import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs'
import { resolve, dirname } from 'path'
import type { WikipediaProviderConfig } from './pretrain/providers/wikipedia'

// ---- Global ----

export interface TokenizerConfig {
  charMarker: string
  vocabSize: number
}

// ---- Pretrain ----

export interface PretrainProvidersConfig {
  wikipedia?: Partial<WikipediaProviderConfig>
  // provider baru cukup tambah optional key di sini
}

export interface PretrainTokenizeConfig {
  inputDir: string
  outputDir: string
}

export interface PretrainEncodeConfig {
  inputDir: string
  vocabDir: string
  outputFile: string
}

export interface PretrainConfig {
  providers?: PretrainProvidersConfig
  tokenize?: Partial<PretrainTokenizeConfig>
  encode?: Partial<PretrainEncodeConfig>
}

export interface TrainConfig {
  dataFile: string
  vocabSize: number
  embedSize: number
  numHeads: number
  numLayers: number
  ffnDim: number
  maxSeqLen: number
  epochs: number
  learningRate: number
  batchSize: number
}

export interface TinyLLMConfig {
  tokenizer?: Partial<TokenizerConfig>
  pretrain?: PretrainConfig
  train?: Partial<TrainConfig>
}

// ---- Defaults ----

export const DEFAULT_TOKENIZER: TokenizerConfig = {
  charMarker: 'Ġ',
  vocabSize: 8000,
}

export const DEFAULT_PRETRAIN_TOKENIZE: PretrainTokenizeConfig = {
  inputDir: 'data/pretrain/raw',
  outputDir: 'data/pretrain/tokenized',
}

export const DEFAULT_PRETRAIN_ENCODE: PretrainEncodeConfig = {
  inputDir: 'data/pretrain/raw',
  vocabDir: 'data/pretrain/tokenized',
  outputFile: 'data/pretrain/train.bin',
}

export const DEFAULT_TRAIN: TrainConfig = {
  dataFile: 'data/pretrain/train.bin',
  vocabSize: 8000,
  embedSize: 128,
  numHeads: 4,
  numLayers: 4,
  ffnDim: 512,
  maxSeqLen: 128,
  epochs: 10,
  learningRate: 3e-4,
  batchSize: 16,
}

// ---- Defaults (full) ----

export const DEFAULT_CONFIG: TinyLLMConfig = {
  tokenizer: DEFAULT_TOKENIZER,
  pretrain: {
    tokenize: DEFAULT_PRETRAIN_TOKENIZE,
    encode: DEFAULT_PRETRAIN_ENCODE,
    providers: {
      wikipedia: {
        topics: ['Indonesia', 'Sejarah Indonesia', 'Geografi', 'Ilmu pengetahuan', 'Matematika', 'Fisika', 'Biologi', 'Kimia', 'Teknologi', 'Seni'],
        language: 'id',
        outputDir: 'data/pretrain/raw',
        searchLimit: 5,
      },
    },
  },
  train: DEFAULT_TRAIN,
}

// ---- Loader ----

const CONFIG_PATH = resolve(process.cwd(), 'data/config.json')

function ensureConfig(): void {
  if (!existsSync(CONFIG_PATH)) {
    mkdirSync(dirname(CONFIG_PATH), { recursive: true })
    writeFileSync(CONFIG_PATH, JSON.stringify(DEFAULT_CONFIG, null, 2), 'utf-8')
    console.log(`[config] Generated default config at data/config.json`)
  }
}

export function loadConfig(): TinyLLMConfig {
  ensureConfig()
  const raw = readFileSync(CONFIG_PATH, 'utf-8')
  return JSON.parse(raw) as TinyLLMConfig
}

// ---- Helpers ----

export function resolveTokenizerConfig(config: TinyLLMConfig): TokenizerConfig {
  return { ...DEFAULT_TOKENIZER, ...config.tokenizer }
}

export function resolvePretrainTokenizeConfig(config: TinyLLMConfig): PretrainTokenizeConfig {
  return { ...DEFAULT_PRETRAIN_TOKENIZE, ...config.pretrain?.tokenize }
}

export function resolvePretrainEncodeConfig(config: TinyLLMConfig): PretrainEncodeConfig {
  return { ...DEFAULT_PRETRAIN_ENCODE, ...config.pretrain?.encode }
}

export function resolveTrainConfig(config: TinyLLMConfig): TrainConfig {
  return { ...DEFAULT_TRAIN, ...config.train }
}