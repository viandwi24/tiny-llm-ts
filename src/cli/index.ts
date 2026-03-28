import { Command } from 'commander'
import { rmSync, existsSync } from 'fs'
import { resolve } from 'path'
import { loadConfig, resolveTokenizerConfig, resolvePretrainTokenizeConfig, resolvePretrainEncodeConfig } from '../config'
import { wikipediaProvider } from '../pretrain/providers/wikipedia'
import { runTokenize } from '../pretrain/tokenize'
import { runEncode } from '../pretrain/encode'
import type { PretrainProvider } from '../pretrain/provider'

const PRETRAIN_PROVIDERS: PretrainProvider<any>[] = [
  wikipediaProvider,
  // daftarkan provider baru di sini
]

export const createCLI = () => {
  const program = new Command()
  const config = loadConfig()

  program
    .name('tiny-llm')
    .description('A tiny LLM implementation in TypeScript')
    .version('1.0.0')

  const pretrain = program
    .command('pretrain')
    .description('Pre-train the model with raw text data')

  pretrain
    .command('reset')
    .description('Reset pre-training raw data directory')
    .option('--dir <path>', 'Directory to reset', 'data/pretrain/raw')
    .action((opts) => {
      const dir = resolve(process.cwd(), opts.dir)
      if (!existsSync(dir)) {
        console.log(`[pretrain] Nothing to reset, directory does not exist: ${dir}`)
        return
      }
      rmSync(dir, { recursive: true, force: true })
      console.log(`[pretrain] Reset: ${dir}`)
    })

  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)
  const encodeCfg = resolvePretrainEncodeConfig(config)

  pretrain
    .command('tokenize')
    .description('Tokenize pre-training data using BPE')
    .option('--input <path>', 'Input directory of raw text files', tokenizeCfg.inputDir)
    .option('--output <path>', 'Output directory for tokenized data', tokenizeCfg.outputDir)
    .option('--vocab-size <number>', 'BPE vocabulary size', String(tokenizerCfg.vocabSize))
    .action((opts) => runTokenize({
      inputDir: resolve(process.cwd(), opts.input),
      outputDir: resolve(process.cwd(), opts.output),
      vocabSize: Number(opts.vocabSize),
      charMarker: tokenizerCfg.charMarker,
    }))

  pretrain
    .command('encode')
    .description('Encode pre-training raw text files into token IDs using BPE vocab')
    .option('--input <path>', 'Input directory of raw text files', encodeCfg.inputDir)
    .option('--vocab <path>', 'Vocab directory (vocab.json + merges.json)', encodeCfg.vocabDir)
    .option('--output <file>', 'Output binary file', encodeCfg.outputFile)
    .action((opts) => runEncode({
      inputDir: resolve(process.cwd(), opts.input),
      vocabDir: resolve(process.cwd(), opts.vocab),
      outputFile: resolve(process.cwd(), opts.output),
      charMarker: tokenizerCfg.charMarker,
    }))

  const pretrainFetch = pretrain
    .command('fetch')
    .description('Fetch data from a provider for pre-training')

  // Daftarkan subcommand per provider secara dynamic
  for (const provider of PRETRAIN_PROVIDERS) {
    const providerConfig = {
      ...provider.defaultConfig,
      ...(config.pretrain?.providers as any)?.[provider.name],
    }

    const cmd = pretrainFetch
      .command(provider.name)
      .description(provider.description)

    provider.registerCommand(cmd, providerConfig)

    cmd.action((opts) => provider.fetch(opts, providerConfig))
  }

  return program
}
