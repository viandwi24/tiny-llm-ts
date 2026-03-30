import { Command } from 'commander'
import { rmSync, existsSync } from 'fs'
import { resolve } from 'path'
import { loadConfig } from '../config'
import { wikipediaProvider } from '../pretrain/providers/wikipedia'
import { runTokenize } from '../pretrain/tokenize'
import { runEncode } from '../pretrain/encode'
import { runTrain } from '../train'
import { runChat } from '../chat'
import type { PretrainProvider } from '../pretrain/provider'

const PRETRAIN_PROVIDERS: PretrainProvider<any>[] = [
  wikipediaProvider,
  // daftarkan provider baru di sini
]

export const createCLI = () => {
  const program = new Command()

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

  pretrain
    .command('tokenize')
    .description('Tokenize pre-training data using BPE (config from data/config.json)')
    .action(() => runTokenize())

  pretrain
    .command('encode')
    .description('Encode raw text files into binary token IDs (config from data/config.json)')
    .action(() => runEncode())

  const pretrainFetch = pretrain
    .command('fetch')
    .description('Fetch data from a provider for pre-training')

  // Daftarkan subcommand per provider secara dynamic
  const config = loadConfig()
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

  program
    .command('train')
    .description('Train the Transformer model (config from data/config.json)')
    .option('--reset', 'Hapus checkpoint lama dan mulai training dari awal')
    .action((opts) => runTrain({ reset: !!opts.reset }))

  program
    .command('chat')
    .description('Chat with the trained model (loads latest checkpoint)')
    .option('--temperature <number>', 'Sampling temperature (0 = greedy)', '0')
    .option('--max-tokens <number>', 'Max new tokens per reply', '50')
    .option('--repetition-penalty <number>', 'Penalize repeated tokens (1.0 = off, 1.3 = default)', '1.3')
    .action((opts) => runChat({
      temperature: parseFloat(opts.temperature),
      maxTokens: parseInt(opts.maxTokens, 10),
      repetitionPenalty: parseFloat(opts.repetitionPenalty),
    }))

  return program
}
