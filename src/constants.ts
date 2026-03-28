// Architecture-level constants — changing these requires updating model code.

export const SPECIAL_TOKENS = [
  '<|system|>',
  '<|user|>',
  '<|assistant|>',
  '<|end|>',
  '<|pad|>',
  '<|unk|>',
] as const

export const UNK_TOKEN = '<|unk|>'
