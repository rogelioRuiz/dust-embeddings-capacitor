import type { ModelDescriptor, SessionPriority } from 'dust-core-capacitor'

export const PoolingStrategy = {
  Mean: 'mean',
  CLS: 'cls',
  EOS: 'eos',
  LastToken: 'last_token',
} as const
export type PoolingStrategy = (typeof PoolingStrategy)[keyof typeof PoolingStrategy]

export const TokenizerType = {
  WordPiece: 'wordpiece',
  BPE: 'bpe',
} as const
export type TokenizerType = (typeof TokenizerType)[keyof typeof TokenizerType]

export interface EmbeddingModelDescriptor extends ModelDescriptor {
  dims: number
  maxSequenceLength: number
  tokenizerType: TokenizerType
  tokenizerVocabUrl: string
  tokenizerVocabSha256: string
  pooling: PoolingStrategy
  normalize: boolean
  inputNames?: {
    inputIds?: string
    attentionMask?: string
    tokenTypeIds?: string
  }
  outputName?: string
}

export interface EmbeddingResult {
  embedding: number[]
  tokenCount: number
  truncated: boolean
  modelId: string
}

export interface EmbeddingModelMetadata {
  dims: number
  maxSequenceLength: number
  tokenizerType: TokenizerType
  pooling: PoolingStrategy
  normalize: boolean
  backend: 'onnx' | 'gguf'
}

export interface TokenizerOutput {
  inputIds: number[]
  attentionMask: number[]
  tokenTypeIds: number[]
}

export interface LoadEmbeddingModelResult {
  modelId: string
  dims: number
  maxSequenceLength: number
}

export interface BatchEmbeddingResult {
  embeddings: EmbeddingResult[]
}

export interface SimilarityResult {
  score: number
}

export interface EmbeddingsPlugin {
  loadModel(options: {
    descriptor: EmbeddingModelDescriptor
    priority?: SessionPriority
  }): Promise<LoadEmbeddingModelResult>
  unloadModel(options: { modelId: string }): Promise<void>
  listLoadedModels(): Promise<{ modelIds: string[] }>
  getModelMetadata(options: { modelId: string }): Promise<EmbeddingModelMetadata>
  embed(options: { modelId: string; text: string; truncate?: boolean }): Promise<EmbeddingResult>
  batchEmbed(options: { modelId: string; texts: string[]; truncate?: boolean }): Promise<BatchEmbeddingResult>
  embedImage(options: { modelId: string; imageBase64: string }): Promise<EmbeddingResult>
  similarity(options: { a: number[]; b: number[] }): Promise<SimilarityResult>
  countTokens(options: { modelId: string; text: string }): Promise<{ count: number; truncated: boolean }>
  tokenize(options: { modelId: string; text: string; maxLength?: number }): Promise<TokenizerOutput>
}
