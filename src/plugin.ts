import { registerPlugin, WebPlugin } from '@capacitor/core'

import type {
  BatchEmbeddingResult,
  EmbeddingModelMetadata,
  EmbeddingResult,
  EmbeddingsPlugin,
  LoadEmbeddingModelResult,
  SimilarityResult,
  TokenizerOutput,
} from './definitions'

class EmbeddingsWeb extends WebPlugin implements EmbeddingsPlugin {
  async loadModel(_options: { descriptor: unknown; priority?: unknown }): Promise<LoadEmbeddingModelResult> {
    throw this.unimplemented('loadModel is not supported on web')
  }

  async unloadModel(_options: { modelId: string }): Promise<void> {
    throw this.unimplemented('unloadModel is not supported on web')
  }

  async listLoadedModels(): Promise<{ modelIds: string[] }> {
    throw this.unimplemented('listLoadedModels is not supported on web')
  }

  async getModelMetadata(_options: { modelId: string }): Promise<EmbeddingModelMetadata> {
    throw this.unimplemented('getModelMetadata is not supported on web')
  }

  async embed(_options: { modelId: string; text: string; truncate?: boolean }): Promise<EmbeddingResult> {
    throw this.unimplemented('embed is not supported on web')
  }

  async batchEmbed(_options: { modelId: string; texts: string[]; truncate?: boolean }): Promise<BatchEmbeddingResult> {
    throw this.unimplemented('batchEmbed is not supported on web')
  }

  async embedImage(_options: { modelId: string; imageBase64: string }): Promise<EmbeddingResult> {
    throw this.unimplemented('embedImage is not supported on web')
  }

  async similarity(_options: { a: number[]; b: number[] }): Promise<SimilarityResult> {
    throw this.unimplemented('similarity is not supported on web')
  }

  async countTokens(_options: { modelId: string; text: string }): Promise<{ count: number; truncated: boolean }> {
    throw this.unimplemented('countTokens is not supported on web')
  }

  async tokenize(_options: { modelId: string; text: string; maxLength?: number }): Promise<TokenizerOutput> {
    throw this.unimplemented('tokenize is not supported on web')
  }
}

export const Embeddings = registerPlugin<EmbeddingsPlugin>('Embeddings', {
  web: () => Promise.resolve(new EmbeddingsWeb()),
})
