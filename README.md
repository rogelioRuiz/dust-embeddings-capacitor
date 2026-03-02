<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rogelioRuiz/dust/main/assets/branding/dust_white.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/rogelioRuiz/dust/main/assets/branding/dust_black.png">
    <img alt="dust" src="https://raw.githubusercontent.com/rogelioRuiz/dust/main/assets/branding/dust_black.png" width="200">
  </picture>
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="npm" src="https://img.shields.io/badge/npm-dust--embeddings--capacitor-cb3837">
  <img alt="Capacitor" src="https://img.shields.io/badge/Capacitor-7%20%7C%208-119EFF">
  <a href="https://github.com/rogelioRuiz/dust-embeddings-capacitor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rogelioRuiz/dust-embeddings-capacitor/actions/workflows/ci.yml/badge.svg?branch=main"></a>
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<strong>capacitor-embeddings</strong>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# capacitor-embeddings

Capacitor plugin for on-device tokenization and embedding inference — BPE and WordPiece tokenizers, ONNX-backed embedding sessions, pooling strategies, and vector math utilities.

**Current version: 0.1.0**

## Features

- **BPETokenizer** — byte-pair encoding tokenizer with vocabulary file loading
- **WordPieceTokenizer** — subword splitting for BERT-family models
- **EmbeddingSession** — load an ONNX embedding model and run inference
- **EmbeddingSessionManager** — multi-session lifecycle with reference counting
- **PoolingStrategy** — CLS token, mean pooling, max pooling
- **VectorMath** — cosine similarity, dot product, L2 normalization
- **DustCore integration** — registers as an `EmbeddingService` in the DustCore service locator

## Platform support

| | Android | iOS | Web |
|---|---|---|---|
| **Runtime** | dust-embeddings-kotlin + dust-onnx-kotlin | dust-embeddings-swift + dust-onnx-swift | Stub (throws) |
| **Min version** | API 26 | iOS 16.0 | — |

## Install

```bash
npm install dust-embeddings-capacitor dust-core-capacitor
npx cap sync
```

`dust-core-capacitor` is a required peer dependency — it provides the shared ML contract types and the `DustCoreRegistry` that `capacitor-embeddings` uses to register itself as an `EmbeddingService`.

### iOS (CocoaPods)

```ruby
# Podfile
platform :ios, '16.0'
use_frameworks! :linkage => :static

pod 'DustCapacitorEmbeddings', :path => '../capacitor-embeddings'
pod 'DustCapacitorCore',       :path => '../capacitor-core'
```

### Android

The Kotlin plugin resolves `dust-embeddings-kotlin` and `dust-onnx-kotlin` as local project dependencies via the Capacitor Gradle build.

## Quick Start

```typescript
import { EmbeddingPlugin } from 'dust-embeddings-capacitor';

// Tokenize text
const { tokens } = await EmbeddingPlugin.tokenize({
  text: 'Hello, world!',
  tokenizer: 'bpe',
  vocabPath: '/path/to/vocab.json',
});

// Load an embedding model and generate a vector
await EmbeddingPlugin.loadModel({ modelId: 'embed-v1', modelPath: '/path/to/model.onnx' });
const { embedding } = await EmbeddingPlugin.embed({
  modelId: 'embed-v1',
  text: 'Hello, world!',
  pooling: 'mean',
});
console.log('embedding dim:', embedding.length);

// Compute cosine similarity
const { score } = await EmbeddingPlugin.cosineSimilarity({ a: embedding, b: otherEmbedding });
```

## Project structure

```
capacitor-embeddings/
├── package.json                         # v0.1.0, peer deps: @capacitor/core ^7||^8, dust-core-capacitor >=0.1.0
├── Package.swift                        # SPM: EmbeddingsPlugin target, depends on dust-embeddings-swift
├── DustCapacitorEmbeddings.podspec      # CocoaPods: depends on DustEmbeddings, DustCapacitorCore
├── src/
│   ├── definitions.ts                   # EmbeddingPlugin interface, tokenizer types, pooling enums
│   ├── plugin.ts                        # WebPlugin stub (all methods throw "unimplemented")
│   └── index.ts                         # Barrel export
├── ios/Sources/EmbeddingsPlugin/
│   └── EmbeddingsPlugin.swift           # CAPPlugin bridge, DustCoreRegistry EmbeddingService registration
├── android/
│   ├── build.gradle                     # api project(':dust-embeddings-kotlin')
│   └── src/main/java/.../EmbeddingsPlugin.kt
└── tests/unit/                          # TypeScript unit tests (Vitest)
```

## Dependency chain

```
capacitor-embeddings
  ├── dust-core-capacitor  (peer)
  ├── dust-embeddings-kotlin  (Android native)
  │   ├── dust-core-kotlin
  │   ├── dust-onnx-kotlin
  │   └── dust-llm-kotlin
  └── dust-embeddings-swift  (iOS native)
      ├── dust-core-swift
      ├── dust-onnx-swift
      └── dust-llm-swift
```

## Related packages

| Package | Layer | Purpose |
|---------|-------|---------|
| [dust-core-capacitor](../capacitor-core/README.md) | Bridge | Shared contracts — `EmbeddingService` protocol |
| [dust-embeddings-kotlin](../dust-embeddings-kotlin/README.md) | Kotlin | Tokenizers, ONNX embedding sessions, vector math |
| [dust-embeddings-swift](../dust-embeddings-swift/README.md) | Swift | Tokenizers, ONNX embedding sessions, vector math |
| [dust-onnx-capacitor](../capacitor-onnx/README.md) | Bridge | ONNX Runtime model loading (underlying runtime) |

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
