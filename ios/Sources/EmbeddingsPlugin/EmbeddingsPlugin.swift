import Capacitor
import Foundation
import DustCore
import DustOnnx
import DustEmbeddings
import DustLlm
#if canImport(ServePlugin)
import ServePlugin
#endif
import UIKit

@objc(EmbeddingsPlugin)
public class EmbeddingsPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "EmbeddingsPlugin"
    public let jsName = "Embeddings"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "loadModel", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "unloadModel", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "listLoadedModels", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getModelMetadata", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "embed", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "batchEmbed", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "embedImage", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "similarity", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "countTokens", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "tokenize", returnType: CAPPluginReturnPromise),
    ]

    private let sessionManager: EmbeddingSessionManager
    private let ggufSessionManager = LLMSessionManager()

    public override init() {
        let onnxManager = ONNXSessionManager()
        self.sessionManager = EmbeddingSessionManager(onnxSessionManager: onnxManager)
        super.init()
    }

    public override func load() {
        super.load()
        DustCoreRegistry.shared.register(embeddingService: sessionManager)
        #if canImport(ServePlugin)
        if let servePlugin = bridge?.plugin(withName: "Serve") as? ServePlugin {
            servePlugin.setSessionFactory(EmbeddingsSessionFactoryAdapter(plugin: self), for: "embeddings")
        }
        #endif
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
    }

    @objc func loadModel(_ call: CAPPluginCall) {
        guard let descriptor = call.getObject("descriptor"),
              let modelId = descriptor["id"] as? String,
              let format = descriptor["format"] as? String else {
            call.reject("descriptor.id and descriptor.format are required", "invalidInput", nil)
            return
        }

        guard format == DustModelFormat.onnx.rawValue || format == DustModelFormat.gguf.rawValue else {
            call.reject("Only onnx and gguf models are supported", "formatUnsupported", nil)
            return
        }

        guard let modelPath = Self.resolveModelPath(from: descriptor) else {
            call.reject("descriptor.url or descriptor.metadata.localPath is required", "invalidInput", nil)
            return
        }

        let priority = DustSessionPriority(rawValue: call.getInt("priority") ?? DustSessionPriority.interactive.rawValue)
            ?? .interactive

        let config: EmbeddingSessionConfig
        do {
            config = try Self.parseSessionConfig(from: descriptor)
        } catch let error as DustCoreError {
            call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            return
        } catch {
            call.reject(error.localizedDescription, "invalidInput", error)
            return
        }

        EmbeddingSessionManager.inferenceQueue.async {
            do {
                if format == DustModelFormat.gguf.rawValue {
                    let session = try self.loadGGUFEmbeddingModel(
                        modelPath: modelPath,
                        modelId: modelId,
                        config: config,
                        priority: priority
                    )
                    call.resolve([
                        "modelId": session.sessionId,
                        "dims": session.config.dims,
                        "maxSequenceLength": session.config.maxSequenceLength,
                    ])
                    return
                }

                guard let vocabPath = Self.resolveTokenizerVocabPath(from: descriptor) else {
                    call.reject("descriptor.tokenizerVocabUrl is required", "invalidInput", nil)
                    return
                }

                let mergesPath = Self.resolveTokenizerMergesPath(from: descriptor)
                let session = try self.sessionManager.loadModel(
                    modelPath: modelPath,
                    modelId: modelId,
                    vocabPath: vocabPath,
                    mergesPath: mergesPath,
                    config: config,
                    onnxConfig: ONNXConfig(),
                    priority: priority
                )
                call.resolve([
                    "modelId": session.sessionId,
                    "dims": session.config.dims,
                    "maxSequenceLength": session.config.maxSequenceLength,
                ])
            } catch let error as DustCoreError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch let error as ONNXError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch {
                call.reject(error.localizedDescription, "unknownError", error)
            }
        }
    }

    @objc func unloadModel(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId") else {
            call.reject("modelId is required", "invalidInput", nil)
            return
        }

        Task {
            do {
                try await sessionManager.forceUnloadModel(id: modelId)
                call.resolve()
            } catch let error as DustCoreError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch let error as ONNXError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch {
                call.reject(error.localizedDescription, "unknownError", error)
            }
        }
    }

    @objc func listLoadedModels(_ call: CAPPluginCall) {
        call.resolve([
            "modelIds": sessionManager.allModelIds(),
        ])
    }

    @objc func getModelMetadata(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId") else {
            call.reject("modelId is required", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        call.resolve([
            "dims": session.config.dims,
            "maxSequenceLength": session.config.maxSequenceLength,
            "tokenizerType": session.config.tokenizerType,
            "pooling": session.config.pooling,
            "normalize": session.config.normalize,
            "backend": "onnx",
        ])
    }

    @objc func embed(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let text = call.getString("text") else {
            call.reject("modelId and text are required", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        let truncate = call.getBool("truncate") ?? true

        EmbeddingSessionManager.inferenceQueue.async {
            do {
                call.resolve(Self.toJSObject(try session.embed(text: text, truncate: truncate)))
            } catch let error as DustCoreError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch let error as ONNXError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch {
                call.reject(error.localizedDescription, "unknownError", error)
            }
        }
    }

    @objc func batchEmbed(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let values = call.getArray("texts") else {
            call.reject("modelId and texts are required", "invalidInput", nil)
            return
        }

        let texts = values.compactMap { $0 as? String }
        guard texts.count == values.count else {
            call.reject("texts must be an array of strings", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        let truncate = call.getBool("truncate") ?? true

        EmbeddingSessionManager.inferenceQueue.async {
            do {
                let results = try session.embedBatch(texts: texts, truncate: truncate)
                call.resolve([
                    "embeddings": results.map(Self.toJSObject),
                ])
            } catch let error as DustCoreError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch let error as ONNXError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch {
                call.reject(error.localizedDescription, "unknownError", error)
            }
        }
    }

    @objc func embedImage(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId") else {
            call.reject("modelId is required", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        let imageData: Data
        do {
            imageData = try Self.decodeRequiredImageData(from: call.getString("imageBase64"))
        } catch let error as DustCoreError {
            call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            return
        } catch {
            call.reject(error.localizedDescription, "invalidInput", error)
            return
        }

        EmbeddingSessionManager.inferenceQueue.async {
            do {
                call.resolve(Self.toJSObject(try session.embedImage(imageData: imageData)))
            } catch let error as DustCoreError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch let error as ONNXError {
                call.reject(Self.errorMessage(for: error), Self.errorCode(for: error), error)
            } catch {
                call.reject(error.localizedDescription, "unknownError", error)
            }
        }
    }

    @objc func similarity(_ call: CAPPluginCall) {
        guard let a = Self.parseFloatArray(from: call.getArray("a")),
              let b = Self.parseFloatArray(from: call.getArray("b")) else {
            call.reject("a and b are required", "invalidInput", nil)
            return
        }

        guard a.count == b.count, !a.isEmpty else {
            call.reject("a and b must be non-empty arrays of the same length", "invalidInput", nil)
            return
        }

        call.resolve([
            "score": VectorMath.cosineSimilarity(a, b),
        ])
    }

    @objc func countTokens(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let text = call.getString("text") else {
            call.reject("modelId and text are required", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        let result = session.countTokens(text: text)
        call.resolve([
            "count": result.count,
            "truncated": result.truncated,
        ])
    }

    @objc func tokenize(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let text = call.getString("text") else {
            call.reject("modelId and text are required", "invalidInput", nil)
            return
        }

        guard let session = sessionManager.session(for: modelId) else {
            call.reject("Model session not found", "modelNotFound", nil)
            return
        }

        let output = session.tokenize(text: text, maxLength: call.getInt("maxLength"))
        call.resolve([
            "inputIds": output.inputIds.map(Int.init),
            "attentionMask": output.attentionMask.map(Int.init),
            "tokenTypeIds": output.tokenTypeIds.map(Int.init),
        ])
    }

    @objc private func handleMemoryWarning() {
        Task {
            await sessionManager.evictUnderPressure(level: .critical)
        }
    }

    private static func toJSObject(_ result: EmbeddingResult) -> [String: Any] {
        [
            "embedding": result.embedding.map(Double.init),
            "tokenCount": result.tokenCount,
            "truncated": result.truncated,
            "modelId": result.modelId,
        ]
    }

    private static func parseSessionConfig(from descriptor: [AnyHashable: Any]) throws -> EmbeddingSessionConfig {
        guard let dims = (descriptor["dims"] as? NSNumber)?.intValue, dims > 0 else {
            throw DustCoreError.invalidInput(detail: "descriptor.dims is required")
        }
        guard let maxSequenceLength = (descriptor["maxSequenceLength"] as? NSNumber)?.intValue,
              maxSequenceLength > 0 else {
            throw DustCoreError.invalidInput(detail: "descriptor.maxSequenceLength is required")
        }
        guard let tokenizerType = descriptor["tokenizerType"] as? String, !tokenizerType.isEmpty else {
            throw DustCoreError.invalidInput(detail: "descriptor.tokenizerType is required")
        }
        guard let pooling = descriptor["pooling"] as? String, !pooling.isEmpty else {
            throw DustCoreError.invalidInput(detail: "descriptor.pooling is required")
        }
        let normalize: Bool
        if let boolValue = descriptor["normalize"] as? Bool {
            normalize = boolValue
        } else if let numberValue = descriptor["normalize"] as? NSNumber {
            normalize = numberValue.boolValue
        } else {
            throw DustCoreError.invalidInput(detail: "descriptor.normalize is required")
        }

        let inputNamesObject = descriptor["inputNames"] as? [AnyHashable: Any]
        let inputNames = EmbeddingSessionConfig.InputNames(
            inputIds: inputNamesObject?["inputIds"] as? String ?? "input_ids",
            attentionMask: inputNamesObject?["attentionMask"] as? String ?? "attention_mask",
            tokenTypeIds: inputNamesObject?["tokenTypeIds"] as? String ?? "token_type_ids"
        )

        return EmbeddingSessionConfig(
            dims: dims,
            maxSequenceLength: maxSequenceLength,
            tokenizerType: tokenizerType,
            pooling: pooling,
            normalize: normalize,
            inputNames: inputNames,
            outputName: descriptor["outputName"] as? String ?? "last_hidden_state"
        )
    }

    private static func parseSessionConfig(from descriptor: DustModelDescriptor) throws -> EmbeddingSessionConfig {
        let metadata = descriptor.metadata

        guard let dims = Self.intValue(in: metadata, for: "dims"), dims > 0 else {
            throw DustCoreError.invalidInput(detail: "descriptor.metadata.dims is required")
        }
        guard let maxSequenceLength = Self.intValue(in: metadata, for: "maxSequenceLength"),
              maxSequenceLength > 0 else {
            throw DustCoreError.invalidInput(detail: "descriptor.metadata.maxSequenceLength is required")
        }
        guard let tokenizerType = Self.stringValue(in: metadata, for: "tokenizerType") else {
            throw DustCoreError.invalidInput(detail: "descriptor.metadata.tokenizerType is required")
        }
        guard let pooling = Self.stringValue(in: metadata, for: "pooling") else {
            throw DustCoreError.invalidInput(detail: "descriptor.metadata.pooling is required")
        }
        guard let normalize = Self.boolValue(in: metadata, for: "normalize") else {
            throw DustCoreError.invalidInput(detail: "descriptor.metadata.normalize is required")
        }

        let inputNames = EmbeddingSessionConfig.InputNames(
            inputIds: Self.stringValue(in: metadata, for: "inputIds") ?? "input_ids",
            attentionMask: Self.stringValue(in: metadata, for: "attentionMask") ?? "attention_mask",
            tokenTypeIds: Self.stringValue(in: metadata, for: "tokenTypeIds") ?? "token_type_ids"
        )

        return EmbeddingSessionConfig(
            dims: dims,
            maxSequenceLength: maxSequenceLength,
            tokenizerType: tokenizerType,
            pooling: pooling,
            normalize: normalize,
            inputNames: inputNames,
            outputName: Self.stringValue(in: metadata, for: "outputName") ?? "last_hidden_state"
        )
    }

    private static func resolveModelPath(from descriptor: [AnyHashable: Any]) -> String? {
        if let url = descriptor["url"] as? String, !url.isEmpty {
            return url
        }

        if let metadata = descriptor["metadata"] as? [AnyHashable: Any],
           let localPath = metadata["localPath"] as? String,
           !localPath.isEmpty {
            return localPath
        }

        return nil
    }

    private static func resolveModelPath(from descriptor: DustModelDescriptor) -> String? {
        if let localPath = descriptor.metadata?["localPath"], !localPath.isEmpty {
            return localPath
        }

        return descriptor.url
    }

    private static func resolveTokenizerVocabPath(from descriptor: [AnyHashable: Any]) -> String? {
        if let vocabPath = descriptor["tokenizerVocabUrl"] as? String, !vocabPath.isEmpty {
            return vocabPath
        }

        if let metadata = descriptor["metadata"] as? [AnyHashable: Any] {
            if let vocabPath = metadata["tokenizerVocabPath"] as? String, !vocabPath.isEmpty {
                return vocabPath
            }
            if let vocabPath = metadata["tokenizerVocabLocalPath"] as? String, !vocabPath.isEmpty {
                return vocabPath
            }
        }

        return nil
    }

    private static func resolveTokenizerVocabPath(from descriptor: DustModelDescriptor) -> String? {
        if let metadata = descriptor.metadata {
            if let vocabPath = metadata["tokenizerVocabPath"], !vocabPath.isEmpty {
                return vocabPath
            }
            if let vocabPath = metadata["tokenizerVocabLocalPath"], !vocabPath.isEmpty {
                return vocabPath
            }
        }

        return nil
    }

    private static func resolveTokenizerMergesPath(from descriptor: [AnyHashable: Any]) -> String? {
        if let mergesPath = descriptor["tokenizerMergesUrl"] as? String, !mergesPath.isEmpty {
            return mergesPath
        }

        if let metadata = descriptor["metadata"] as? [AnyHashable: Any] {
            if let mergesPath = metadata["tokenizerMergesPath"] as? String, !mergesPath.isEmpty {
                return mergesPath
            }
            if let mergesPath = metadata["tokenizerMergesLocalPath"] as? String, !mergesPath.isEmpty {
                return mergesPath
            }
        }

        return nil
    }

    private static func resolveTokenizerMergesPath(from descriptor: DustModelDescriptor) -> String? {
        if let metadata = descriptor.metadata {
            if let mergesPath = metadata["tokenizerMergesPath"], !mergesPath.isEmpty {
                return mergesPath
            }
            if let mergesPath = metadata["tokenizerMergesLocalPath"], !mergesPath.isEmpty {
                return mergesPath
            }
        }

        return nil
    }

    private static func stringValue(in metadata: [String: String]?, for key: String) -> String? {
        guard let value = metadata?[key], !value.isEmpty else {
            return nil
        }
        return value
    }

    private static func intValue(in metadata: [String: String]?, for key: String) -> Int? {
        guard let value = stringValue(in: metadata, for: key) else {
            return nil
        }
        return Int(value)
    }

    private static func boolValue(in metadata: [String: String]?, for key: String) -> Bool? {
        switch stringValue(in: metadata, for: key)?.lowercased() {
        case "true", "1":
            return true
        case "false", "0":
            return false
        default:
            return nil
        }
    }

    private func loadGGUFEmbeddingModel(
        modelPath: String,
        modelId: String,
        config: EmbeddingSessionConfig,
        priority: DustSessionPriority
    ) throws -> EmbeddingSession {
        let llmSession = try ggufSessionManager.loadModel(
            path: modelPath,
            modelId: modelId,
            config: LLMConfig(),
            priority: priority
        )
        let engine = LlamaSessionGGUFEngine(session: llmSession) { [weak self] in
            Task {
                guard let self else {
                    return
                }
                try? await self.ggufSessionManager.unloadModel(id: modelId)
            }
        }
        return sessionManager.loadGGUFModel(
            modelId: modelId,
            engine: engine,
            config: config
        )
    }

    private final class EmbeddingsSessionFactoryAdapter: DustModelSessionFactory, @unchecked Sendable {
        private weak var plugin: EmbeddingsPlugin?

        init(plugin: EmbeddingsPlugin) {
            self.plugin = plugin
        }

        func makeSession(
            descriptor: DustModelDescriptor,
            priority: DustSessionPriority
        ) async throws -> any DustModelSession {
            guard let plugin else {
                throw DustCoreError.inferenceFailed(detail: "EmbeddingsPlugin unavailable")
            }

            guard let modelPath = EmbeddingsPlugin.resolveModelPath(from: descriptor) else {
                throw DustCoreError.invalidInput(
                    detail: "descriptor.url or descriptor.metadata.localPath is required"
                )
            }

            let config = try EmbeddingsPlugin.parseSessionConfig(from: descriptor)

            switch descriptor.format {
            case .onnx:
                guard let vocabPath = EmbeddingsPlugin.resolveTokenizerVocabPath(from: descriptor) else {
                    throw DustCoreError.invalidInput(
                        detail: "descriptor.metadata.tokenizerVocabPath is required"
                    )
                }
                let mergesPath = EmbeddingsPlugin.resolveTokenizerMergesPath(from: descriptor)
                _ = try plugin.sessionManager.loadModel(
                    modelPath: modelPath,
                    modelId: descriptor.id,
                    vocabPath: vocabPath,
                    mergesPath: mergesPath,
                    config: config,
                    onnxConfig: ONNXConfig(),
                    priority: priority
                )
            case .gguf:
                _ = try plugin.loadGGUFEmbeddingModel(
                    modelPath: modelPath,
                    modelId: descriptor.id,
                    config: config,
                    priority: priority
                )
            default:
                throw DustCoreError.formatUnsupported
            }

            return EmbeddingsModelSessionAdapter(
                modelId: descriptor.id,
                priority: priority,
                sessionManager: plugin.sessionManager
            )
        }
    }

    private final class EmbeddingsModelSessionAdapter: DustModelSession, @unchecked Sendable {
        private let modelId: String
        private let priorityValue: DustSessionPriority
        private let sessionManager: EmbeddingSessionManager

        init(
            modelId: String,
            priority: DustSessionPriority,
            sessionManager: EmbeddingSessionManager
        ) {
            self.modelId = modelId
            self.priorityValue = priority
            self.sessionManager = sessionManager
        }

        func predict(inputs: [DustInputTensor]) async throws -> [DustOutputTensor] {
            []
        }

        func status() -> DustModelStatus {
            sessionManager.session(for: modelId) == nil ? .notLoaded : .ready
        }

        func priority() -> DustSessionPriority {
            priorityValue
        }

        func close() async throws {
            do {
                try sessionManager.unloadModel(id: modelId)
            } catch let error as DustCoreError where error == .modelNotFound {
                return
            }

            if sessionManager.refCount(for: modelId) == 0 {
                try await sessionManager.forceUnloadModel(id: modelId)
            }
        }
    }

    private static func decodeRequiredImageData(from imageBase64: String?) throws -> Data {
        guard let imageBase64, !imageBase64.isEmpty else {
            throw DustCoreError.invalidInput(detail: "imageBase64 is required")
        }

        guard let data = Data(base64Encoded: imageBase64) else {
            throw DustCoreError.invalidInput(detail: "imageBase64 must be valid base64")
        }

        return data
    }

    private static func parseFloatArray(from values: [Any]?) -> [Float]? {
        guard let values else {
            return nil
        }

        let floats = values.compactMap { value -> Float? in
            if let number = value as? NSNumber {
                return number.floatValue
            }
            return nil
        }

        return floats.count == values.count ? floats : nil
    }

    private static func errorCode(for error: DustCoreError) -> String {
        switch error {
        case .modelNotFound:
            return "modelNotFound"
        case .modelNotReady:
            return "modelNotReady"
        case .formatUnsupported:
            return "formatUnsupported"
        case .sessionClosed:
            return "sessionClosed"
        case .invalidInput:
            return "invalidInput"
        case .inferenceFailed:
            return "inferenceFailed"
        default:
            return "unknownError"
        }
    }

    private static func errorMessage(for error: DustCoreError) -> String {
        switch error {
        case .modelNotFound:
            return "Model session not found"
        case .modelNotReady:
            return "Model session is busy"
        case .sessionClosed:
            return "Model session is closed"
        case .formatUnsupported:
            return "Model format not supported"
        case .invalidInput(let detail):
            return detail ?? "Invalid input"
        case .inferenceFailed(let detail):
            return detail ?? "Inference failed"
        default:
            return "Unknown error"
        }
    }

    private static func errorCode(for error: ONNXError) -> String {
        switch error {
        case .fileNotFound, .loadFailed, .inferenceError, .preprocessError:
            return "inferenceFailed"
        case .formatUnsupported:
            return "formatUnsupported"
        case .sessionClosed:
            return "sessionClosed"
        case .modelEvicted:
            return "modelEvicted"
        case .shapeError:
            return "shapeError"
        case .dtypeError:
            return "dtypeError"
        }
    }

    private static func errorMessage(for error: ONNXError) -> String {
        switch error {
        case .fileNotFound(let path):
            return "Model file not found at \(path)"
        case .loadFailed(let path, _):
            return "Failed to load ONNX model at \(path)"
        case .formatUnsupported(let format):
            return "Unsupported model format: \(format)"
        case .sessionClosed:
            return "Model session is closed"
        case .modelEvicted:
            return "Model was evicted from memory"
        case .shapeError(let name, let expected, let got):
            return "Shape mismatch for \(name): expected \(expected), got \(got)"
        case .dtypeError(let name, let expected, let got):
            return "Dtype mismatch for \(name): expected \(expected), got \(got)"
        case .inferenceError(let detail):
            return detail
        case .preprocessError(let detail):
            return detail
        }
    }
}
