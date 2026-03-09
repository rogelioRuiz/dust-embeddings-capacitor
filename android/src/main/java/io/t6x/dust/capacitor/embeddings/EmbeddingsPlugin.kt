package io.t6x.dust.capacitor.embeddings

import android.content.ComponentCallbacks2
import android.content.res.Configuration
import android.os.Handler
import android.os.HandlerThread
import com.getcapacitor.JSArray
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import io.t6x.dust.capacitor.serve.ServePlugin
import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.DustCoreRegistry
import io.t6x.dust.core.DustInputTensor
import io.t6x.dust.core.DustOutputTensor
import io.t6x.dust.core.ModelDescriptor
import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.ModelSession
import io.t6x.dust.core.ModelSessionFactory
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import io.t6x.dust.embeddings.EmbeddingResult
import io.t6x.dust.embeddings.EmbeddingSessionConfig
import io.t6x.dust.embeddings.EmbeddingSessionManager
import io.t6x.dust.embeddings.LlamaSessionGGUFEngine
import io.t6x.dust.embeddings.TokenizerOutput
import io.t6x.dust.embeddings.VectorMath
import io.t6x.dust.llm.LLMConfig
import io.t6x.dust.llm.LLMSessionManager
import io.t6x.dust.onnx.MemoryPressureLevel
import io.t6x.dust.onnx.ONNXConfig
import io.t6x.dust.onnx.ONNXError
import io.t6x.dust.onnx.ONNXSessionManager
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.android.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.Base64

@CapacitorPlugin(name = "Embeddings")
class EmbeddingsPlugin : Plugin(), ComponentCallbacks2 {
    private val workerThread = HandlerThread("embeddings-inference")
    private lateinit var dispatcher: CoroutineDispatcher
    private lateinit var scope: CoroutineScope
    private val sessionManager = EmbeddingSessionManager(ONNXSessionManager())
    private val ggufSessionManager = LLMSessionManager()

    override fun load() {
        workerThread.start()
        dispatcher = Handler(workerThread.looper).asCoroutineDispatcher()
        scope = CoroutineScope(dispatcher + SupervisorJob())
        DustCoreRegistry.getInstance().registerEmbeddingService(sessionManager)
        (bridge.getPlugin("Serve")?.getInstance() as? ServePlugin)
            ?.setSessionFactory(EmbeddingsSessionFactoryAdapter(), "embeddings")
        bridge.context.registerComponentCallbacks(this)
    }

    override fun handleOnDestroy() {
        bridge.context.unregisterComponentCallbacks(this)
        super.handleOnDestroy()
        if (::scope.isInitialized) {
            scope.cancel()
        }
        workerThread.quitSafely()
    }

    @PluginMethod
    fun loadModel(call: PluginCall) {
        val descriptor = call.getObject("descriptor")
        val modelId = descriptor?.getString("id")
        val format = descriptor?.getString("format")

        if (modelId.isNullOrEmpty() || format.isNullOrEmpty()) {
            call.reject("descriptor.id and descriptor.format are required", "invalidInput")
            return
        }

        if (format != ModelFormat.ONNX.value && format != ModelFormat.GGUF.value) {
            call.reject("Only onnx and gguf models are supported", "formatUnsupported")
            return
        }

        val modelPath = resolveModelPath(descriptor)
        if (modelPath.isNullOrEmpty()) {
            call.reject("descriptor.url or descriptor.metadata.localPath is required", "invalidInput")
            return
        }

        val config = try {
            parseSessionConfig(descriptor)
        } catch (error: DustCoreError) {
            call.reject(error.message ?: "Invalid input", error.code())
            return
        } catch (error: Throwable) {
            call.reject(error.message ?: "Invalid input", "invalidInput")
            return
        }

        val priority = SessionPriority.fromRawValue(call.getInt("priority") ?: SessionPriority.INTERACTIVE.rawValue)
            ?: SessionPriority.INTERACTIVE

        scope.launch {
            try {
                if (format == ModelFormat.GGUF.value) {
                    val session = loadGGUFEmbeddingModel(modelPath, modelId, config, priority)
                    call.resolve(
                        JSObject()
                            .put("modelId", session.sessionId)
                            .put("dims", session.config.dims)
                            .put("maxSequenceLength", session.config.maxSequenceLength),
                    )
                    return@launch
                }

                val vocabPath = resolveTokenizerVocabPath(descriptor)
                if (vocabPath.isNullOrEmpty()) {
                    call.reject("descriptor.tokenizerVocabUrl is required", "invalidInput")
                    return@launch
                }

                val mergesPath = resolveTokenizerMergesPath(descriptor)
                val session = sessionManager.loadModel(
                    modelPath = modelPath,
                    modelId = modelId,
                    vocabPath = vocabPath,
                    mergesPath = mergesPath,
                    config = config,
                    onnxConfig = ONNXConfig(),
                    priority = priority,
                )
                call.resolve(
                    JSObject()
                        .put("modelId", session.sessionId)
                        .put("dims", session.config.dims)
                        .put("maxSequenceLength", session.config.maxSequenceLength),
                )
            } catch (error: DustCoreError) {
                call.reject(error.message ?: "Failed to load model", error.code())
            } catch (error: ONNXError) {
                call.reject(error.message ?: "Failed to load model", error.code())
            } catch (error: Throwable) {
                call.reject(error.message ?: "Unknown error", "unknownError")
            }
        }
    }

    @PluginMethod
    fun unloadModel(call: PluginCall) {
        val modelId = call.getString("modelId")
        if (modelId.isNullOrEmpty()) {
            call.reject("modelId is required", "invalidInput")
            return
        }

        scope.launch {
            try {
                sessionManager.forceUnloadModel(modelId)
                call.resolve()
            } catch (error: DustCoreError) {
                call.reject(error.message ?: "Failed to unload model", error.code())
            } catch (error: ONNXError) {
                call.reject(error.message ?: "Failed to unload model", error.code())
            } catch (error: Throwable) {
                call.reject(error.message ?: "Unknown error", "unknownError")
            }
        }
    }

    @PluginMethod
    fun listLoadedModels(call: PluginCall) {
        val modelIds = JSArray()
        for (modelId in sessionManager.allModelIds()) {
            modelIds.put(modelId)
        }
        call.resolve(JSObject().put("modelIds", modelIds))
    }

    @PluginMethod
    fun getModelMetadata(call: PluginCall) {
        val modelId = call.getString("modelId")
        if (modelId.isNullOrEmpty()) {
            call.reject("modelId is required", "invalidInput")
            return
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        call.resolve(
            JSObject()
                .put("dims", session.config.dims)
                .put("maxSequenceLength", session.config.maxSequenceLength)
                .put("tokenizerType", session.config.tokenizerType)
                .put("pooling", session.config.pooling)
                .put("normalize", session.config.normalize)
                .put("backend", "onnx"),
        )
    }

    @PluginMethod
    fun embed(call: PluginCall) {
        val modelId = call.getString("modelId")
        val text = call.getString("text")
        if (modelId.isNullOrEmpty() || text == null) {
            call.reject("modelId and text are required", "invalidInput")
            return
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        val truncate = call.getBoolean("truncate") ?: true

        scope.launch {
            try {
                call.resolve(session.embed(text, truncate).toJSObject())
            } catch (error: DustCoreError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: ONNXError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: Throwable) {
                call.reject(error.message ?: "Unknown error", "unknownError")
            }
        }
    }

    @PluginMethod
    fun batchEmbed(call: PluginCall) {
        val modelId = call.getString("modelId")
        val textsArray = call.getArray("texts")
        if (modelId.isNullOrEmpty() || textsArray == null) {
            call.reject("modelId and texts are required", "invalidInput")
            return
        }

        val texts = mutableListOf<String>()
        for (index in 0 until textsArray.length()) {
            val text = textsArray.optString(index, null)
            if (text == null) {
                call.reject("texts must be an array of strings", "invalidInput")
                return
            }
            texts += text
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        val truncate = call.getBoolean("truncate") ?: true

        scope.launch {
            try {
                val results = session.embedBatch(texts, truncate)
                val embeddings = JSArray()
                for (result in results) {
                    embeddings.put(result.toJSObject())
                }
                call.resolve(JSObject().put("embeddings", embeddings))
            } catch (error: DustCoreError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: ONNXError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: Throwable) {
                call.reject(error.message ?: "Unknown error", "unknownError")
            }
        }
    }

    @PluginMethod
    fun embedImage(call: PluginCall) {
        val modelId = call.getString("modelId")
        if (modelId.isNullOrEmpty()) {
            call.reject("modelId is required", "invalidInput")
            return
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        val imageBytes = try {
            decodeImageBytes(call.getString("imageBase64"))
        } catch (error: DustCoreError) {
            call.reject(error.message ?: "Invalid input", error.code())
            return
        }

        scope.launch {
            try {
                call.resolve(session.embedImage(imageBytes).toJSObject())
            } catch (error: DustCoreError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: ONNXError) {
                call.reject(error.message ?: "Embedding failed", error.code())
            } catch (error: Throwable) {
                call.reject(error.message ?: "Unknown error", "unknownError")
            }
        }
    }

    @PluginMethod
    fun similarity(call: PluginCall) {
        val a = parseFloatArray(call.getArray("a"))
        val b = parseFloatArray(call.getArray("b"))

        if (a == null || b == null) {
            call.reject("a and b are required", "invalidInput")
            return
        }
        if (a.isEmpty() || a.size != b.size) {
            call.reject("a and b must be non-empty arrays of the same length", "invalidInput")
            return
        }

        call.resolve(
            JSObject().put("score", VectorMath.cosineSimilarity(a, b)),
        )
    }

    @PluginMethod
    fun countTokens(call: PluginCall) {
        val modelId = call.getString("modelId")
        val text = call.getString("text")
        if (modelId.isNullOrEmpty() || text == null) {
            call.reject("modelId and text are required", "invalidInput")
            return
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        val result = session.countTokens(text)
        call.resolve(
            JSObject()
                .put("count", result.count)
                .put("truncated", result.truncated),
        )
    }

    @PluginMethod
    fun tokenize(call: PluginCall) {
        val modelId = call.getString("modelId")
        val text = call.getString("text")
        if (modelId.isNullOrEmpty() || text == null) {
            call.reject("modelId and text are required", "invalidInput")
            return
        }

        val session = sessionManager.session(modelId)
        if (session == null) {
            call.reject("Model session not found", "modelNotFound")
            return
        }

        call.resolve(
            session.tokenize(text, call.getInt("maxLength")).toJSObject(),
        )
    }

    private fun parseSessionConfig(descriptor: JSObject?): EmbeddingSessionConfig {
        val safeDescriptor = descriptor ?: throw DustCoreError.InvalidInput("descriptor is required")
        val dims = safeDescriptor.getInteger("dims")
            ?: throw DustCoreError.InvalidInput("descriptor.dims is required")
        val maxSequenceLength = safeDescriptor.getInteger("maxSequenceLength")
            ?: throw DustCoreError.InvalidInput("descriptor.maxSequenceLength is required")
        val tokenizerType = safeDescriptor.getString("tokenizerType")
            ?: throw DustCoreError.InvalidInput("descriptor.tokenizerType is required")
        val pooling = safeDescriptor.getString("pooling")
            ?: throw DustCoreError.InvalidInput("descriptor.pooling is required")
        val normalize = safeDescriptor.getBoolean("normalize", null)
            ?: throw DustCoreError.InvalidInput("descriptor.normalize is required")

        val inputNamesObject = safeDescriptor.getJSObject("inputNames")
        val inputNames = EmbeddingSessionConfig.InputNames(
            inputIds = inputNamesObject?.getString("inputIds") ?: "input_ids",
            attentionMask = inputNamesObject?.getString("attentionMask") ?: "attention_mask",
            tokenTypeIds = inputNamesObject?.getString("tokenTypeIds") ?: "token_type_ids",
        )

        return EmbeddingSessionConfig(
            dims = dims,
            maxSequenceLength = maxSequenceLength,
            tokenizerType = tokenizerType,
            pooling = pooling,
            normalize = normalize,
            inputNames = inputNames,
            outputName = safeDescriptor.getString("outputName") ?: "last_hidden_state",
        )
    }

    private fun parseSessionConfig(descriptor: ModelDescriptor): EmbeddingSessionConfig {
        val metadata = descriptor.metadata
        val dims = metadata.intValue("dims")
            ?: throw DustCoreError.InvalidInput("descriptor.metadata.dims is required")
        val maxSequenceLength = metadata.intValue("maxSequenceLength")
            ?: throw DustCoreError.InvalidInput("descriptor.metadata.maxSequenceLength is required")
        val tokenizerType = metadata.stringValue("tokenizerType")
            ?: throw DustCoreError.InvalidInput("descriptor.metadata.tokenizerType is required")
        val pooling = metadata.stringValue("pooling")
            ?: throw DustCoreError.InvalidInput("descriptor.metadata.pooling is required")
        val normalize = metadata.booleanValue("normalize")
            ?: throw DustCoreError.InvalidInput("descriptor.metadata.normalize is required")

        val inputNames = EmbeddingSessionConfig.InputNames(
            inputIds = metadata.stringValue("inputIds") ?: "input_ids",
            attentionMask = metadata.stringValue("attentionMask") ?: "attention_mask",
            tokenTypeIds = metadata.stringValue("tokenTypeIds") ?: "token_type_ids",
        )

        return EmbeddingSessionConfig(
            dims = dims,
            maxSequenceLength = maxSequenceLength,
            tokenizerType = tokenizerType,
            pooling = pooling,
            normalize = normalize,
            inputNames = inputNames,
            outputName = metadata.stringValue("outputName") ?: "last_hidden_state",
        )
    }

    private fun resolveModelPath(descriptor: JSObject?): String? {
        if (descriptor == null) {
            return null
        }

        val url = descriptor.getString("url")
        if (!url.isNullOrEmpty()) {
            return url
        }

        val metadata = descriptor.getJSObject("metadata")
        val localPath = metadata?.getString("localPath")
        if (!localPath.isNullOrEmpty()) {
            return localPath
        }

        return null
    }

    private fun resolveModelPath(descriptor: ModelDescriptor): String? {
        val localPath = descriptor.metadata?.get("localPath")
        if (!localPath.isNullOrEmpty()) {
            return localPath
        }

        return descriptor.url
    }

    private fun resolveTokenizerVocabPath(descriptor: JSObject?): String? {
        if (descriptor == null) {
            return null
        }

        val vocabPath = descriptor.getString("tokenizerVocabUrl")
        if (!vocabPath.isNullOrEmpty()) {
            return vocabPath
        }

        val metadata = descriptor.getJSObject("metadata")
        val metadataPath = metadata?.getString("tokenizerVocabPath")
            ?: metadata?.getString("tokenizerVocabLocalPath")
        if (!metadataPath.isNullOrEmpty()) {
            return metadataPath
        }

        return null
    }

    private fun resolveTokenizerVocabPath(descriptor: ModelDescriptor): String? {
        val metadata = descriptor.metadata
        return metadata.stringValue("tokenizerVocabPath")
            ?: metadata.stringValue("tokenizerVocabLocalPath")
    }

    private fun resolveTokenizerMergesPath(descriptor: JSObject?): String? {
        if (descriptor == null) {
            return null
        }

        val mergesPath = descriptor.getString("tokenizerMergesUrl")
        if (!mergesPath.isNullOrEmpty()) {
            return mergesPath
        }

        val metadata = descriptor.getJSObject("metadata")
        val metadataPath = metadata?.getString("tokenizerMergesPath")
            ?: metadata?.getString("tokenizerMergesLocalPath")
        if (!metadataPath.isNullOrEmpty()) {
            return metadataPath
        }

        return null
    }

    private fun resolveTokenizerMergesPath(descriptor: ModelDescriptor): String? {
        val metadata = descriptor.metadata
        return metadata.stringValue("tokenizerMergesPath")
            ?: metadata.stringValue("tokenizerMergesLocalPath")
    }

    private fun loadGGUFEmbeddingModel(
        modelPath: String,
        modelId: String,
        config: EmbeddingSessionConfig,
        priority: SessionPriority,
    ) = sessionManager.loadGGUFModel(
        modelId = modelId,
        engine = LlamaSessionGGUFEngine(
            ggufSessionManager.loadModel(modelPath, modelId, LLMConfig(), priority),
        ) {
            scope.launch {
                try {
                    ggufSessionManager.unloadModel(modelId)
                } catch (_: Throwable) {
                }
            }
        },
        config = config,
    )

    private fun Map<String, String>?.stringValue(key: String): String? =
        this?.get(key)?.takeIf { it.isNotEmpty() }

    private fun Map<String, String>?.intValue(key: String): Int? =
        stringValue(key)?.toIntOrNull()

    private fun Map<String, String>?.booleanValue(key: String): Boolean? =
        when (stringValue(key)?.lowercase()) {
            "true", "1" -> true
            "false", "0" -> false
            else -> null
        }

    private inner class EmbeddingsSessionFactoryAdapter : ModelSessionFactory {
        override suspend fun makeSession(descriptor: ModelDescriptor, priority: SessionPriority): ModelSession {
            val modelPath = resolveModelPath(descriptor)
                ?: throw DustCoreError.InvalidInput("descriptor.url or descriptor.metadata.localPath is required")
            val config = parseSessionConfig(descriptor)

            when (descriptor.format) {
                ModelFormat.ONNX -> {
                    val vocabPath = resolveTokenizerVocabPath(descriptor)
                        ?: throw DustCoreError.InvalidInput("descriptor.metadata.tokenizerVocabPath is required")
                    val mergesPath = resolveTokenizerMergesPath(descriptor)
                    sessionManager.loadModel(
                        modelPath = modelPath,
                        modelId = descriptor.id,
                        vocabPath = vocabPath,
                        mergesPath = mergesPath,
                        config = config,
                        onnxConfig = ONNXConfig(),
                        priority = priority,
                    )
                }
                ModelFormat.GGUF -> loadGGUFEmbeddingModel(modelPath, descriptor.id, config, priority)
                else -> throw DustCoreError.FormatUnsupported
            }

            return EmbeddingsModelSessionAdapter(descriptor.id, priority)
        }
    }

    private inner class EmbeddingsModelSessionAdapter(
        private val modelId: String,
        private val sessionPriority: SessionPriority,
    ) : ModelSession {
        override suspend fun predict(inputs: List<DustInputTensor>): List<DustOutputTensor> = emptyList()

        override fun status(): ModelStatus {
            return if (sessionManager.session(modelId) != null) {
                ModelStatus.Ready
            } else {
                ModelStatus.NotLoaded
            }
        }

        override fun priority(): SessionPriority = sessionPriority

        override suspend fun close() {
            try {
                sessionManager.unloadModel(modelId)
            } catch (error: DustCoreError) {
                if (error !is DustCoreError.ModelNotFound) {
                    throw error
                }
                return
            }

            if (sessionManager.refCount(modelId) == 0) {
                sessionManager.forceUnloadModel(modelId)
            }
        }
    }

    private fun decodeImageBytes(imageBase64: String?): ByteArray {
        if (imageBase64.isNullOrEmpty()) {
            throw DustCoreError.InvalidInput("imageBase64 is required")
        }

        return try {
            Base64.getDecoder().decode(imageBase64)
        } catch (_: IllegalArgumentException) {
            throw DustCoreError.InvalidInput("imageBase64 must be valid base64")
        }
    }

    private fun parseFloatArray(array: JSArray?): List<Float>? {
        if (array == null) {
            return null
        }

        return try {
            List(array.length()) { index -> array.getDouble(index).toFloat() }
        } catch (_: Throwable) {
            null
        }
    }

    @Suppress("DEPRECATION")
    override fun onTrimMemory(level: Int) {
        val pressureLevel = when {
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> MemoryPressureLevel.CRITICAL
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> MemoryPressureLevel.STANDARD
            level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> MemoryPressureLevel.CRITICAL
            else -> null
        }

        if (pressureLevel != null && ::scope.isInitialized) {
            scope.launch { sessionManager.evictUnderPressure(pressureLevel) }
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {}

    @Deprecated("Required legacy fallback for Android low-memory callbacks")
    @Suppress("DEPRECATION")
    override fun onLowMemory() {
        if (::scope.isInitialized) {
            scope.launch { sessionManager.evictUnderPressure(MemoryPressureLevel.CRITICAL) }
        }
    }
}

private fun EmbeddingResult.toJSObject(): JSObject {
    val embeddingArray = JSArray()
    for (value in embedding) {
        embeddingArray.put(value.toDouble())
    }
    return JSObject()
        .put("embedding", embeddingArray)
        .put("tokenCount", tokenCount)
        .put("truncated", truncated)
        .put("modelId", modelId)
}

private fun TokenizerOutput.toJSObject(): JSObject {
    val inputIdsArray = JSArray()
    val attentionMaskArray = JSArray()
    val tokenTypeIdsArray = JSArray()

    for (value in inputIds) {
        inputIdsArray.put(value)
    }
    for (value in attentionMask) {
        attentionMaskArray.put(value)
    }
    for (value in tokenTypeIds) {
        tokenTypeIdsArray.put(value)
    }

    return JSObject()
        .put("inputIds", inputIdsArray)
        .put("attentionMask", attentionMaskArray)
        .put("tokenTypeIds", tokenTypeIdsArray)
}

private fun DustCoreError.code(): String = when (this) {
    is DustCoreError.ModelNotFound -> "modelNotFound"
    is DustCoreError.ModelNotReady -> "modelNotReady"
    is DustCoreError.FormatUnsupported -> "formatUnsupported"
    is DustCoreError.SessionClosed -> "sessionClosed"
    is DustCoreError.InvalidInput -> "invalidInput"
    is DustCoreError.InferenceFailed -> "inferenceFailed"
    else -> "unknownError"
}
