// file: semanticTurningPointDetector.ts
import fs from "fs-extra";
import winston from "winston";
import { Ollama } from "ollama";

import dotenv from "dotenv";
dotenv.config();

import async from "async";
import { OpenAI } from "openai";
import { LRUCache } from "lru-cache";
import crypto from "crypto";
import { countTokens, createEmbeddingCache } from "./tokensUtil";
import { MetaMessage, Message, MessageSpan } from "./Message";
import { returnFormattedMessageContent } from "./stripContent";
import {
  circularToneSimilarity,
  computeSignificance,
  computeSignificanceWithChoquet,
  formAnalysisResponseFormat,
  formAnalysisSystemPromptEnding,
  formScoringResponseFormat,
  formScoringSystemPromptEnding,
  formSystemMessage,
} from "./prompt";
import {
  ChunkingResult,
  ConvergenceState,
  EpistemicPrimitives,
  MessageEmbedding,
  TurningPoint,
  turningPointCategories,
  TurningPointCategory,
  TurningPointDetectorConfig,
} from "./types";
import { CounterfactualAnalyzer } from "./counterfactual";

// Cache for token counts to avoid recalculating - implements atomic memory concept
const tokenCountCache = new LRUCache<string, number>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 24,
});

/**
 * Semantic Turning Point Detector using ARC/CRA/DAO Framework
 * 
 * Identifies semantically significant moments in conversations where meaning shifts,
 * emotional ruptures occur, or intellectual pivots happen. Uses multi-dimensional 
 * analysis with optional phi-field enhancement for improved accuracy.
 * 
 * @example Basic Usage
 * ```typescript
 * const detector = new SemanticTurningPointDetector({
 *   apiKey: process.env.OPENAI_API_KEY,
 *   classificationModel: "gpt-4o-mini",
 *   semanticShiftThreshold: 0.4,
 *   maxTurningPoints: 10
 * });
 * 
 * const result = await detector.detectTurningPoints(messages);
 * console.log(`Found ${result.points.length} turning points`);
 * ```
 * 
 * @example Advanced Configuration
 * ```typescript
 * const detector = new SemanticTurningPointDetector({
 *   enableExperimentalPhi: true,          // Enhanced significance scoring
 *   enableCounterfactualAnalysis: true,   // Additional validation
 *   maxRecursionDepth: 4,                 // Multi-dimensional analysis depth
 *   dynamicallyAdjustSemanticShiftThreshold: true
 * });
 * ```
 * 
 * ## Key Parameters
 * - `semanticShiftThreshold` (0.2-0.7): Sensitivity control
 * - `maxRecursionDepth` (2-5): Analysis depth  
 * - `enableExperimentalPhi`: Activates phi-field enhancement
 * - `maxTurningPoints`: Limit final results
 * 
 * ## Scoring
 * - **Confidence**: How semantically distinct the turning point is
 * - **Necessity**: How epistemically essential it is (phi-aware)
 * 
 * @see {@link TurningPointDetectorConfig} Configuration options
 * @see {@link TurningPoint} Result structure
 */
export class SemanticTurningPointDetector {
  private config: TurningPointDetectorConfig;

  /**
   * For ease of use in llm requests, openai's client is used as it allows configurable endpoints. Further expoloration might be reasonable in leveraging other libaries, such as ollama, llmstudio, genai, etc, for more direct compatibility with other LLM providers. Though at this time, the OpenAI client is sufficient for requests done by this detector.
   */
  private openai: OpenAI;
  /**
   * This provides the array of the initial messages that were passed to the detector. This is noted as such as throughout the process, ARC involves analyzing subsets of the original messages, and the original messages are not modified.
   */
  private originalMessages: Message[] = [];
  /**
   * AN array of changes of state across iterations, used for convergence measurement.
   * This is used to track the evolution of turning points across iterations and dimensions.
   * This is used when returning the final results, to determine whether the turning points have converged.
   */
  private convergenceHistory: ConvergenceState[] = [];
  /**
   * Used to help mitigate repeat embedding requests for the same message content. And can be configured to avoid excessive RAM usage via `embeddingCacheRamLimitMB`.
   */
  private embeddingCache: LRUCache<string, Float32Array>;

  private endpointType: "ollama" | "openai" | "unknown" | "openrouter" =
    "unknown";

  private ollama: Ollama | null = null;
  readonly logger: winston.Logger | Console;
  private counterfactualAnalyzer?: CounterfactualAnalyzer;

  /**
   * Creates a new instance of the semantic turning point detector
   */
  constructor(config: Partial<TurningPointDetectorConfig> = {}) {
    // Default configuration (from your provided code)
    this.config = {
      apiKey: config.apiKey || process.env.OPENAI_API_KEY || "",
      classificationModel: config.classificationModel || "gpt-4o-mini",
      embeddingModel: config.embeddingModel || "text-embedding-3-small",
      embeddingEndpoint: config.embeddingEndpoint,
      semanticShiftThreshold: config.semanticShiftThreshold || 0.22,
      minTokensPerChunk: config.minTokensPerChunk || 250,
      maxTokensPerChunk: config.maxTokensPerChunk || 2000,
      concurrency: (config.concurrency ?? config?.endpoint) ? 1 : 4,
      embeddingConcurrency: config.embeddingConcurrency ?? 5,
      logger: config?.logger ?? undefined,
      embeddingCacheRamLimitMB: config.embeddingCacheRamLimitMB || 256,
      maxRecursionDepth: config.maxRecursionDepth || 3,
      onlySignificantTurningPoints: config.onlySignificantTurningPoints ?? true,
      significanceThreshold: config.significanceThreshold || 0.0,
      minMessagesPerChunk: config.minMessagesPerChunk || 3,
      maxTurningPoints: config.maxTurningPoints || 5,
      debug: config.debug || false,
      turningPointCategories:
        config?.turningPointCategories &&
          config?.turningPointCategories.length > 0
          ? config.turningPointCategories
          : turningPointCategories,
      endpoint: config.endpoint,

      temperature: config?.temperature ?? 0.6,
      top_p: config?.top_p ?? 0.95,
      complexitySaturationThreshold:
        config.complexitySaturationThreshold || 4.5,
      measureConvergence: config.measureConvergence ?? true,
      enableExperimentalPhi: config.enableExperimentalPhi ?? false,
      dynamicallyAdjustSemanticShiftThreshold:
        config.dynamicallyAdjustSemanticShiftThreshold ?? false,
      phiMergeThresholdMultiplier: config.phiMergeThresholdMultiplier ?? 0.5,
      overlapThreshold: config.overlapThreshold ?? 0.4,
      enableDynamicComplexitySaturation:
        config.enableDynamicComplexitySaturation ?? false,
      dynamicSaturationTargetPercentile:
        config.dynamicSaturationTargetPercentile ?? 0.15,
      dynamicSaturationMinSamples: config.dynamicSaturationMinSamples ?? 10,
      epistemicThreshold: config.epistemicThreshold ?? 0.01,

      enableCounterfactualAnalysis:
        config.enableCounterfactualAnalysis ?? false, // NEW: Enable counterfactual analysis
    };
    // Initialize counterfactual analyzer if enabled
    if (config.enableCounterfactualAnalysis) {
      this.counterfactualAnalyzer = new CounterfactualAnalyzer();
    }
    this.endpointType = config?.endpoint
      ? config.endpoint.includes("api.openai.com")
        ? "openai"
        : config.endpoint.includes("openrouter.ai")
          ? "openrouter"
          : "unknown"
      : "unknown";

    if (this.config.logger === undefined) {
      fs.ensureDirSync("results");
      this.logger = winston.createLogger({
        level: "info",
        format: winston.format.combine(
          winston.format.timestamp(),
          winston.format.json(),
        ),
        transports: [
          new winston.transports.Console({
            format: winston.format.combine(
              winston.format.colorize(),
              winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
              winston.format.printf(({ timestamp, level, message }) => {
                return `${timestamp} ${level}: ${message}`;
              }),
            ),
          }),
          new winston.transports.File({
            filename: "results/semanticTurningPointDetector.log",
            format: winston.format.json(),
          }),
        ],
      });
    }

    // now validate the turning point categories (that wil simply log warnings), and also after the logging is setup above.
    if (
      config?.turningPointCategories &&
      config?.turningPointCategories.length > 0
    ) {
      this.validateTurningPointCategories(config.turningPointCategories);
    }

    // Initialize OpenAI client
    this.openai = new OpenAI({
      apiKey:
        this.config.apiKey ??
        process.env.LLM_API_KEY ??
        process.env.OPENAI_API_KEY,
      baseURL: this.config.endpoint,
    });

    /**
     * Initialize the embedding cache with the specified RAM limit.
     */

    this.embeddingCache = createEmbeddingCache(
      this.config.embeddingCacheRamLimitMB,
    );

    if (this.config.debug) {
      this.logger.info(
        `[TurningPointDetector] Initialized with config:\n${JSON.stringify(
          {
            ...this.config,
            apiKey: "[REDACTED]",
          },
          null,
          2,
        )}`,
      );

      this.logger.info(
        `[TurningPointDetector] Embedding cache initialized with ${this.embeddingCache.max} max entries (${this.config.embeddingCacheRamLimitMB}MB limit)`,
      );
    }
  }

  public getModelName(): string {
    return this.config.classificationModel;
  }
  /**
   * Calculates a thematic similarity score between two turning points based on their
   * emotional tone, sentiment, and LLM-assigned significance. This serves as the core
   * metric for φ-aware grouping and sectioning.
   *
   * @param tp1 - The first turning point.
   * @param tp2 - The second turning point.
   * @returns A similarity score between 0 and 1.
   */

  private calculateThematicSimilarity(
    tp1: TurningPoint,
    tp2: TurningPoint,
  ): number {
    const weights = { tone: 0.5, sentiment: 0.3, significance: 0.2 };

    const toneSim = circularToneSimilarity(
      tp1.emotionalTone,
      tp2.emotionalTone,
    );
    const sentimentSim = tp1.sentiment === tp2.sentiment ? 1 : 0;

    const sigDiff = Math.abs(tp1.significance - tp2.significance);
    const sigSim = 1 - sigDiff; // assumes significance already 0–1

    return (
      toneSim * weights.tone +
      sentimentSim * weights.sentiment +
      sigSim * weights.significance
    );
  }

  /**
   * Recalculates the significance score of a turning point using phi-awareness.
   * This creates a powerful feedback loop where the emergent phi field enhances
   * the base significance score.
   */
  private recalculateSignificanceWithPhi(
    tp: TurningPoint,
    phi: number,
  ): number {
    // Extract emotion intensity based on emotional tone

    const emotionIntensity: { [key: string]: number } = {
      joyful: 0.9,
      excited: 0.8,
      surprised: 0.9,
      worried: 0.7,
      anxious: 0.7,
      angry: 0.9,
      furious: 0.95,
      skeptical: 0.6,
      disgusted: 0.8,
      sad: 0.8,
      discouraged: 0.7,
      hopeful: 0.4,
      neutral: 0.3,
    };
    const intensity = emotionIntensity[tp.emotionalTone.toLowerCase()] || 0.3;

    // Re-run through Choquet but with phi-aware parameters
    return computeSignificanceWithChoquet(
      {
        // Reuse original significance as structural certainty but amplify with phi
        certainty: Math.min(1.0, tp.significance * (1.0 + (phi - 0.5) * 0.6)),

        // Use complexity as novelty credibility (normalized to 0-1)
        novelty: Math.min(1.0, tp.complexityScore / 5),

        // Amplify affective delta based on phi
        affectiveDelta: Math.min(1.0, intensity * (1.0 + (phi - 0.5) * 0.8)),

        // Scale semantic shift magnitude to 0-10 range
        impact: Math.min(10.0, tp.semanticShiftMagnitude * 10),
      },
      tp.emotionalTone,
      {
        enableExperimentalPhi: true,
        phiScore: phi,
        dimension: tp.detectionLevel,
        averageDistance: tp.semanticShiftMagnitude,
      },
    );
  }
  /**
   * Computes the φ (Significance) field by interpreting LLM-derived emotional and
   * significance data from each turning point. This creates a rich, self-referential
   * measure of thematic importance.
   */
  private computePhiSignificanceField(
    turningPoints: TurningPoint[],
  ): Map<string, number> {
    const phiMap = new Map<string, number>();
    if (turningPoints.length === 0) return phiMap;

    // Map emotional tones to intensity scores (0  // Update emotion mapping to match the new wheel of emotions
    const emotionalIntensity: { [key: string]: number } = {
      // High intensity
      furious: 0.95,
      angry: 0.9,
      disgusted: 0.9,
      surprised: 0.8,

      // Medium intensity
      anxious: 0.7,
      worried: 0.7,
      sad: 0.7,
      discouraged: 0.7,

      // Low intensity
      joyful: 0.6,
      excited: 0.6,
      hopeful: 0.4,
      skeptical: 0.5,

      // Default/Neutral
      neutral: 0.1,
      unknown: 0.1,
    };

    for (const tp of turningPoints) {
      // Normalize the 0-100 significance score from the LLM to a 0-1 scale only if it seems like signficance is from 0-100, or if 0-10, accordingly

      // assess if signifance needs to diviced by 100
      const isSignfianceFromZeroToHundred =
        tp.significance >= 0 &&
        tp.significance <= 100 &&
        Number.isInteger(tp.significance);

      const isSignficanceCorrectlyScaledAlready =
        tp.significance >= 0 && tp.significance <= 1;

      const normSignificance = isSignficanceCorrectlyScaledAlready
        ? tp.significance
        : isSignfianceFromZeroToHundred
          ? (tp.significance || 0) / 100
          : (tp.significance || 0) / 10;
      // Get the intensity from the emotional tone, defaulting to a low value
      const toneIntensity =
        emotionalIntensity[tp.emotionalTone.toLowerCase()] || 0.1;

      // Sentiment can provide a small boost for stronger emotions
      const sentimentModifier = tp.sentiment === "negative" ? 1.1 : 1.0;

      // --- The Phi Calculation ---
      // This weighted formula prioritizes the LLM's direct significance assessment,
      // but amplifies it with emotional intensity.
      const phi =
        normSignificance * 0.7 + toneIntensity * sentimentModifier * 0.3;

      // Clamp the final score to ensure it's within the [0, 1] range
      phiMap.set(tp.id, Math.max(0, Math.min(1, phi)));
    }

    return phiMap;
  }

  /**
   * Main entry point: Detect turning points in a conversation
   * Implements the full ARC/CRA framework
   */
  public async detectTurningPoints(messages: Message[]): Promise<{
    confidence: number;
    necessity: number; // The new Choquet-based score
    points: TurningPoint[];
  }> {
    this.logger.info(
      "Starting turning-point detection (ARC/CRA) on with provided " +
      messages.length +
      " messages",
    );

    // log the key config aspects, enableExperimentalPhi, endpoint, and maxTurningPoints, significanceThreshold, semanticShiftThreshold, minTokensPerChunk, maxTokensPerChunk, classificationModel, embeddingModel, endpointType
    this.logger.info(` Turning Point Detection Configuration:
      dynamicallyAdjustSemanticShiftThreshold: ${this.config.dynamicallyAdjustSemanticShiftThreshold},
      dynamicallyAdjustComplexitySaturation: ${this.config.enableDynamicComplexitySaturation},
      
      enableExperimentalPhi: ${this.config.enableExperimentalPhi},
      endpoint: ${this.config.endpoint},
      maxTurningPoints: ${this.config.maxTurningPoints},    
      significanceThreshold: ${this.config.significanceThreshold},
      semanticShiftThreshold: ${this.config.semanticShiftThreshold},
      minTokensPerChunk: ${this.config.minTokensPerChunk},
      maxTokensPerChunk: ${this.config.maxTokensPerChunk},
      classificationModel: ${this.config.classificationModel},    
      `);

    this.convergenceHistory = [];

    const isEndpointOllamaBased = await this.isOllamaEndpoint(
      this.config.endpoint,
    );

    if (isEndpointOllamaBased) {
      this.endpointType = "ollama";
      const url = new URL(this.config.endpoint);
      const host = `${url.protocol}//${url.hostname}${url.port ? `:${url.port}` : ""}`;
      this.logger.info(
        `Detected Ollama endpoint: ${host}. Initializing Ollama client.`,
      );
      this.ollama = new Ollama({ host });
    }

    // ── cache original conversation for downstream helpers
    const totalTokens = await this.getMessageArrayTokenCount(messages);
    this.logger.info(`Total conversation tokens: ${totalTokens}`);
    this.originalMessages = messages.map((m) => ({ ...m }));

    // ── 1️⃣  full multi-layer detection (dim-0 entry)
    const turningPointsFound = await this.multiLayerDetection(messages, 0);
    this.logger.info(
      `Multi-layer detection returned ${turningPointsFound?.length} turning points`,
    );
    const phiScoresByPoint =
      this.computePhiSignificanceField(turningPointsFound);

    // ── 2️⃣  compute per-TP confidence (softmax) and necessity (Choquet) scores
    const confidenceScoresByPoint: number[] = [];
    const necessityScoresByPoint: number[] = [];

    // Helper to collapse per-message embeddings into a single mean vector
    const meanEmbedding = (embs: MessageEmbedding[]): Float32Array => {
      // determine the ongoing length from a valid embedding
      const embeddingDimension = embs.find((emb) => emb.embedding.length > 0)
        ?.embedding.length;
      if (embeddingDimension === undefined || embeddingDimension <= 0) {
        this.logger.warn("No valid embeddings found, returning empty vector");
        return new Float32Array();
      }

      if (embs.length === 0) return new Float32Array(embeddingDimension);
      const dim = embs[0].embedding.length;
      const softMax = (values: number[]): number[] => {
        const maxVal = Math.max(...values);
        const exps = values.map((v) => Math.exp(v - maxVal));
        const sumExps = exps.reduce((sum, v) => sum + v, 0);
        return exps.map((v) => v / sumExps);
      };
      const magnitudes = embs.map(({ embedding }) =>
        Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0)),
      );
      const attnWeights = softMax(magnitudes);
      const acc = new Float32Array(dim);
      for (let idx = 0; idx < embs.length; idx++) {
        const { embedding } = embs[idx];
        const weight = attnWeights[idx];
        for (let i = 0; i < dim; i++) {
          acc[i] += embedding[i] * weight;
        }
      }
      return acc;
    };

    const calculateStructuralNecessity = async (
      tp: TurningPoint,
      allTPs: TurningPoint[],
      conversationEmbeddings: MessageEmbedding[],
      phi?: number,
    ): Promise<number> => {
      if (!this.config.enableExperimentalPhi || phi === undefined) {
        return tp.significance * 0.6;
      }

      // Simple, elegant structural calculation
      const tpEmbedding = meanEmbedding(
        conversationEmbeddings.slice(tp.span.startIndex, tp.span.endIndex + 1),
      );
      const conversationCenterEmbedding = meanEmbedding(conversationEmbeddings);
      const centralityScore =
        1 -
        this.calculateSemanticDistance(
          tpEmbedding,
          conversationCenterEmbedding,
        );

      const relativePosition = tp.span.startIndex / messages.length;
      const positionWeight = 1 - Math.abs(relativePosition - 0.5) * 1.5;

      let uniquenessScore = 1.0;
      if (allTPs.length > 1) {
        const otherTPs = allTPs.filter((other) => other.id !== tp.id);
        const thematicSimilarities = otherTPs.map((other) =>
          this.calculateThematicSimilarity(tp, other),
        );
        const avgSimilarity =
          thematicSimilarities.reduce((sum, sim) => sum + sim, 0) /
          thematicSimilarities.length;
        uniquenessScore = 1 - avgSimilarity;
      }

      const spanCoverage =
        (tp.span.endIndex - tp.span.startIndex + 1) / messages.length;
      const coverageWeight = Math.min(1.0, spanCoverage * 3);

      // Fixed, interpretable weights
      const structuralComponents = [
        centralityScore * 0.3,
        positionWeight * 0.25,
        uniquenessScore * 0.25,
        coverageWeight * 0.2,
      ];

      const baseStructural = structuralComponents.reduce(
        (sum, comp) => sum + comp,
        0,
      );
      const necessity = baseStructural * phi; // Direct phi amplification

      return Math.min(1.0, Math.max(0.0, necessity));
    };

    await async.eachOfLimit(
      turningPointsFound,
      this.config.concurrency,
      async (tp, idxStr) => {
        const pre = messages.slice(0, tp.span.startIndex);
        const turn = messages.slice(tp.span.startIndex, tp.span.endIndex + 1);
        const post = messages.slice(tp.span.endIndex + 1);

        if (pre.length === 0 || post.length === 0) {
          this.logger.info(`TP ${tp.id} at edges of convo – skipping scores`);
          return;
        }

        const [preE, turnE, postE] = await Promise.all([
          this.generateMessageEmbeddings(pre, 0),
          this.generateMessageEmbeddings(turn, 0),
          this.generateMessageEmbeddings(post, 0),
        ]);

        const vPre = meanEmbedding(preE);
        const vTurn = meanEmbedding(turnE);
        const vPost = meanEmbedding(postE);

        const distPre = this.calculateSemanticDistance(vPre, vTurn);
        const distPost = this.calculateSemanticDistance(vTurn, vPost);

        // Calculate Softmax-based "Confidence" (unchanged - this is appropriate)
        const confidence = (distPre + distPost) / 2;
        confidenceScoresByPoint.push(confidence);

        // Calculate TRUE φ-aware "Necessity" using structural analysis
        const phi = this.config.enableExperimentalPhi
          ? phiScoresByPoint.get(tp.id)
          : undefined;
        const allConversationEmbeddings = await this.generateMessageEmbeddings(
          messages,
          0,
        );

        const necessity = await calculateStructuralNecessity(
          tp,
          turningPointsFound,
          allConversationEmbeddings,
          phi,
        );

        necessityScoresByPoint.push(necessity);

        this.logger.info(
          `TP ${tp.id}: distPre=${distPre.toFixed(3)}, distPost=${distPost.toFixed(3)}, conf=${confidence.toFixed(3)}, necessity=${necessity.toFixed(3)}${phi ? `, φ=${phi.toFixed(3)}` : " (φ-disabled)"}`,
        );
      },
    );

    // ── 3️⃣  Aggregate conversation-level scores
    const validConf = confidenceScoresByPoint.filter((v) => v > 0);
    const aggregateConfidence =
      validConf.length === 0
        ? 0
        : validConf.reduce((s, v) => s + v, 0) / validConf.length;

    const validNec = necessityScoresByPoint.filter((v) => v > 0);
    const aggregateNecessity =
      validNec.length === 0
        ? 0
        : validNec.reduce((s, v) => s + v, 0) / validNec.length;

    this.logger.info(
      `Aggregate scores: Confidence=${aggregateConfidence.toFixed(3)}, Necessity=${aggregateNecessity.toFixed(3)}, enabledExperimentalPhi=${this.config.enableExperimentalPhi}`,
    );

    return {
      confidence: aggregateConfidence,
      necessity: this.config.enableExperimentalPhi ? aggregateNecessity : null, // If φ is disabled, necessity is not applicable, as we require the ideation of some metric involving essentiality or
      points: turningPointsFound,
    };
  }

  /**
   * Multi-layer detection implementing the ARC/CRA dimensional processing
   * This is the primary implementation of the transition operator Ψ
   */
  private async multiLayerDetection(
    messages: Message[],
    dimension: number,
  ): Promise<TurningPoint[]> {
    this.logger.info(`Starting dimensional analysis at n=${dimension}`);

    // Check recursion depth - hard limit on dimensional expansion
    if (dimension >= this.config.maxRecursionDepth) {
      this.logger.info(
        `Maximum dimension (n=${dimension}) reached, processing directly without further expansion`,
      );
      // Pass originalMessages context only at dimension 0 if needed by detectTurningPointsInChunk->classifyTurningPoint
      return await this.detectTurningPointsInChunk(
        messages,
        dimension,
        0,
        this.originalMessages,
      );
    }

    // For very small conversations (or at deeper levels), use sliding window
    let localTurningPoints: TurningPoint[] = [];

    // Adjusted condition to handle small message counts more directly
    if (
      messages.length < this.config.minMessagesPerChunk * 2 &&
      dimension === 0
    ) {
      this.logger.info(
        `Dimension ${dimension}: Small conversation (${messages.length} msgs), processing directly`,
      );
      // Optionally adjust threshold for small conversations
      const originalThreshold = this.config.semanticShiftThreshold;
      this.config.semanticShiftThreshold = Math.max(
        0.3,
        originalThreshold * 1.1,
      ); // Slightly higher threshold

      localTurningPoints = await this.detectTurningPointsInChunk(
        messages,
        dimension,
        0,
        this.originalMessages,
      );

      // Restore config
      this.config.semanticShiftThreshold = originalThreshold;
    } else {
      // Chunk the conversation
      const { chunks } = await this.chunkConversation(messages, dimension);
      this.logger.info(
        `Dimension ${dimension}: Split into ${chunks.length} chunks`,
      );

      if (chunks.length === 0) {
        this.logger.info(
          `Dimension ${dimension}: No valid chunks created, returning empty.`,
        );
        return [];
      }

      // Process each chunk in parallel to find local turning points
      const chunkTurningPoints: TurningPoint[][] = new Array(chunks.length);
      const durationsSeconds: number[] = new Array(chunks.length).fill(-1);
      const limit = this.config.concurrency;

      await async.eachOfLimit(chunks, limit, async (chunk, indexStr) => {
        const index = Number(indexStr);
        const startTime = Date.now();

        if (index % 10 === 0 || limit < 10 || this.config.debug) {
          this.logger.info(
            ` - Dimension ${dimension}: Processing chunk ${index + 1}/${chunks.length} (${chunk.length} messages)`,
          );
        }

        // Pass originalMessages context only at dimension 0
        chunkTurningPoints[index] = await this.detectTurningPointsInChunk(
          chunk,
          dimension,
          index,
          this.originalMessages,
        );
        const durationSecs = (Date.now() - startTime) / 1000;
        durationsSeconds[index] = durationSecs;

        if (index % 10 === 0 || limit < 10 || this.config.debug) {
          const processedCount = durationsSeconds.filter((d) => d > 0).length;
          if (processedCount > 0) {
            const averageDuration =
              durationsSeconds.filter((d) => d > 0).reduce((a, b) => a + b, 0) /
              processedCount;
            const remainingChunks = durationsSeconds.length - processedCount;
            const remainingTime = (averageDuration * remainingChunks).toFixed(
              1,
            );
            const percentageComplete =
              (processedCount / durationsSeconds.length) * 100;
            this.logger.info(
              `    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s. Est. remaining: ${remainingTime}s (${percentageComplete.toFixed(1)}% complete)`,
            );
          } else {
            this.logger.info(
              `    - Chunk ${index + 1} processed in ${durationSecs.toFixed(1)}s.`,
            );
          }
        }
      });

      // Flatten all turning points from all chunks
      localTurningPoints = chunkTurningPoints.flat();
    }

    this.logger.info(
      `Dimension ${dimension}: Found ${localTurningPoints.length} raw turning points`,
    );

    // --- PHI-AWARE ARC STEP 1: Calculate initial Phi Field ---
    // Calculate phi based on the raw, unmerged turning points. This will guide the merging process itself.
    const initialPhiMap = this.config.enableExperimentalPhi
      ? this.computePhiSignificanceField(localTurningPoints)
      : new Map<string, number>();

    // If we found zero or one turning point at this level, return it directly (after potential filtering if needed)
    if (localTurningPoints.length <= 1) {
      // --- REVISED LOGIC ---
      // Even in this early exit, we must compute the phiMap to ensure the
      // final filtering step uses the correct, potentially φ-aware, ranking logic.
      const phiMapForFilter = this.config.enableExperimentalPhi
        ? initialPhiMap
        : new Map();

      // Apply filtering even for single points, now with the correct phiMap context.
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(
          localTurningPoints,
          phiMapForFilter,
        )
        : localTurningPoints;
    }

    // First merge any similar turning points at this level
    const mergedLocalTurningPoints = this.mergeSimilarTurningPoints(
      localTurningPoints,
      this.config.enableExperimentalPhi
        ? initialPhiMap
        : new Map<string, number>(),
    );

    this.logger.info(
      `Dimension ${dimension}: Merged similar TPs to ${mergedLocalTurningPoints.length}`,
    );

    // If merging resulted in 0 or 1 TP, return it (after filtering)
    if (mergedLocalTurningPoints.length <= 1) {
      return this.config.onlySignificantTurningPoints
        ? this.filterSignificantTurningPoints(
          mergedLocalTurningPoints,
          initialPhiMap,
        ) // Use initialPhiMap as it's relevant to this set of points
        : mergedLocalTurningPoints;
    }

    // --- CRITICAL ARC/CRA + PHI INTEGRATION ---

    // 1. Re-compute the Significance Field (φ) for the now-merged turning points.
    // This provides a more stable phi for the escalation decision.
    const phiMap = this.config.enableExperimentalPhi
      ? this.computePhiSignificanceField(mergedLocalTurningPoints)
      : new Map<string, number>();

    // 2. If phi is active, update complexity scores to be φ-aware.
    if (this.config.enableExperimentalPhi) {
      this.logger.info(
        `Dimension ${dimension}: Enhancing significance scores with phi-field influence.`,
      );

      // Update both complexity AND significance scores to be φ-aware
      for (const tp of mergedLocalTurningPoints) {
        if (phiMap.has(tp.id)) {
          // Update complexity score (already implemented)
          tp.complexityScore = this.calculateComplexityScoreWithPhi(
            tp,
            phiMap.get(tp.id)!,
          );

          // NEW: Update significance score with phi-awareness
          tp.significance = this.recalculateSignificanceWithPhi(
            tp,
            phiMap.get(tp.id)!,
          );

          // Store phi on the turning point for reference
          tp.phi = phiMap.get(tp.id)!;
        }
      }
    }

    // 3. Determine dimensional escalation based on the (now potentially φ-aware) complexity.
    const effectiveThreshold = this.calculateDynamicComplexitySaturation(
      mergedLocalTurningPoints,
    );
    // Update the config for this decision (but don't modify the original)

    const maxComplexity = Math.max(
      0,
      ...mergedLocalTurningPoints.map((tp) => tp.complexityScore),
    );
    // const needsDimensionalEscalation = maxComplexity >= this.config.complexitySaturationThreshold;
    const needsDimensionalEscalation = maxComplexity >= effectiveThreshold; // Use local var

    this.logger.info(
      `Dimension ${dimension}: Max complexity = ${maxComplexity.toFixed(2)}, Saturation threshold = ${this.config.complexitySaturationThreshold}`,
    );
    this.logger.info(
      `Dimension ${dimension}: Needs Escalation (Ψ)? ${needsDimensionalEscalation}`,
    );

    if (
      dimension >= this.config.maxRecursionDepth - 1 ||
      mergedLocalTurningPoints.length <= 2 ||
      !needsDimensionalEscalation
    ) {
      this.logger.info(
        `Dimension ${dimension}: Finalizing at this level. Applying final filtering.`,
      );
      // Track convergence for this dimension
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: [], // No previous state at the final level of processing
          currentTurningPoints: mergedLocalTurningPoints, // TPs before final filtering
          dimension,
          distanceMeasure: 0, // No comparison needed at final step
          hasConverged: true, // Considered converged as processing stops here
          didEscalate: false,
        });
      }
      // Filter the merged points before returning
      return this.filterSignificantTurningPoints(
        mergedLocalTurningPoints,
        phiMap,
      );
    }

    // ----- DIMENSIONAL ESCALATION (n → n+1) -----
    this.logger.info(
      `Dimension ${dimension}: Escalating to dimension ${dimension + 1}`,
    );

    // Create meta-messages from the merged turning points at this level
    // Pass originalMessages for context if needed by createMetaMessagesFromTurningPoints
    const metaMessages = this.createMetaMessagesFromTurningPoints(
      mergedLocalTurningPoints,
      this.originalMessages,
    );
    this.logger.info(
      `Dimension ${dimension}: Created ${metaMessages.length} meta-messages for dimension ${dimension + 1}`,
    );

    if (metaMessages.length < 2) {
      this.logger.info(
        `Dimension ${dimension}: Not enough meta-messages (${metaMessages.length}) to perform higher-level analysis. Finalizing with current TPs.`,
      );
      if (this.config.measureConvergence) {
        this.convergenceHistory.push({
          previousTurningPoints: mergedLocalTurningPoints, // State before attempted escalation
          currentTurningPoints: mergedLocalTurningPoints, // State after failed escalation
          dimension: dimension + 1, // Represents the attempted next dimension
          distanceMeasure: 0, // No change
          hasConverged: true, // Converged because escalation failed
          didEscalate: false, // Escalation attempted but yielded no processable result
        });
      }
      return this.filterSignificantTurningPoints(
        mergedLocalTurningPoints,
        this.config.enableExperimentalPhi ? phiMap : new Map<string, number>(),
      );
    }

    // Recursively process the meta-messages to find higher-dimensional turning points
    const higherDimensionTurningPoints = await this.multiLayerDetection(
      metaMessages,
      dimension + 1,
    );
    this.logger.info(
      `Dimension ${dimension + 1}: Found ${higherDimensionTurningPoints.length} higher-dimension TPs.`,
    );

    // Track convergence and dimension escalation
    if (this.config.measureConvergence) {
      const convergenceState: ConvergenceState = {
        previousTurningPoints: mergedLocalTurningPoints, // TPs from dim n
        currentTurningPoints: higherDimensionTurningPoints, // TPs found in dim n+1
        dimension: dimension + 1,
        distanceMeasure: this.calculateStateDifference(
          mergedLocalTurningPoints,
          higherDimensionTurningPoints,
          phiMap, // Pass the phi map, only used if `enableExperimentalPhi` is true via config
        ),
        hasConverged: higherDimensionTurningPoints.length > 0, // Converged if TPs were found at higher level
        didEscalate: true,
      };
      this.convergenceHistory.push(convergenceState);
      this.logger.info(
        `Dimension ${dimension} → ${dimension + 1}: Convergence distance: ${convergenceState.distanceMeasure.toFixed(3)}. Converged: ${convergenceState.hasConverged}`,
      );
    }

    // Combine turning points from local (n) and higher (n+1) dimensions
    // The combine function will handle merging, prioritizing higher-dim, and filtering
    return this.combineTurningPoints(
      mergedLocalTurningPoints,
      higherDimensionTurningPoints,
      phiMap, // Pass the phi map for context (only used if `enableExperimentalPhi` is true via config)
    );
  }

  /**
   * Calculates a difference measure between two states (sets of turning points) for
   * convergence tracking. When the experimental phi feature is enabled, this metric
   * becomes φ-aware by blending the LLM-assigned significance with the emergent
   * phi score for a more holistic comparison.
   *
   * @param state1 - The first set of turning points.
   * @param state2 - The second set of turning points.
   * @param phiMap - The map of phi scores for turning points in the states.
   * @returns A single numeric value representing the distance between the two states.
   */
  private calculateStateDifference(
    state1: TurningPoint[],
    state2: TurningPoint[],
    phiMap: Map<string, number>, // Pass the phi map for context
  ): number {
    // Handle empty states
    if (state1.length === 0 && state2.length === 0) return 0.0;
    if (state1.length === 0 || state2.length === 0) return 1.0;

    // Helper to calculate the average adjusted significance for a state
    const getAvgAdjustedSig = (state: TurningPoint[]): number => {
      const totalSig = state.reduce((sum, tp) => {
        // If phi is enabled, use a composite score of LLM-significance and emergent-phi
        if (this.config.enableExperimentalPhi && phiMap.has(tp.id)) {
          return sum + (tp.significance + phiMap.get(tp.id)!) / 2;
        }
        // Otherwise, use only the LLM-assigned significance
        return sum + tp.significance;
      }, 0);
      return state.length > 0 ? totalSig / state.length : 0;
    };

    // 1. Calculate the difference in average adjusted significance
    const sigDiff = Math.abs(
      getAvgAdjustedSig(state1) - getAvgAdjustedSig(state2),
    );

    // 2. Calculate structural difference using Jaccard index on the message spans
    const spans1 = new Set(
      state1.map((tp) => `${tp.span.startIndex}-${tp.span.endIndex}`),
    );
    const spans2 = new Set(
      state2.map((tp) => `${tp.span.startIndex}-${tp.span.endIndex}`),
    );
    const intersection = new Set(
      [...spans1].filter((span) => spans2.has(span)),
    );
    const union = new Set([...spans1, ...spans2]);
    const jaccardDistance =
      union.size > 0 ? 1.0 - intersection.size / union.size : 0.0;

    // 3. Return a weighted combination of the two difference measures
    const combinedDistance = sigDiff * 0.5 + jaccardDistance * 0.5;

    return Math.min(1.0, Math.max(0.0, combinedDistance));
  }

  /**
   * Apply complexity function χ from the ARC/CRA framework
   * - Complexity is part of CRA specifically within the ARC/CRA Duality framework
   */
  private calculateComplexityScore(
    significance: number,
    semanticShiftMagnitude: number,
  ): number {
    // Return to the older, simpler approach
    // Complexity should reflect content significance, not distance redundancy
    const complexity = 1 + significance * 4;
    return Math.max(1, Math.min(5, complexity));
  }
  /**
   * @experimental
   * Calculates the complexity score for a Turning Point, dynamically modulated by the
   * experimental φ (Significance) field. This function is only called when
   * `config.enableExperimentalPhi` is true.
   *
   * @param tp - The TurningPoint object being scored.
   * @param phi - The calculated φ score (emergent significance) for this turning point.
   * @returns A φ-aware complexity score, clamped between 1 and 5.
   */
  private calculateComplexityScoreWithPhi(
    tp: TurningPoint,
    phi: number,
  ): number {
    const baseComplexity = this.calculateComplexityScore(
      tp.significance,
      tp.semanticShiftMagnitude,
    );

    const phiAdjustment = (phi - 0.5) * 0.6; // Adjust the phi influence factor as needed
    const adjustedComplexity = baseComplexity + phiAdjustment; // REMOVE: * baseComplexity

    return Math.max(1, Math.min(5, adjustedComplexity));
  }

  /**
   * Detect turning points within a single chunk of the conversation
   * This represents the local refinement process in the current dimension
   * - Or in other words, this is the Ψ operator in the ARC/CRA framework
   * - or specifically, within the ARC framework
   */
  private async detectTurningPointsInChunk(
    messages: MetaMessage[] | Message[],
    dimension: number,

    chunkIndex: number, // Optional index for logging purposes
    originalMessages: Message[],
  ): Promise<TurningPoint[]> {
    if (messages.length < 2) return [];

    /**
     * Higher dimensions , given how it is then the exponent value, will cause then the factor to be more aggressive, or in otherwords, the threshold to be lower.
     * This is because the higher the dimension, the more complex the conversation is, and thus the more likely that the semantic shifts are more subtle and nuanced.
     * @param dimension
     * @param baseThreshold
     * @returns
     */
    const dynamicallyAdjustThresholdBasedOnDimension = (
      dimension: number,
      baseThreshold: number,
    ): number => {
      // Defines the decay factor based on the base threshold.
      // The decay rate changes based on the initial sensitivity.
      const decayFactors = [
        { limit: 0.9, factor: 0.4 }, // Very high thresholds decay slower
        { limit: 0.8, factor: 0.25 }, // High thresholds decay aggressively
        { limit: 0.5, factor: 0.35 }, // Medium thresholds
      ];

      // Find the appropriate decay factor, defaulting to 0.5 for low thresholds.
      const decayFactor =
        decayFactors.find((d) => baseThreshold > d.limit)?.factor || 0.5;

      // Apply exponential decay based on the dimension.
      const thresholdScaleFactor = Math.pow(decayFactor, dimension);

      return thresholdScaleFactor * baseThreshold;
    };

    // Generate embeddings for all messages in the chunk
    const embeddings = await this.generateMessageEmbeddings(
      messages,
      dimension,
    );

    // Find significant semantic shifts between adjacent messages
    const turningPoints: TurningPoint[] = [];
    const distances: {
      current: number;
      next: number;
      distance: number;
    }[] = []; // Store distances for logging
    const allDistances: {
      current: number;
      next: number;
      distance: number;
    }[] = []; // Store all distances for logging
    for (let i = 0; i < embeddings.length - 1; i++) {
      const current = embeddings[i];
      const next = embeddings[i + 1];

      // Calculate semantic distance between current and next message

      const distance = this.calculateSemanticDistance(
        current.embedding,
        next.embedding,
      );

      const dimensionAdjustedThreshold =
        this.config.dynamicallyAdjustSemanticShiftThreshold &&
          this.config.dynamicallyAdjustSemanticShiftThreshold === true
          ? dynamicallyAdjustThresholdBasedOnDimension(
            dimension,
            this.config.semanticShiftThreshold,
          )
          : this.config.semanticShiftThreshold;

      this.logger.debug(
        `Anlyzing with dimensionAdjustedThreshold: ${dimensionAdjustedThreshold.toFixed(3)}, compared to original threshold: ${this.config.semanticShiftThreshold.toFixed(3)}, with the difference in embeddings or distance of: ${distance.toFixed(3)}`,
      );
      if (distance > dimensionAdjustedThreshold) {
        distances.push({
          current: current.index,
          next: next.index,
          distance: distance,
        }); // Store distance for logging
        this.logger.debug(
          `  - After analyzing, determined this distance is to be added to the list of distances to process: ${distance.toFixed(3)}`,
        );
      } else {
        this.logger.debug(
          `  - After analyzing, determined this distance is NOT significant enough to be added to the list of distances to process: ${distance.toFixed(3)}, from the difference of the two embeddings: ${current.embedding.length} and ${next.embedding.length}`,
        );
      }
      allDistances.push({
        current: current.index,
        next: next.index,
        distance: distance,
      });
    }

    this.logger.info(
      `For a total number of points: ${embeddings.length}, there were ${distances.length} distances found as being greater ${this.config.dynamicallyAdjustSemanticShiftThreshold &&
        this.config.dynamicallyAdjustSemanticShiftThreshold === true
        ? `than the dynamically adjusted threshold of ${dynamicallyAdjustThresholdBasedOnDimension(dimension, this.config.semanticShiftThreshold).toFixed(3)}`
        : `than the threshold of ${this.config.semanticShiftThreshold.toFixed(3)}`
      }. Across this span of messages of length ${messages.length}, the following distances were found:
    - The top 3 greatest distances are: ${allDistances
        .sort((a, b) => b.distance - a.distance) // Sort FIRST
        .slice(0, 3) // Then take the top 3
        .map((d) => d.distance.toFixed(3))
        .join(", ")}
      
      
      Found ${distances.length} potential turning points at this level (${dimension === 0 ? "base messages" : "meta-messages"}).`

    );
    if (distances.length === 0) {
      this.logger.info(
        `No significant semantic shifts detected in chunk ${chunkIndex}`,
      );
      return [];
    }
    await async.eachOfLimit(
      distances,
      this.config.concurrency,
      async (distanceObj, idxStr) => {
        const d = Number(idxStr);

        const i = distanceObj.current; // Current message index
        const current = embeddings[i]; // Current message embedding
        const next = embeddings[distanceObj.next]; // Next message embedding
        // If the distance exceeds our threshold, we've found a turning point
        // Use direct array indices to get the messages
        const distance = distanceObj.distance; // Semantic distance between current and next message
        const beforeMessage = messages[i];
        const afterMessage = messages[i + 1];
        if (beforeMessage == undefined || afterMessage == undefined) {
          this.logger.info(
            `detectTurningPointsInChunk: warning beforeMessage or afterMessage is undefined, beforeMessage: ${beforeMessage}, afterMessage: ${afterMessage}`,
          );
          return;
        }

        // Classify the turning point using LLM
        const turningPoint = await this.classifyTurningPoint(
          beforeMessage,
          afterMessage,
          distance,
          dimension,
          originalMessages,
          d,
        );

        if (d === 0) {
          this.logger.info(`Now proceeding to process every turning point`);
        }

        this.logger.info(
          `    ...${chunkIndex ? `[Chunk ${chunkIndex}] ` : ""
          }Potential turning point detected between messages ${current.id
          } and ${next.id} (distance: ${distance.toFixed(
            3,
          )}, complexity: ${turningPoint.complexityScore.toFixed(
            1,
          )}), signif: ${turningPoint.significance.toFixed(2)} category: ${turningPoint.category
          }, number of quotes: ${turningPoint.quotes.length}, emotionalTone: ${turningPoint.emotionalTone}`,
        );

        // normaliz

        turningPoints.push(turningPoint);
      },
    );

    return turningPoints;
  }



  /**
   * Use LLM to classify a turning point and generate metadata.
   * *** MODIFIED to prioritize message.spanData over regex ***
   */
  /**
   * Use LLM to classify a turning point and generate metadata.
   * This implementation uses a highly modular prompt architecture with
   * multiple distinct user messages to ensure clarity. The payload consists of:
   * - A system message that sets the core identity and universal constraints.
   * - A static context user message containing framework and evaluation criteria.
   * - A dynamic data user message that provides conversation context and the specific messages to analyze.
   * - A final user instruction message that tells the model what to do with all this information.
   */
  private async classifyTurningPoint(
    beforeMessage: Message,
    afterMessage: Message,
    distance: number,
    dimension: number,
    originalMessages: Message[],
    index: number = 0,
  ): Promise<TurningPoint> {
    let span: MessageSpan;

    if (dimension > 0) {
      if (
        !(beforeMessage instanceof MetaMessage) ||
        !(afterMessage instanceof MetaMessage)
      ) {
        throw new Error(
          "Before or after message is not a MetaMessage at higher dimension",
        );
      }
      const beforeMessageMeta = beforeMessage as MetaMessage;
      const afterMessageMeta = afterMessage as MetaMessage;
      // For higher dimensions, extract the starting and ending message from within the meta-message's inner list
      span = {
        startId:
          beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
            .id,
        endId:
          afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0].id,
        startIndex: this.originalMessages.findIndex(
          (candidateM) =>
            candidateM.id ===
            beforeMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
              .id,
        ),
        endIndex: this.originalMessages.findIndex(
          (candidateM) =>
            candidateM.id ===
            afterMessageMeta.getMessagesInTurningPointSpanToMessagesArray()[0]
              .id,
        ),
        originalSpan: {
          startId: beforeMessage.id,
          endId: afterMessage.id,
          startIndex: index,
          endIndex: index + 1,
        },
      };
    } else {
      // For base-level conversations, use the original message IDs and find their indices.
      span = {
        startId: beforeMessage.id,
        endId: afterMessage.id,
        startIndex: MetaMessage.findIndexOfMessageFromId({
          id: beforeMessage.id,
          beforeMessage,
          afterMessage,
          messages: originalMessages,
        }),
        endIndex: MetaMessage.findIndexOfMessageFromId({
          id: afterMessage.id,
          beforeMessage,
          afterMessage,
          messages: originalMessages,
        }),
      };
    }

    // --- Constructing the Modular Prompt ---

    // 1. System Message: Core identity and immutable instructions.
    const systemMessage =
      this.config.customSystemInstruction &&
        this.config.customSystemInstruction.length > 0
        ? this.config.customSystemInstruction
        : formSystemMessage({
          distance,
          dimension,
        });

    // 3. Dynamic Data User Message: Conversation context and messages to analyze.
    const contextualInfo = this.prepareContextualInfoMeta(
      beforeMessage,
      afterMessage,
      span,
      originalMessages,
      dimension,
      2,
      dimension > 0,
      "modular",
    ) as { beforeAndAfterContext: string; withinSpanContext?: string };

    const tagNameForAnalysisContent = "content_to_analyze";

    const tagNameForBeforeContent =
      dimension > 0 ? "turning_point_before" : "before_message";
    const tagNameForAfterContent =
      dimension > 0 ? "turning_point_after" : "after_message";

    const dynamicDataMessage = `<conversation_context>
${contextualInfo.beforeAndAfterContext}
</conversation_context>`;

    const analysisContent = `<${tagNameForAnalysisContent}>
    <${tagNameForBeforeContent}>
      ${returnFormattedMessageContent(this.config, beforeMessage, dimension)
        .split("\n")
        .map((line) => `      ${line}`)
        .join("\n")
        .trim()}
    </${tagNameForBeforeContent}>
    <${tagNameForAfterContent}>
      ${returnFormattedMessageContent(this.config, afterMessage, dimension)
        .split("\n")
        .map((line) => `      ${line}`)
        .join("\n")
        .trim()}
    </${tagNameForAfterContent}>
</${tagNameForAnalysisContent}>`;

    // 4. Final Task Instruction User Message: Direct instruction to the LLM.
    const customEndMessage =
      formAnalysisSystemPromptEnding(dimension, this.config) +
      (this.config.customUserInstruction ?? "");

    // Assemble all messages as a multi-message payload
    const messagesPayload: OpenAI.ChatCompletionMessageParam[] = [
      {
        role:
          this.config.customSystemInstruction &&
            this.config.customSystemInstruction.length > 0
            ? "user"
            : "system",
        content: systemMessage,
      },
      { role: "user", content: dynamicDataMessage },
      { role: "user", content: analysisContent + "\n\n" + customEndMessage },
    ];

    if (
      this.config.customSystemInstruction &&
      this.config.customSystemInstruction.length > 0
    ) {
      messagesPayload.unshift({
        role: "system",
        content: this.config.customSystemInstruction,
      });
    }
    let classification: Partial<{
      label: string;
      category: string;
      emotionalTone: string;
      sentiment: string;
      quotes: string[];
      best_id: string;
      significanceFactors?: {
        impact: number;
        novelty: number;
        affectiveDelta: number;
        certainty: number;
      };
    }> = {};

    try {
      const parallelRequests = await async.parallel<
        unknown,
        { classification: string | null; significance: string | null }
      >({
        classification: async () => {
          if (this.endpointType !== "ollama") {
            const response = await this.openai.chat.completions.create({
              model: this.config.classificationModel,
              messages: messagesPayload,
              temperature: this.config.temperature,
              response_format: formAnalysisResponseFormat(
                dimension,
                this.config,
              ),
              top_p: this.config.top_p,

              //@ts-ignore
              reasoning: {
                enabled: true,
              },
            });

            return response.choices[0]?.message?.content || "{}";

            // write payload and response to json as debug
          } else {
            const response = await this.ollama.chat({
              model: this.config.classificationModel,
              messages: messagesPayload.map((msg) => ({
                role: msg.role,
                content: String(msg.content),
              })),
              stream: false,
              format: formAnalysisResponseFormat(dimension, this.config)
                .json_schema.schema,
              options: {
                temperature: this.config.temperature,
                top_p: this.config.top_p,
                top_k: 20,
                num_keep: 0,
                repeat_penalty: 1.1,
                num_ctx: Math.ceil(
                  this.config.maxTokensPerChunk * (1.5 + dimension / 10),
                ),
              },
            });

            // now try to json parse, if failure do the ame fallback
            return response?.message?.content ?? "{}";
          }
        },

        significance: async () => {
          const messages = [
            {
              role:
                this.config.customSystemInstruction &&
                  this.config.customSystemInstruction.length > 0
                  ? "user"
                  : "system",
              content: systemMessage,
            },
            { role: "user", content: dynamicDataMessage },
            {
              role: "user",
              content:
                analysisContent +
                formScoringSystemPromptEnding(dimension) +
                (this.config.customUserInstruction ?? ""),
            },
          ] as OpenAI.Chat.Completions.ChatCompletionMessageParam[];

          if (
            this.config.customSystemInstruction &&
            this.config.customSystemInstruction.length > 0
          ) {
            messages.unshift({
              role: "system",
              content: this.config.customSystemInstruction,
            });
          }

          if (this.endpointType !== "ollama") {
            const response = await this.openai.chat.completions.create({
              model: this.config.classificationModel,
              messages,
              temperature: this.config.temperature,
              response_format: formScoringResponseFormat(),
              top_p: this.config.top_p,

              //@ts-ignore
              reasoning: {
                enabled: true,
              },
            });
            return response.choices[0]?.message?.content || "{}";
          } else {
            const response = await this.ollama.chat({
              model: this.config.classificationModel,
              messages: messages.map((msg) => ({
                role: msg.role,
                content: String(msg.content),
              })),
              stream: false,
              format: formScoringResponseFormat().json_schema.schema,
              options: {
                temperature: this.config.temperature,
                top_p: this.config.top_p,
                top_k: 15,
                num_ctx: Math.ceil(
                  this.config.maxTokensPerChunk * (1.5 + dimension / 10),
                ),
              },
            });
            return response?.message?.content ?? "{}";
          }
        },
      });

      if (parallelRequests?.classification && parallelRequests?.significance) {
        classification = this.parseClassificationResponse(
          parallelRequests.classification,
          span,
        );
        const significances = this.parseClassificationResponse(
          parallelRequests.significance,
          span,
        );

        classification.quotes =
          significances.quotes || classification.quotes || [];

        classification.emotionalTone =
          significances.emotionalTone ||
          classification.emotionalTone ||
          "neutral";
        classification.significanceFactors = {
          novelty: significances.novelty || 0.0,
          impact: significances.impact || 0.0,
          affectiveDelta: significances.affectiveDelta || 0.0,
          certainty: significances.certainty || 0.0,
        };
      } else {
        // Fallback if no response content
        classification = {
          label: "No Response - Unclassified",
          category: "Other",
          emotionalTone: "neutral",
          sentiment: "neutral",
          significanceFactors: {
            novelty: 0.0,
            impact: 0.0,
            affectiveDelta: 0.0,
            certainty: 0.0,
          },
          quotes: [],
          best_id: span.startId,
        };
      }

      // Validate and sanitize the LLM output.
      const validatedClassification = {
        label:
          typeof classification.label === "string"
            ? classification.label.substring(0, 50)
            : "Unknown Turning Point",
        category:
          typeof classification.category === "string"
            ? classification.category
            : "Other",

        emotionalTone:
          typeof classification.emotionalTone === "string"
            ? classification.emotionalTone
            : "neutral",
        sentiment: ["positive", "negative", "neutral"].includes(
          classification.sentiment,
        )
          ? classification.sentiment
          : "neutral",
        significance: this.config.enableExperimentalPhi
          ? computeSignificanceWithChoquet(
            classification.significanceFactors,
            classification.emotionalTone,
          )
          : computeSignificance(
            classification.significanceFactors,
            classification.emotionalTone,
          ),
        quotes: Array.isArray(classification.quotes)
          ? classification.quotes.map(String).slice(0, 3)
          : [],
        best_id:
          typeof classification.best_id === "string"
            ? classification.best_id
            : span.startId,
      };

      this.logger.debug(
        `Validated classification: ${JSON.stringify(validatedClassification, null, 2)}`,
      );

      // Calculate complexity score using the significance and the raw distance.
      let complexityScore: number;

      if (this.config.enableExperimentalPhi) {
        // Calculate initial phi for this turning point
        const tempTp = {
          id: `temp-${dimension}-${span.startIndex}-${span.endIndex}`,
          significance: validatedClassification.significance,
          emotionalTone: validatedClassification.emotionalTone,
          sentiment: validatedClassification.sentiment,
        } as TurningPoint;

        const tempPhiMap = this.computePhiSignificanceField([tempTp]);
        const phi = tempPhiMap.get(tempTp.id) || 0.5;

        complexityScore = this.calculateComplexityScoreWithPhi(tempTp, phi);
      } else {
        complexityScore = this.calculateComplexityScore(
          validatedClassification.significance,
          distance,
        );
      }

      const turningPointMessages = originalMessages.slice(
        span.startIndex,
        span.endIndex + 1,
      );
      const turningPointEmbeddings = await this.generateMessageEmbeddings(
        turningPointMessages,
        dimension,
      );
      const meanEmbedding = (embs: MessageEmbedding[]): Float32Array => {
        if (embs.length === 0) return new Float32Array();
        const dim = embs[0].embedding.length;
        const acc = new Float32Array(dim);
        for (const { embedding } of embs) {
          for (let i = 0; i < dim; i++) {
            acc[i] += embedding[i];
          }
        }
        for (let i = 0; i < dim; i++) {
          acc[i] /= embs.length;
        }
        return acc;
      };
      const turningPointEmbedding = meanEmbedding(turningPointEmbeddings);

      // Construct and return the final TurningPoint object.
      return {
        id: `tp-${dimension}-${span.startIndex}-${span.endIndex}`,
        label: validatedClassification.label,
        category: validatedClassification.category,
        span: span,
        semanticShiftMagnitude: distance,
        quotes: validatedClassification.quotes,
        emotionalTone: validatedClassification.emotionalTone,
        sentiment: validatedClassification.sentiment,
        detectionLevel: dimension,
        significance: validatedClassification.significance,
        complexityScore: complexityScore,
        embedding: turningPointEmbedding,
      };
    } catch (err: any) {
      this.logger.info(
        `Error during LLM call for turning point classification: ${err.message}`,
      );
      if (this.config.throwOnError) {
        throw err;
      } else {
        return {
          id: `tp-err-${dimension}-${span.startId}`,
          label: "LLM Error - Unclassified",
          category: "Other",
          span: span,
          semanticShiftMagnitude: distance,
          quotes: [],
          emotionalTone: "neutral",
          sentiment: "neutral",
          detectionLevel: dimension,
          significance: 0.1,
          complexityScore: 1.0,
        };
      }
    }
  }

  /**
   * Creates meta-messages from turning points to escalate to the next dimension (CRA process).
   * This definitive version generates meta-messages from three independent perspectives,
   * making both the categorical and chronological groupings φ-aware by using a thematic
   * similarity score derived from emotion, sentiment, and significance.
   */
  private createMetaMessagesFromTurningPoints(
    turningPoints: TurningPoint[],
    originalMessages: Message[],
  ): Message[] {
    if (turningPoints.length === 0) return [];

    const metaMessages: Message[] = [];
    const createdMessageLabels = new Set<string>();

    // --- 1. ALWAYS CREATE CATEGORY-BASED META-MESSAGES (Structural View) ---
    const groupedByCategory: Record<string, TurningPoint[]> = {};
    turningPoints.forEach((tp) => {
      const category = tp.category;
      if (!groupedByCategory[category]) {
        groupedByCategory[category] = [];
      }
      groupedByCategory[category].push(tp);
    });

    this.logger.info(
      `Grouped turning points into ${Object.keys(groupedByCategory).length} categories.`,
    );

    Object.entries(groupedByCategory).forEach(([category, points], index) => {
      if (!createdMessageLabels.has(category)) {
        const metaMessage = MetaMessage.createCategoryMetaMessage(
          category,
          points,
          index,
          originalMessages,
        );
        metaMessages.push(metaMessage);
        createdMessageLabels.add(category);
      }
    });

    // --- 2. IF PHI IS ENABLED, ADD THEMATIC & THEMATIC-TEMPORAL VIEWS ---
    if (this.config.enableExperimentalPhi && turningPoints.length > 1) {
      this.logger.info("Applying φ-aware thematic and chronological merging.");

      // --- 2a. Thematic Clustering (Significance View) ---
      const thematicGroups: TurningPoint[][] = [];
      const assignedToTheme = new Set<string>();
      const thematicSimilarityThreshold = 0.6; // Stricter threshold for thematic grouping
      const sortedBySignificance = [...turningPoints].sort(
        (a, b) => b.significance - a.significance,
      );

      for (const tp1 of sortedBySignificance) {
        if (assignedToTheme.has(tp1.id)) continue;
        const currentGroup = [tp1];
        assignedToTheme.add(tp1.id);

        for (const tp2 of sortedBySignificance) {
          if (assignedToTheme.has(tp2.id)) continue;
          if (
            this.calculateThematicSimilarity(tp1, tp2) >=
            thematicSimilarityThreshold
          ) {
            currentGroup.push(tp2);
            assignedToTheme.add(tp2.id);
          }
        }
        thematicGroups.push(currentGroup);
      }

      thematicGroups.forEach((group, index) => {
        if (group.length > 1) {
          const thematicLabel = `Theme: ${group[0].emotionalTone} - ${group[0].label}`;
          if (!createdMessageLabels.has(thematicLabel)) {
            const metaMessage = MetaMessage.createCategoryMetaMessage(
              thematicLabel,
              group,
              100 + index,
              originalMessages,
            );
            metaMessages.push(metaMessage);
            createdMessageLabels.add(thematicLabel);
          }
        }
      });

      // --- 2b. Thematic-Temporal Sectioning (Temporal View) ---
      const sortedByTime = [...turningPoints].sort(
        (a, b) => a.span.startIndex - b.span.startIndex,
      );
      const temporalSections: TurningPoint[][] = [];
      let currentSection = [sortedByTime[0]];
      const temporalBreakThreshold = 0.4; // Looser threshold to find breaks in the flow

      for (let i = 1; i < sortedByTime.length; i++) {
        const prevTp = sortedByTime[i - 1];
        const currentTp = sortedByTime[i];
        if (
          this.calculateThematicSimilarity(prevTp, currentTp) <
          temporalBreakThreshold
        ) {
          temporalSections.push(currentSection);
          currentSection = [];
        }
        currentSection.push(currentTp);
      }
      temporalSections.push(currentSection);

      this.logger.info(
        `Segmented the timeline into ${temporalSections.length} thematic sections.`,
      );

      temporalSections.forEach((section, i) => {
        if (section.length > 0) {
          const sectionLabel = `Phase ${i + 1}: ${section[0].label}`;
          if (!createdMessageLabels.has(sectionLabel)) {
            const sectionMetaMessage = MetaMessage.createSectionMetaMessage(
              section,
              i,
              this.originalMessages,
            );
            metaMessages.push(sectionMetaMessage);
            createdMessageLabels.add(sectionLabel);
          }
        }
      });
    } else {
      // --- Fallback to original, non-φ-aware chronological sectioning ---
      const sortedPoints = [...turningPoints].sort(
        (a, b) => a.span.startIndex - b.span.startIndex,
      );
      const sectionCount = Math.min(4, Math.ceil(sortedPoints.length / 2));
      const pointsPerSection = Math.ceil(sortedPoints.length / sectionCount);

      for (let i = 0; i < sectionCount; i++) {
        const sectionPoints = sortedPoints.slice(
          i * pointsPerSection,
          (i + 1) * pointsPerSection,
        );
        if (sectionPoints.length > 0) {
          const sectionLabel = `Section ${i + 1}`;
          if (!createdMessageLabels.has(sectionLabel)) {
            const sectionMetaMessage = MetaMessage.createSectionMetaMessage(
              sectionPoints,
              i,
              this.originalMessages,
            );
            metaMessages.push(sectionMetaMessage);
            createdMessageLabels.add(sectionLabel);
          }
        }
      }
    }

    this.logger.info(
      `Created a total of ${metaMessages.length} meta-messages for dimensional expansion.`,
    );
    // FIX: Sort meta-messages chronologically before returning them for the next dimension.
    // This prevents the creation of invalid, reversed spans in higher dimensions.
    // Note: This assumes MetaMessage has a `getSpan()` method that returns its span.
    return metaMessages.sort((a, b) => {
      const spanA = (a as MetaMessage).spanData;
      const spanB = (b as MetaMessage).spanData;
      return spanA.startIndex - spanB.startIndex;
    });
  }

  /**
   * Filters the detected turning points to retain only those deemed significant.
   * (Implements the original logic from the second code block.)
   *
   * @param turningPoints - An array of detected turning points to be filtered.
   * @param phiMap - (Experimental) A map containing phi scores for each turning point, used when `enableExperimentalPhi` is true. @experimental
   */
  private filterSignificantTurningPoints(
    turningPoints: TurningPoint[],
    phiMap: Map<string, number> = new Map(),
  ): TurningPoint[] {
    if (
      !this.config.onlySignificantTurningPoints ||
      turningPoints.length === 0
    ) {
      return turningPoints.sort(
        (a, b) => a.span.startIndex - b.span.startIndex,
      );
    }

    // --- Possibilistic Gating & Fusion ---
    // This replaces the simpler significance/overlap filtering with a more robust method.

    // 1. Calculate Epistemic Primitives for all TPs
    const allEmbeddings = turningPoints
      .map((tp) => tp.embedding)
      .filter(Boolean) as Float32Array[];
    const evidenceDistribution = allEmbeddings.reduce(
      (acc, emb) => {
        emb.forEach((val, i) => (acc[i] = (acc[i] || 0) + val));
        return acc;
      },
      new Float32Array(allEmbeddings[0]?.length || 0),
    );

    if (allEmbeddings.length > 0) {
      evidenceDistribution.forEach(
        (_, i) => (evidenceDistribution[i] /= allEmbeddings.length),
      );
    }

    turningPoints.forEach((tp) => {
      if (tp.embedding) {
        tp.epistemicPrimitives = this.calculateEpistemicPrimitives(
          tp,
          evidenceDistribution,
          turningPoints,
        );
        // Define support score as a combination of compatibility and necessity
        tp.supportScore =
          (tp.epistemicPrimitives.compatibility +
            tp.epistemicPrimitives.necessity) /
          2;
      } else {
        tp.supportScore = tp.significance; // Fallback for TPs without embeddings
      }
    });

    const byDimension = new Map<number, TurningPoint[]>();
    turningPoints.forEach((tp) => {
      if (!byDimension.has(tp.detectionLevel)) {
        byDimension.set(tp.detectionLevel, []);
      }
      byDimension.get(tp.detectionLevel)!.push(tp);
    });

    // 3. Use the possibilistic gating network to fuse and select the best TPs
    this.logger.info(
      `Applying possibilistic gating network to ${turningPoints.length} candidates.`,
    );
    const fusedTurningPoints =
      this.createPossibilisticGatingNetwork(byDimension);

    this.logger.info(
      `Possibilistic fusion resulted in ${fusedTurningPoints.length} total TPs`,
    );

    // NEW: Apply counterfactual analysis if enabled
    const enhancedTurningPoints =
      this.counterfactualAnalyzer && this.config.enableCounterfactualAnalysis
        ? this.counterfactualAnalyzer.enhanceTurningPointSelection(
          fusedTurningPoints,
        )
        : fusedTurningPoints;

    // Apply hard cap on the enhanced results
    if (enhancedTurningPoints.length > this.config.maxTurningPoints) {
      const sortedByQuality = enhancedTurningPoints.sort((a, b) => {
        const scoreA = a.supportScore || a.significance;
        const scoreB = b.supportScore || b.significance;
        return scoreB - scoreA;
      });

      return sortedByQuality
        .slice(0, this.config.maxTurningPoints)
        .sort((a, b) => a.span.startIndex - b.span.startIndex);
    }

    // 4. ENFORCE HARD CAP: Sort by quality and take top maxTurningPoints
    if (fusedTurningPoints.length > this.config.maxTurningPoints) {
      this.logger.info(
        `Enforcing hard cap: Reducing ${fusedTurningPoints.length} TPs to ${this.config.maxTurningPoints}`,
      );

      // Sort by composite quality score (support score if available, otherwise significance)
      const sortedByQuality = fusedTurningPoints.sort((a, b) => {
        const scoreA = a.supportScore || a.significance;
        const scoreB = b.supportScore || b.significance;
        if (scoreB !== scoreA) return scoreB - scoreA; // Higher scores first

        // Tie-breaker: prefer higher dimensions
        if (b.detectionLevel !== a.detectionLevel)
          return b.detectionLevel - a.detectionLevel;

        // Final tie-breaker: chronological order
        return a.span.startIndex - b.span.startIndex;
      });

      const cappedTurningPoints = sortedByQuality.slice(
        0,
        this.config.maxTurningPoints,
      );

      this.logger.info(
        `Hard cap applied: Selected top ${cappedTurningPoints.length} TPs by quality score`,
      );

      // 5. Final sort by chronological order for output
      return cappedTurningPoints.sort(
        (a, b) => a.span.startIndex - b.span.startIndex,
      );
    }

    // 5. Final sort and return (if under the cap)
    return fusedTurningPoints.sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
  }
  /**
   * Combine turning points from different dimensions
   * (Using original logic from the second code block)
   * @param localTurningPoints - An array of turning points detected in the current dimension.
   * @param higherDimensionTurningPoints - An array of turning points detected in a higher dimension
   * @param phiMap - (Experimental) A map containing phi scores for each turning point, used when `enableExperimentalPhi` is true. @experimental
   */
  private combineTurningPoints(
    localTurningPoints: TurningPoint[],
    higherDimensionTurningPoints: TurningPoint[],
    phiMap: Map<string, number> = new Map(),
  ): TurningPoint[] {
    this.logger.info(
      `Combining ${localTurningPoints.length} local (dim ${localTurningPoints[0]?.detectionLevel ?? "N/A"}) and ${higherDimensionTurningPoints.length} higher (dim ${higherDimensionTurningPoints[0]?.detectionLevel ?? "N/A"}) TPs.`,
    );

    // Prioritize higher-dimensional turning points by boosting their significance (original logic)
    const boostedHigher = higherDimensionTurningPoints.map((tp) => ({
      ...tp,
      // Apply a boost, ensuring it doesn't exceed 1.0
      significance: Math.min(1.0, tp.significance * 1.2), // Adjusted boost factor slightly
      // Keep original detectionLevel for merging logic
    }));

    // Combine all turning points
    const allTurningPoints = [...localTurningPoints, ...boostedHigher];
    this.logger.info(
      `Total TPs before cross-level merge: ${allTurningPoints.length}`,
    );

    // Merge overlapping turning points across dimensions, prioritizing higher dimensions/significance
    const mergedTurningPoints = this.mergeAcrossLevels(allTurningPoints);
    this.logger.info(
      `Merged across levels to ${mergedTurningPoints.length} TPs.`,
    );

    // Filter the combined & merged list to keep the most significant ones overall
    const filteredTurningPoints = this.filterSignificantTurningPoints(
      mergedTurningPoints,
      phiMap,
    );

    this.logger.info(
      `Final combined and filtered TPs: ${filteredTurningPoints.length}`,
    );
    // Sort by position in conversation before returning
    return filteredTurningPoints.sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
  }

  /**
   * Merge similar or overlapping turning points *within* the same dimension
   * (Using original logic from the second code block)
   *
   * @param phiMap - (Experimental) A map containing phi scores for each turning point, used when `enableExperimentalPhi` is true. @experimental
   */
  private mergeSimilarTurningPoints(
    turningPoints: TurningPoint[],
    phiMap: Map<string, number>,
  ): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    // Sort turning points by start index
    const sorted = [...turningPoints].sort(
      (a, b) => a.span.startIndex - b.span.startIndex,
    );
    const merged: TurningPoint[] = [];
    let currentTp = sorted[0]; // Use a more descriptive name

    // --- DYNAMIC THRESHOLD CALCULATION (NEW) ---
    let phiDivergenceThreshold = 0.3; // A reasonable default fallback
    if (this.config.enableExperimentalPhi && turningPoints.length > 2) {
      this.logger.info(
        `mergeSimilarTurningPoints: Calculating dynamic phi divergence threshold for merging based on ${turningPoints.length} TPs since enableExperimentalPhi is true.`,
      );
      // Create a sorted list of the phi scores for the current set of TPs
      const phiScores = turningPoints
        .map((tp) => phiMap.get(tp.id) || 0.5)
        .sort((a, b) => a - b);

      // Calculate the 25th percentile of the phi scores. This represents the
      // lower bound of significance difference within this specific context.
      const percentileIndex = Math.floor(phiScores.length * 0.25);
      const percentileValue = phiScores[percentileIndex];

      // The threshold is the percentile value, but capped to prevent it from being too loose.
      // This ensures we only merge points whose significance is closer than the bottom 25% of scores.
      phiDivergenceThreshold = Math.min(percentileValue, 0.4);
      this.logger.info(
        `Dynamically calculated phi divergence threshold for merging: ${phiDivergenceThreshold.toFixed(3)}`,
      );
    }

    for (let i = 1; i < sorted.length; i++) {
      const nextTp = sorted[i];
      // FIX: Skip if the next turning point is the exact same as the current one.
      // This handles duplicates created by overlapping chunks.
      if (currentTp.id === nextTp.id) {
        continue;
      }
      // Check conditions for merging (original logic)
      // Check conditions for merging (original logic)
      const isOverlapping =
        nextTp.span.startIndex <= currentTp.span.endIndex + 2; // Allow small gap
      const isSimilarCategory =
        nextTp.category.toLocaleLowerCase() ===
        currentTp.category.toLocaleLowerCase();

      // --- NEW PHI-AWARE MERGE CONDITION ---
      let isThematicallyCoherent = true;
      if (this.config.enableExperimentalPhi) {
        const phiCurrent = phiMap.get(currentTp.id) || 0.5;
        const phiNext = phiMap.get(nextTp.id) || 0.5;

        // MORE PERMISSIVE: Use larger threshold and bonus for high-phi points
        const adjustedThreshold =
          phiDivergenceThreshold *
          (this.config.phiMergeThresholdMultiplier || 1.5);

        // High-phi points get more leeway
        const phiBonus = Math.max(phiCurrent, phiNext) > 0.7 ? 0.1 : 0;

        isThematicallyCoherent =
          Math.abs(phiCurrent - phiNext) < adjustedThreshold + phiBonus;

        this.logger.info(
          `Phi coherence check: ${isThematicallyCoherent} (threshold: ${adjustedThreshold + phiBonus})`,
        );
      }

      // Added closeness check from original code
      const hasCloseIndices =
        nextTp.span.startIndex - currentTp.span.endIndex <= 3;

      // Merge if overlapping OR close, AND same category
      if (
        (isOverlapping || hasCloseIndices) &&
        isSimilarCategory &&
        isThematicallyCoherent
      ) {
        this.logger.info(
          `    Merging similar TPs (Dim ${currentTp.detectionLevel}): ${currentTp.id} and ${nextTp.id}`,
        );
        // Merge the turning points
        const newLabel = this.createMergedLabel(currentTp.label, nextTp.label);

        // Create merged span (min start, max end)
        const mergedSpan = this.ensureChronologicalSpan({
          startId:
            currentTp.span.startIndex <= nextTp.span.startIndex
              ? currentTp.span.startId
              : nextTp.span.startId,
          endId:
            currentTp.span.endIndex >= nextTp.span.endIndex
              ? currentTp.span.endId
              : nextTp.span.endId,
          startIndex: Math.min(
            currentTp.span.startIndex,
            nextTp.span.startIndex,
          ),
          endIndex: Math.max(currentTp.span.endIndex, nextTp.span.endIndex),
        });

        // Update the deprecated span too (original logic, though less relevant now)
        // Note: deprecatedSpan might not exist if TPs came from meta-messages
        const mergedDeprecatedSpan =
          currentTp.deprecatedSpan && nextTp.deprecatedSpan
            ? {
              startIndex: Math.min(
                currentTp.deprecatedSpan.startIndex,
                nextTp.deprecatedSpan.startIndex,
              ),
              endIndex: Math.max(
                currentTp.deprecatedSpan.endIndex,
                nextTp.deprecatedSpan.endIndex,
              ),
              startMessageId:
                mergedSpan.startIndex === currentTp.deprecatedSpan.startIndex
                  ? currentTp.deprecatedSpan.startMessageId
                  : nextTp.deprecatedSpan.startMessageId,
              endMessageId:
                mergedSpan.endIndex === currentTp.deprecatedSpan.endIndex
                  ? currentTp.deprecatedSpan.endMessageId
                  : nextTp.deprecatedSpan.endMessageId,
            }
            : undefined; // Handle cases where deprecatedSpan might be missing

        // Combine  quotes (unique, limited)

        const mergedQuotes = Array.from(
          new Set([...(currentTp.quotes || []), ...(nextTp.quotes || [])]),
        ).slice(0, 3); // Limit quotes too

        // determine the emotional Tone and sentiment

        let mergedEmotionalTone = currentTp.emotionalTone;
        let mergedSentiment = currentTp.sentiment;

        if (
          this.config.enableExperimentalPhi &&
          currentTp.emotionalTone !== nextTp.emotionalTone
        ) {
          const tones = [currentTp.emotionalTone, nextTp.emotionalTone].sort();
          mergedEmotionalTone = `${tones[0]}-${tones[1]}`; // e.g., "curiosity-surprise"
        } else {
          // Fallback to the original significance-based selection
          mergedEmotionalTone =
            currentTp.significance >= nextTp.significance
              ? currentTp.emotionalTone
              : nextTp.emotionalTone;
        }

        // A simple rule for sentiment: if any part is negative, the merged sentiment is negative.
        mergedSentiment =
          currentTp.sentiment?.toLocaleLowerCase().startsWith("n") ||
            nextTp.sentiment?.toLocaleLowerCase().startsWith("n")
            ? "negative"
            : "positive";

        // Update the current TP to be the merged version
        currentTp = {
          ...currentTp, // Keep most properties of the first TP
          id: `${currentTp.id}-merged-${nextTp.span.startIndex}`, // Indicate merge in ID
          label: newLabel,
          span: mergedSpan,
          // Only include deprecatedSpan if it was successfully merged
          ...(mergedDeprecatedSpan && { deprecatedSpan: mergedDeprecatedSpan }),
          semanticShiftMagnitude:
            (currentTp.semanticShiftMagnitude + nextTp.semanticShiftMagnitude) /
            2,
          quotes: mergedQuotes,
          // Boost significance slightly, cap at 1.0 (original logic)
          significance: Math.min(
            1.0,
            ((currentTp.significance + nextTp.significance) / 2) * 1.1,
          ),
          // Take max complexity (original logic)
          complexityScore: Math.max(
            currentTp.complexityScore,
            nextTp.complexityScore,
          ),
          // Combine emotional tone/sentiment logically (e.g., take the one from the more significant TP)
          emotionalTone: mergedEmotionalTone,

          sentiment: mergedSentiment,
        };
      } else {
        // If not merging, push the completed current TP and move to the next
        merged.push(currentTp);
        currentTp = nextTp;
      }
    }

    // Add the last processed TP
    merged.push(currentTp);

    return merged;
  }

  /**
   * Merge turning points across different dimensions with priority to higher dimensions
   * (Using original logic from the second code block)
   */
  private mergeAcrossLevels(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length <= 1) return turningPoints;

    const sorted = [...turningPoints].sort((a, b) => {
      if (b.detectionLevel !== a.detectionLevel)
        return b.detectionLevel - a.detectionLevel;
      if (b.significance !== a.significance)
        return b.significance - a.significance;
      return a.span.startIndex - b.span.startIndex;
    });

    const merged: TurningPoint[] = [];
    const coveredIndices: Set<number> = new Set();

    this.logger.info(
      `Merging across levels. Input count: ${sorted.length}. Prioritizing higher dimension/significance.`,
    );

    for (const tp of sorted) {
      let overlapCount = 0;
      const spanSize = tp.span.endIndex - tp.span.startIndex + 1;
      if (spanSize <= 0) continue;

      for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
        if (coveredIndices.has(i)) {
          overlapCount++;
        }
      }
      const overlapRatio = overlapCount / spanSize;

      // FIXED: Use dynamic overlap threshold based on dimension and span size
      const dynamicOverlapThreshold = this.calculateDynamicOverlapThreshold(
        tp.detectionLevel,
        spanSize,
      );

      if (overlapRatio < dynamicOverlapThreshold) {
        merged.push(tp);
        for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
          coveredIndices.add(i);
        }
        this.logger.info(
          `Keeping TP ${tp.id} (Dim ${tp.detectionLevel}, Sig ${tp.significance.toFixed(2)}). ` +
          `Overlap: ${(overlapRatio * 100).toFixed(0)}% < ${(dynamicOverlapThreshold * 100).toFixed(0)}%`,
        );
      } else {
        this.logger.info(
          `Skipping TP ${tp.id} (Dim ${tp.detectionLevel}) due to overlap ` +
          `(${(overlapRatio * 100).toFixed(0)}% >= ${(dynamicOverlapThreshold * 100).toFixed(0)}%)`,
        );
      }
    }

    this.logger.info(
      `Finished merging across levels. Output count: ${merged.length}.`,
    );
    return merged.sort((a, b) => a.span.startIndex - b.span.startIndex);
  }

  /**
   * Create a merged label (Using original logic)
   */
  private createMergedLabel(label1: string, label2: string): string {
    if (label1 === label2) return label1;
    if (label1.includes("Unclassified")) return label2;
    if (label2.includes("Unclassified")) return label1;

    const commonWords = this.findCommonWords(label1, label2);
    if (commonWords.length > 0) {
      // Simple common word approach (original)
      return commonWords.join(" ") + " Discussion";
    }
    // Fallback concatenation (original)
    return `${label1} / ${label2}`.substring(0, 70); // Add length limit
  }

  /**
   * Find common significant words (Using original logic)
   */
  private findCommonWords(label1: string, label2: string): string[] {
    const words1 = label1.toLowerCase().split(/\s+/);
    const words2 = label2.toLowerCase().split(/\s+/);
    const stopwords = new Set([
      "to",
      "the",
      "in",
      "of",
      "and",
      "a",
      "an",
      "on",
      "for",
      "with",
      "shift",
      "discussion",
      "about",
      "summary",
    ]); // Added more stopwords
    // Filter common words, exclude stopwords, ensure decent length
    return words1.filter(
      (word) =>
        word.length > 3 && words2.includes(word) && !stopwords.has(word),
    );
  }

  /**
   * Get a unique key for a message span (Using original logic)
   */
  private getSpanKey(tp: TurningPoint): string {
    return `${tp.span.startIndex}-${tp.span.endIndex}`;
  }

  /**
   * Check if a span overlaps with any spans in the covered set (Using original logic)
   * Note: This might be less accurate than the index-based check in mergeAcrossLevels now.
   * Kept for potential use by filterSignificantTurningPoints if it uses range strings.
   */
  private isSpanOverlapping(
    tp: TurningPoint,
    coveredSpans: Set<string>,
  ): boolean {
    // Check exact span match
    if (coveredSpans.has(this.getSpanKey(tp))) return true;

    // Check partial overlap (original logic)
    for (let i = tp.span.startIndex; i <= tp.span.endIndex; i++) {
      for (let j = i; j <= tp.span.endIndex; j++) {
        if (coveredSpans.has(`${i}-${j}`)) {
          const overlapSize = j - i + 1;
          const tpSize = tp.span.endIndex - tp.span.startIndex + 1;
          // Original 50% threshold
          if (tpSize > 0 && overlapSize / tpSize >= 0.5) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Generate embeddings for an array of messages (Using original logic)
   */
  private async generateMessageEmbeddings(
    messages: Message[],
    dimension = 0,
  ): Promise<MessageEmbedding[]> {
    const embeddings: MessageEmbedding[] = new Array(messages.length);
    // Using original concurrency limit of 4
    // console.info(`Generating embeddings for ${messages.length} messages with dimension ${dimension}.`);
    await async.eachOfLimit(
      messages,
      this.config.embeddingConcurrency,
      async (message, indexStr) => {
        let candidateText = message.message;
        if (dimension > 0 && message instanceof MetaMessage) {
          // For meta-messages, use the original message text
          const metaMessage = message as MetaMessage;
          const messagesWithinMeta =
            metaMessage.getMessagesInTurningPointSpanToMessagesArray();
          // naiviely concatenate the last and first message
          candidateText = `${messagesWithinMeta[0].message} ${messagesWithinMeta[messagesWithinMeta.length - 1].message}`;
          console.info(
            `Meta message ${message.id} contains ${messagesWithinMeta.length} messages.`,
          );
        }
        const index = Number(indexStr);
        try {
          const embedding = await this.getEmbedding(candidateText);
          // Store the original index from the input 'messages' array
          embeddings[index] = {
            id: message.id,
            index: index, // Store the index within the current chunk/message list being processed
            embedding,
          };
        } catch (error: any) {
          this.logger.info(
            `Error generating embedding for msg ${message.id} at index ${index}: ${error.message}. Creating zero vector.`,
          );
          const embeddingSize = 1536; // Assuming text-embedding-3-small
          embeddings[index] = {
            id: message.id,
            index: index,
            embedding: new Float32Array(embeddingSize).fill(0),
          };
        }
      },
    );
    // Filter out any potential null/undefined entries if errors occurred
    return embeddings.filter((e) => e);
  }

  /**
   * Retrieves the embedding vector for a given text string using the configured embedding model.
   *
   * This method implements intelligent token management, crypto-hashed caching, and multi-provider
   * support for embedding generation within the ARC/CRA framework.
   *
   * @param text - The input text to generate embeddings for. Will be automatically truncated if exceeding token limits.
   * @param naiveTokenLimit - Maximum number of tokens allowed before truncation (default: 8192).
   *                          Uses a character-to-token ratio estimation for efficient preprocessing.
   *
   * @returns Promise<Float32Array> - The embedding vector as a Float32Array. Dimension depends on the model:
   *   - text-embedding-3-small: 1536 dimensions
   *   - text-embedding-3-large: 3072 dimensions
   *   - Other models: 1024 dimensions (fallback)
   *
   * @throws Will not throw errors directly, but logs warnings and returns zero vectors on API failures.
   *
   * @remarks
   * **Token Management:**
   * - Implements dynamic text truncation based on token counting to prevent API errors
   * - Uses character-to-token ratio estimation for efficient preprocessing
   * - Recalculates ratio iteratively until under the specified limit
   *
   * **Caching Strategy:**
   * - Uses SHA-256 crypto hashing for cache keys to avoid issues with special characters and long text
   * - Cache keys include both model name and (truncated) text content for uniqueness
   * - Leverages the class's LRU cache with configurable RAM limits via `embeddingCacheRamLimitMB`
   * - Cache operations are logged in debug mode for monitoring effectiveness
   *
   * **Multi-Provider Support:**
   * - Supports OpenAI API and OpenAI-compatible endpoints (Ollama, LM Studio, etc.)
   * - Uses EMBEDDINGS_API_KEY environment variable first, falls back to OPENAI_API_KEY
   * - Respects the configured `embeddingEndpoint` for custom embedding providers
   * - Maintains compatibility with standard OpenAI embedding response format
   *
   * **Error Handling:**
   * - Returns zero vectors instead of throwing on API failures
   * - Logs detailed error information for debugging
   * - Gracefully handles invalid API responses or network issues
   * - Maintains system stability during transient embedding service outages
   *
   * **Integration with ARC/CRA Framework:**
   * - Critical component for semantic distance calculations in turning point detection
   * - Supports both base message embeddings (dimension 0) and meta-message embeddings (dimension > 0)
   * - Cache performance directly impacts overall framework processing speed
   * - Embedding quality affects the accuracy of semantic shift detection
   */

  async getEmbedding(
    text: string,
    naiveTokenLimit = 8192,
  ): Promise<Float32Array> {
    // Ensure that the input text length is less than 8192 tokens
    let tokensCount = countTokens(text);
    let textCharToTokenRatio = text.length / tokensCount;

    while (tokensCount > naiveTokenLimit) {
      // Remove the exact number of characters to get the token count under limit
      const charsToRemove = Math.ceil(
        (tokensCount - naiveTokenLimit) * textCharToTokenRatio,
      );
      text = text.substring(0, text.length - charsToRemove);
      tokensCount = countTokens(text);
    }

    // Create crypto hash cache key AFTER text truncation (important!)
    const cacheKey = crypto
      .createHash("sha256")
      .update(`${this.config.embeddingModel}:${text}`)
      .digest("hex");

    // Check cache first
    const cachedEmbedding = this.embeddingCache.get(cacheKey);
    if (cachedEmbedding) {
      if (this.config.debug) {
        this.logger.debug(
          `Cache hit for embedding (${cacheKey.substring(0, 8)}...)`,
        );
      }
      return cachedEmbedding;
    }

    try {
      // Create OpenAI client with proper configuration
      const openai = new OpenAI({
        apiKey:
          this.config.embeddingEndpoint == undefined
            ? process.env.OPENAI_API_KEY
            : process.env.EMBEDDINGS_API_KEY,
        baseURL: this.config.embeddingEndpoint,
      });

      const response = await openai.embeddings.create({
        model: this.config.embeddingModel,
        input: text,
        encoding_format: "float",
      });

      if (
        response.data &&
        response.data.length > 0 &&
        response.data[0].embedding
      ) {
        const embedding = new Float32Array(response.data[0].embedding);

        // Store in cache with crypto hash key
        this.embeddingCache.set(cacheKey, embedding);

        // if (this.config.debug) { no need for this its too muc hnoise
        //   this.logger.info(
        //     `Cache miss - stored new embedding (${cacheKey.substring(0, 8)}...)`,
        //   );
        // }

        return embedding;
      } else {
        throw new Error("Invalid embedding response structure from OpenAI");
      }
    } catch (err: any) {
      this.logger.info(
        `Error generating embedding: ${err.message}. Returning zero vector.`,
      );
      const embeddingSize =
        this.config.embeddingModel === "text-embedding-3-small"
          ? 1536
          : this.config.embeddingModel === "text-embedding-3-large"
            ? 3072
            : 1024;
      return new Float32Array(embeddingSize).fill(0);
    }
  }

  /**
   * Dynamically calculates the optimal complexity saturation threshold based on
   * the distribution of complexity scores from detected turning points.
   *
   * This implements adaptive threshold selection that ensures the saturation
   * mechanism activates for the top X% of complexity scores rather than using
   * a fixed threshold that may not match the actual data distribution.
   */
  private calculateDynamicComplexitySaturation(
    turningPoints: TurningPoint[],
  ): number {
    if (!this.config.enableDynamicComplexitySaturation) {
      return this.config.complexitySaturationThreshold;
    }

    // Need minimum samples for statistical validity
    const minSamples = this.config.dynamicSaturationMinSamples || 10;
    if (turningPoints.length < minSamples) {
      this.logger.info(
        `Dynamic complexity saturation: Need ${minSamples} samples, have ${turningPoints.length}. Using initial threshold: ${this.config.complexitySaturationThreshold}`,
      );
      return this.config.complexitySaturationThreshold;
    }

    // Extract and sort complexity scores
    const complexityScores = turningPoints
      .map((tp) => tp.complexityScore)
      .filter((score) => isFinite(score) && score > 0)
      .sort((a, b) => a - b);

    if (complexityScores.length === 0) {
      this.logger.warn(
        "No valid complexity scores found, using initial threshold",
      );
      return this.config.complexitySaturationThreshold;
    }

    // Calculate target percentile (default 85th percentile means top 15% get saturated)
    const targetPercentile =
      this.config.dynamicSaturationTargetPercentile || 0.15;
    const percentileIndex = Math.floor(
      complexityScores.length * (1 - targetPercentile),
    );
    const calculatedThreshold = complexityScores[Math.max(0, percentileIndex)];

    // Apply bounds to prevent extreme values
    const initialThreshold = this.config.complexitySaturationThreshold;
    const minThreshold = Math.max(2.0, initialThreshold * 0.7); // Don't go below 70% of initial
    const maxThreshold = Math.min(5.0, initialThreshold * 1.3); // Don't exceed 130% of initial

    const boundedThreshold = Math.max(
      minThreshold,
      Math.min(maxThreshold, calculatedThreshold),
    );

    this.logger.info(
      `Dynamic complexity saturation calculated: ${boundedThreshold.toFixed(2)} ` +
      `(from ${complexityScores.length} samples, target ${(targetPercentile * 100).toFixed(0)}th percentile, ` +
      `raw: ${calculatedThreshold.toFixed(2)}, initial: ${initialThreshold})`,
    );

    return boundedThreshold;
  }

  /**
   * Calculate semantic distance (Using original logic without sigmoid adjustment)
   */
  private calculateSemanticDistance(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }

    if (magA === 0 || magB === 0) return 1; // Maximum distance if either is a zero vector

    const similarity = dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
    const distance = 1 - similarity; // Convert similarity to distance

    return distance;
  }

  /**
   * Chunk a conversation (Using original logic)
   */
  private async chunkConversation(
    messages: Message[],
    dimension = 0,
  ): Promise<ChunkingResult> {
    const chunks: Message[][] = [];
    // scale down the min messages based on each depth increase
    const baseMinMessages = this.config.minMessagesPerChunk; // Preserve original value
    const dimensionScaleFactor = Math.max(0.1, Math.pow(0.35, dimension));
    const minMessages = Math.max(
      2,
      Math.round(baseMinMessages * dimensionScaleFactor),
    );
    // But you need to scale maxTokensPerChunk too
    const tokenScaleFactor = Math.max(0.2, Math.pow(0.5, dimension));
    const maxTokens = Math.max(
      this.config.minTokensPerChunk,
      Math.round(this.config.maxTokensPerChunk * tokenScaleFactor),
    );

    // Handle case where input has fewer than minimum messages (original logic)
    if (messages.length < minMessages) {
      this.logger.info(
        `Input messages (${messages.length}) less than minMessagesPerChunk (${minMessages}). Returning as single chunk.`,
      );
      // Return single chunk only if it's not empty
      return {
        chunks: messages.length > 0 ? [[...messages]] : [],
        numChunks: messages.length > 0 ? 1 : 0,
        avgTokensPerChunk:
          messages.length > 0
            ? await this.getMessageArrayTokenCount(messages)
            : 0,
      };
    }

    let currentChunk: Message[] = [];
    let currentTokens = 0;
    let totalTokens = 0;
    const overlapSize = Math.max(0, Math.min(2, Math.floor(minMessages / 2))); // Keep overlap small, max 2

    // Ideal chunk size logic (original) - helps guide chunking beyond just token limits
    const idealMessageCount = Math.max(
      minMessages,
      Math.min(
        10,
        Math.ceil(
          messages.length / Math.max(1, Math.floor(messages.length / 10)),
        ),
      ), // Aim for ~10 chunks max?
    );
    this.logger.info(
      `    Chunking ${messages.length} messages. MinMsgs: ${minMessages}, MaxTokens: ${maxTokens}, IdealMsgCount: ${idealMessageCount}, Overlap: ${overlapSize}`,
    );

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      // Handle potential undefined messages just in case
      if (!message) continue;

      const tokens = await this.getMessageTokenCount(message.message);
      totalTokens += tokens;

      // Add message to current chunk
      currentChunk.push(message);
      currentTokens += tokens;

      // Determine if we should close this chunk (original logic)
      const hasMinMessages = currentChunk.length >= minMessages;
      const hasIdealSize = currentChunk.length >= idealMessageCount; // Use ideal count
      const approachingMaxTokens = currentTokens >= maxTokens * 0.9; // Increase threshold slightly
      const isLastMessage = i === messages.length - 1;
      const significantlyOverMaxTokens = currentTokens > maxTokens * 1.1; // Check if significantly over

      // Close chunk conditions (refined slightly)
      // 1. Last message: always close.
      // 2. Min messages met AND (approaching/over max tokens OR reached ideal size)
      if (
        isLastMessage ||
        (hasMinMessages &&
          (approachingMaxTokens || hasIdealSize || significantlyOverMaxTokens))
      ) {
        // Add the chunk
        chunks.push([...currentChunk]);
        this.logger.info(
          `      Created chunk ${chunks.length} with ${currentChunk.length} messages, ${currentTokens} tokens. Ends at index ${i}.`,
        );

        // If not the last message, start next chunk with overlap
        if (!isLastMessage) {
          const startIndexForNextChunk = Math.max(
            0,
            currentChunk.length - overlapSize,
          );
          currentChunk = currentChunk.slice(startIndexForNextChunk);
          currentTokens = await this.getMessageArrayTokenCount(currentChunk);
          this.logger.info(
            `      Starting next chunk with ${currentChunk.length} overlapping messages, ${currentTokens} tokens.`,
          );
        } else {
          currentChunk = []; // Clear chunk if it was the last message
          currentTokens = 0;
        }
      }
    }

    // --- Post-processing Chunks (similar to original logic but simplified) ---
    // If only one chunk was created for a large conversation, try splitting it
    if (chunks.length === 1 && messages.length >= minMessages * 2) {
      const singleChunk = chunks[0];
      const midPointIndex = Math.floor(singleChunk.length / 2);

      // Ensure split results in chunks meeting min size requirement
      if (
        midPointIndex >= minMessages &&
        singleChunk.length - midPointIndex >= minMessages
      ) {
        this.logger.info(
          "    Single chunk detected for large conversation, attempting to split.",
        );
        const firstChunk = singleChunk.slice(0, midPointIndex);
        // Apply overlap when splitting
        const secondChunkStartIndex = Math.max(0, midPointIndex - overlapSize);
        const secondChunk = singleChunk.slice(secondChunkStartIndex);

        chunks.splice(0, 1, firstChunk, secondChunk); // Replace single chunk with two
        this.logger.info(
          `    Successfully split into two chunks: ${firstChunk.length} msgs and ${secondChunk.length} msgs.`,
        );
      }
    }

    // Ensure all chunks meet minimum message requirement, merging small trailing chunks if necessary
    if (chunks.length > 1) {
      const lastChunk = chunks[chunks.length - 1];
      if (lastChunk.length < minMessages) {
        this.logger.info(
          `    Last chunk (${lastChunk.length} msgs) is smaller than min size (${minMessages}). Merging with previous.`,
        );
        const secondLastChunk = chunks[chunks.length - 2];
        // Combine, avoiding duplicates from overlap if possible
        const combinedChunk = [...secondLastChunk];
        const lastIdInSecondLast =
          secondLastChunk[secondLastChunk.length - 1]?.id;
        let appendStartIndex = 0;
        if (overlapSize > 0 && lastChunk[0]?.id === lastIdInSecondLast) {
          appendStartIndex = 1; // Skip first element of last chunk if it's the same as end of previous
        }
        combinedChunk.push(...lastChunk.slice(appendStartIndex));

        // Replace the last two chunks with the merged one
        chunks.splice(chunks.length - 2, 2, combinedChunk);
        this.logger.info(
          `    Merged last two chunks. New chunk count: ${chunks.length}.`,
        );
      }
    }
    // --- End Post-processing ---

    const numChunks = chunks.length;
    const avgTokens = numChunks > 0 ? totalTokens / numChunks : 0; // Avoid division by zero
    const avgMessagesPerChunk = numChunks > 0 ? messages.length / numChunks : 0;

    this.logger.info(
      `    Finished chunking. Created ${numChunks} chunks. Avg Tokens: ${avgTokens.toFixed(0)}, Avg Msgs: ${avgMessagesPerChunk.toFixed(1)}`,
    );

    return {
      chunks,
      numChunks,
      avgTokensPerChunk: avgTokens,
    };
  }

  /**
   * Ensures a message span is in chronological order by index
   * (Using original logic)
   */
  private ensureChronologicalSpan(span: MessageSpan): MessageSpan {
    if (span.startIndex > span.endIndex) {
      this.logger.info(
        `Warning: Correcting reversed span indices (${span.startIndex} > ${span.endIndex}) for IDs ${span.startId}/${span.endId}.`,
      );
      // Create a new span with swapped values to maintain immutability
      return {
        startId: span.endId, // Swap IDs
        endId: span.startId,
        startIndex: span.endIndex, // Swap indices
        endIndex: span.startIndex,
        originalSpan: span.originalSpan ?? span, // Store original if needed
      };
    }
    return span; // Return original if already chronological
  }

  /**
   * Get token count for a message with caching (Using original logic)
   */
  private async getMessageTokenCount(text: string): Promise<number> {
    const hash = crypto.createHash("sha256").update(text).digest("hex");
    if (tokenCountCache.has(hash)) {
      return tokenCountCache.get(hash)!;
    }

    let count: number;
    try {
      // Use external countTokens function (original logic)
      count = countTokens(text);
    } catch (err: any) {
      this.logger.info(
        `Error counting tokens: ${err.message}. Falling back to length/4.`,
      );
      /**
       * Fallback (original logic) based on a naive approach that uses a ratio of four characters per token. This method is inaccurate because the token count is also influenced by certain special strings or characters.
       * - The ratio can vary significantly depending on the type of text content; for example, JSON data may yield a slightly different ratio. However, this four-character ratio is generally reasonable for semantic text.
       */
      count = Math.ceil(text.length / 4);
    }

    // Cache the result (original logic)
    tokenCountCache.set(hash, count);

    return count;
  }

  /**
   * Get token count for multiple messages (Using original logic)
   */
  async getMessageArrayTokenCount(messages: Message[]): Promise<number> {
    let total = 0;
    for (const message of messages) {
      // Handle potentially undefined messages in array
      if (message?.message) {
        total += await this.getMessageTokenCount(message.message);
      }
    }
    return total;
  }

  /**
   * Get the convergence history for analysis (Using original logic)
   */
  public getConvergenceHistory(): ConvergenceState[] {
    return this.convergenceHistory;
  }

  private prepareContextualInfoMeta(
    _beforeMessage: Message,
    _afterMessage: Message,
    span: MessageSpan,
    originalMessages?: Message[],
    dimension = 0,
    messagesToAddPerContextualUnit = 2,
    addMessagesWithinSpan = false,
    outputFormat: "modular" | "markdown" = "modular",
  ): string | { beforeAndAfterContext: string; withinSpanContext?: string } {
    // Validate inputs
    if (!_beforeMessage || !_afterMessage) {
      throw new Error("Both before and after messages are required");
    }

    if (!originalMessages || originalMessages.length === 0) {
      throw new Error(
        "Original messages array is required and cannot be empty",
      );
    }

    // --- MODULAR CONTEXT GENERATION ---
    const generateContextualInfo = () => {
      if (
        _beforeMessage instanceof MetaMessage &&
        _afterMessage instanceof MetaMessage
      ) {
        try {
          // Get "before-and-after" context
          const beforeAndAfterContext =
            MetaMessage.getMessagesContentContextualAidFromJustProvidedBeforeAndAfterMessages(
              _beforeMessage,
              _afterMessage,
              dimension,
              messagesToAddPerContextualUnit,
              this.config.max_character_length,
              originalMessages,
              "before-and-after",
            );

          // Optionally get "within" context
          let withinSpanContext: string | undefined;
          if (addMessagesWithinSpan) {
            withinSpanContext =
              MetaMessage.getMessagesContentContextualAidFromJustProvidedBeforeAndAfterMessages(
                _beforeMessage,
                _afterMessage,
                dimension,
                messagesToAddPerContextualUnit,
                this.config.max_character_length,
                originalMessages,
                "within",
              );
          }

          return { beforeAndAfterContext, withinSpanContext };
        } catch (error) {
          console.error(
            "Error in MetaMessage contextual aid generation:",
            error,
          );
          // Fallback to regular message processing
          const fallbackContext = this.prepareContextualInfoMessage(
            _beforeMessage,
            _afterMessage,
            span,
            originalMessages,
            dimension,
            addMessagesWithinSpan,
          );
          return { beforeAndAfterContext: fallbackContext };
        }
      } else {
        // Handle non-MetaMessage cases
        const regularContext = this.prepareContextualInfoMessage(
          _beforeMessage,
          _afterMessage,
          span,
          originalMessages,
          dimension,
          addMessagesWithinSpan,
        );
        return { beforeAndAfterContext: regularContext };
      }
    };

    const contextData = generateContextualInfo();

    // --- OUTPUT FORMATTING ---
    if (outputFormat === "modular") {
      // Return structured object for modular assembly
      return contextData;
    } else if (outputFormat === "markdown") {
      // Return formatted markdown string
      let markdownOutput = `## Contextual Information\n\n### Surrounding Context\n${contextData.beforeAndAfterContext}`;

      if (contextData.withinSpanContext) {
        markdownOutput += `\n\n### Within Span Context\n${contextData.withinSpanContext}`;
      }

      return markdownOutput;
    } else {
      // Default: concatenated string (backward compatibility)
      return contextData.withinSpanContext
        ? `${contextData.beforeAndAfterContext}\n\n${contextData.withinSpanContext}`
        : contextData.beforeAndAfterContext;
    }
  }

  /**
   * Prepares contextual information to be appended to the LLM prompt.
   * Gathers nearby messages (before, after, and within the span) for additional context.
   * - This preperation is soley meant for dimension at 0, wherein the messages are still base level messages, rather than MetaMessages, which encompass a group of turning points.
   */
  private prepareContextualInfoMessage(
    _beforeMessage: Message,
    _afterMessage: Message,
    span: MessageSpan,
    originalMessages?: Message[],
    dimension: number = 0,
    addMessagesWithinSpan = false,
  ): string {
    if (
      dimension > 0 ||
      _beforeMessage instanceof MetaMessage ||
      _afterMessage instanceof MetaMessage
    ) {
      throw new Error(
        "Contextual information preparation is not supported for dimensions greater than 0.",
      );
    }

    const neighborsToAdd = Math.max(
      Math.round(this.config.minMessagesPerChunk / 2),
      1,
    );

    const originalMessagesNeighborsBefore = originalMessages
      ?.slice(
        Math.max(
          0,
          MetaMessage.findIndexOfMessageFromId({
            beforeMessage: _beforeMessage,
            afterMessage: _afterMessage,
            id: span.startId,
            messages: originalMessages,
          }),
        ),
        span.startIndex,
      )
      .filter(Boolean);
    const messageIndexAfterStart = MetaMessage.findIndexOfMessageFromId({
      beforeMessage: _beforeMessage,
      afterMessage: _afterMessage,
      id: span.endId,
      messages: originalMessages,
    });
    const originalMessagesNeighborsAfter = originalMessages
      ?.slice(
        // span.endIndex + 1,
        // span.endIndex + 1 + neighborsToAdd
        messageIndexAfterStart + 1, // Start from the message right after the end of the span
        span.endIndex + 1 + neighborsToAdd, // Add a few more messages after the end of the span for context
      )
      .filter(Boolean);

    let contextualSystemInstruction = `### Messages Content For Contextual Aid
  - The following provides the message content within the span you are analyzing for the turning point.
  - Use this information to help you analyze the turning point and provide a more informed response (e.g. to identify quotes, and/or keywords, etc).
  - Possibly also included are messages before the turning point, and messages after the turning point, this is meant as broader contextual info.
    `;

    // Add context before the turning point
    if (originalMessagesNeighborsBefore?.length) {
      contextualSystemInstruction +=
        originalMessagesNeighborsBefore.length > 0
          ? `\n### Messages Before As Context\n` +
          originalMessagesNeighborsBefore
            .map((m) =>
              returnFormattedMessageContent(this.config, m, dimension),
            )
            .join("\n\n")
          : `\n### There does not exist any messages before this span of messages that encompass a potential ${dimension > 0 ? "meta turning point to formulate based on a grouping of turning points that encapuslate a single conversation" : "a potential turning point of two messages (that are part of a bigger single converation) being analyzed as provided in the user message content."}.\n`;
    }

    // Add context after the turning point
    if (originalMessagesNeighborsAfter?.length) {
      contextualSystemInstruction +=
        `\n### Messages After As Context\n` +
        originalMessagesNeighborsAfter
          .map((m) => {
            return returnFormattedMessageContent(this.config, m, dimension);
          })

          .join("\n\n");
    }

    // Add context within the turning point span or with the two messages if dimension is 0
    if (addMessagesWithinSpan && originalMessagesNeighborsAfter?.length) {
      contextualSystemInstruction +=
        `\n### Messages Within this potential turning point of the two messages below\n` +
        [_beforeMessage, _afterMessage]
          .map((m) => {
            return returnFormattedMessageContent(this.config, m, dimension);
          })
          .join("\n\n");
    }

    return contextualSystemInstruction;
  }

  /**
   * Validates turning point categories configuration
   * Ensures each category has required fields and logs warnings for issues
   */
  private validateTurningPointCategories(
    categories?: TurningPointCategory[],
  ): TurningPointCategory[] {
    // If no categories provided, use defaults
    if (!categories || categories.length === 0) {
      return turningPointCategories;
    }

    // Check if too many categories
    if (categories.length > 15) {
      this.logger.warn(
        `Warning: ${categories.length} turning point categories provided. ` +
        `Maximum recommended is 15. Consider reducing for better LLM performance.`,
      );
    }

    const validatedCategories: TurningPointCategory[] = [];
    const seenCategories = new Set<string>();
    const warningClose = `Proceeding anyway - if this was intentional for testing purposes, you can ignore this warning.`;
    categories.forEach((categoryConfig, index) => {
      // Check if category config exists and is an object
      if (!categoryConfig || typeof categoryConfig !== "object") {
        this.logger.warn(
          `Warning: Invalid category configuration at index ${index}. ` +
          `Expected object with 'category' and 'description' properties but found ${JSON.stringify(categoryConfig)}. ${warningClose}`,
        );
      }

      const { category, description } = categoryConfig;

      // Validate category field exists and is a string
      if (
        !category ||
        typeof category !== "string" ||
        category.trim().length === 0
      ) {
        this.logger.warn(
          `Warning: Missing or invalid 'category' field at index ${index}. ` +
          `Expected non-empty string. Using anyway with fallback value "unknown". ${warningClose}`,
        );
      }

      // Validate description field exists and is a string
      if (
        !description ||
        typeof description !== "string" ||
        description.trim().length === 0
      ) {
        this.logger.warn(
          `Warning: Missing or invalid 'description' field for category "${category}" at index ${index}. ` +
          `Expected non-empty string, but got ${JSON.stringify(description)}. Using anyway with fallback. ${warningClose}`,
        );
      }

      const trimmedCategory = category?.trim() || "unknown";
      const trimmedDescription =
        description?.trim() || "[no description provided]";

      // Check for duplicate categories (case-insensitive)
      const categoryLower = trimmedCategory.toLowerCase();
      if (seenCategories.has(categoryLower)) {
        this.logger.warn(
          `Warning: Duplicate category "${trimmedCategory}" found at index ${index}. ` +
          `Categories should be unique. Using anyway. ${warningClose}`,
        );
      }

      // Check if category name has more than two words
      const wordCount = trimmedCategory.split(/\s+/).length;
      if (wordCount > 2) {
        this.logger.warn(
          `Warning: Category "${trimmedCategory}" at index ${index} has ${wordCount} words. ` +
          `Consider using 1-2 words for better categorization. Using anyway. ${warningClose}`,
        );
      }

      seenCategories.add(categoryLower);

      validatedCategories.push({
        category: trimmedCategory,
        description: trimmedDescription,
      });


    });
    // Log what we're using
    if (this.config?.debug) {
      this.logger.info(
        `Using ${validatedCategories.length} turning point categories: ${validatedCategories.map((c) => c.category).join(", ")}`,
      );
    }

    return validatedCategories;
  }
  /**
   * Check if an endpoint is running Ollama by checking the root response
   */
  private async isOllamaEndpoint(endpoint: string): Promise<boolean> {
    try {
      // Remove trailing /v1 or other paths to get base URL
      const baseUrl = endpoint.replace(/\/v1\/?$/, "").replace(/\/$/, "");

      const response = await fetch(baseUrl, {
        method: "GET",
      });

      if (response.ok) {
        const text = await response.text();
        return text.includes("Ollama is running");
      }

      return false;
    } catch (error) {
      // If fetch fails, assume it's not Ollama
      return false;
    }
  }

  private calculateDynamicOverlapThreshold(
    dimension: number,
    spanSize: number,

    phiScore?: number | null,
  ): number {
    const baseThreshold = this.config.overlapThreshold || 0.6; // Higher default

    // Dimension scaling
    const dimensionScaleFactor = 1 + dimension * 0.15;

    // Span size scaling
    const spanSizeScaleFactor = Math.min(1.2, 1 + (spanSize / 100) * 0.1);

    // Phi-aware bonus for important turning points
    const phiBonus =
      phiScore && phiScore > 0.7
        ? 0.15 // Give high-phi turning points more overlap allowance
        : 0;

    // Combined scaling with higher bounds
    const dynamicThreshold = Math.min(
      0.9,
      baseThreshold * dimensionScaleFactor * spanSizeScaleFactor + phiBonus,
    );

    return Math.max(0.5, dynamicThreshold);
  }

  private fuseExpertsWithChoquet(
    turningPoints: TurningPoint[],
    capacityFunction: (subset: number[]) => number,
  ): TurningPoint[] {
    if (turningPoints.length === 0) return [];

    // Sort by epistemic support score (descending)
    const sorted = turningPoints
      .map((tp, idx) => ({
        tp,
        originalIdx: idx,
        score: tp.supportScore || tp.significance,
      }))
      .sort((a, b) => b.score - a.score);

    // Calculate Choquet integral value for the entire set
    let choquetValue = 0;

    for (let i = 0; i < sorted.length; i++) {
      const currentScore = sorted[i].score;
      const nextScore = i < sorted.length - 1 ? sorted[i + 1].score : 0;
      const delta = currentScore - nextScore;

      // Subset of indices for items with score >= currentScore
      const subset = sorted.slice(0, i + 1).map((item) => item.originalIdx);
      const capacity = capacityFunction(subset);

      choquetValue += delta * capacity;
    }

    // Now use the Choquet value to determine selection threshold
    const avgScore =
      sorted.reduce((sum, item) => sum + item.score, 0) / sorted.length;
    const choquetNormalizedThreshold = (choquetValue / sorted.length) * 0.8; // 80% of normalized Choquet value

    // Select turning points that contribute meaningfully to the Choquet integral
    const selected: TurningPoint[] = [];
    let cumulativeContribution = 0;

    for (let i = 0; i < sorted.length; i++) {
      const currentScore = sorted[i].score;
      const nextScore = i < sorted.length - 1 ? sorted[i + 1].score : 0;
      const delta = currentScore - nextScore;

      const subset = sorted.slice(0, i + 1).map((item) => item.originalIdx);
      const capacity = capacityFunction(subset);
      const contribution = delta * capacity;

      cumulativeContribution += contribution;

      // Include if: high individual score OR significant contribution to Choquet integral
      if (
        currentScore >= avgScore ||
        contribution >= this.config.epistemicThreshold
      ) {
        selected.push(sorted[i].tp);
      }

      // Stop if we've captured most of the Choquet value
      if (cumulativeContribution >= choquetValue * 0.9) {
        break;
      }
    }

    this.logger.info(
      `Choquet fusion: ${choquetValue.toFixed(3)} integral value, ` +
      `selected ${selected.length}/${turningPoints.length} TPs ` +
      `(threshold: ${choquetNormalizedThreshold.toFixed(3)})`,
    );

    return selected;
  }
  private createPossibilisticGatingNetwork(
    dimensionTurningPoints: Map<number, TurningPoint[]>,
  ): TurningPoint[] {
    const experts = Array.from(dimensionTurningPoints.entries());

    if (experts.length === 0) return [];

    // Calculate support scores for each dimensional "expert"
    const expertSupport = experts.map(([dimension, tps]) => {
      if (tps.length === 0) return { dimension, support: 0, tps: [] };

      const avgCompatibility =
        tps.reduce(
          (sum, tp) => sum + (tp.epistemicPrimitives?.compatibility || 0),
          0,
        ) / tps.length;
      const avgNecessity =
        tps.reduce(
          (sum, tp) => sum + (tp.epistemicPrimitives?.necessity || 0),
          0,
        ) / tps.length;

      // Use geometric mean for more balanced expert assessment
      const support = Math.sqrt(avgCompatibility * avgNecessity);

      return { dimension, support, tps };
    });

    // Create a capacity function that favors higher dimensions and better support
    const capacityFunction = (subset: number[]): number => {
      if (subset.length === 0) return 0;

      const relevantExperts = subset
        .map((i) => expertSupport[i])
        .filter(Boolean);
      if (relevantExperts.length === 0) return 0;

      // Higher capacity for higher dimensions and better support
      const maxDimension = Math.max(...relevantExperts.map((e) => e.dimension));
      const avgSupport =
        relevantExperts.reduce((sum, e) => sum + e.support, 0) /
        relevantExperts.length;

      // Capacity increases with dimension and support quality
      const dimensionWeight = 1 + maxDimension * 0.2;
      const supportWeight = Math.min(1, avgSupport * 2); // Cap at 1

      return Math.min(1, dimensionWeight * supportWeight * 0.7); // Scale down to reasonable range
    };

    // Apply Choquet fusion across all turning points
    const allTurningPoints = experts.flatMap(([_, tps]) => tps);
    return this.fuseExpertsWithChoquet(allTurningPoints, capacityFunction);
  }

  private calculateCompatibility(a: Float32Array, b: Float32Array): number {
    // Cosine similarity is a good measure for compatibility
    return 1 - this.calculateSemanticDistance(a, b);
  }

  private calculateMaxRejection(
    tp: TurningPoint,
    evidence: Float32Array,
  ): number {
    // This is a placeholder. A real implementation would be more complex.
    // For now, let's assume rejection is the inverse of compatibility.
    if (!tp.embedding) return 1.0;
    return 1 - this.calculateCompatibility(tp.embedding, evidence);
  }

  private calculateEpistemicPrimitives(
    tp: TurningPoint,
    evidenceDistribution: Float32Array,
    allTurningPoints: TurningPoint[],
  ): EpistemicPrimitives {
    // Compatibility: how well TP aligns with evidence
    const compatibility = this.calculateSemanticDistance(
      tp.embedding,
      evidenceDistribution,
    );

    // Necessity: how non-falsifiable this TP is
    const necessity = 1 - this.calculateMaxRejection(tp, evidenceDistribution);

    // Possibility: maximum plausibility
    const possibility = Math.max(
      ...allTurningPoints.map((other) =>
        this.calculateThematicSimilarity(tp, other),
      ),
    );

    // Surprisal: unexpectedness
    const surprisal = -Math.log(compatibility * necessity + 0.001);

    return { compatibility, necessity, possibility, surprisal };
  }

  /**
   * Processes the raw string content received from an LLM classification call
   * and attempts to parse it into a structured JSON object.
   *
   * This method implements a robust parsing strategy:
   * 1. It first attempts a direct `JSON.parse()` on the input string.
   * 2. If direct parsing fails, it looks for a JSON object embedded within
   *    a markdown-style code block (e.g., ```json\n{...}\n```) and tries to parse that.
   * 3. If that also fails, it attempts a more lenient match for any string
   *    that starts with `{` and ends with `}` and tries to parse that.
   * 4. If all parsing attempts fail, or if the resulting object is empty,
   *    it returns a default "Parsing Error - Unclassified" object. This ensures
   *    that the subsequent processing steps always receive an object with expected
   *    (though potentially default) properties.
   *
   * The `span` parameter is used to provide a default `best_id` in the
   * fallback classification object if parsing fails, ensuring some contextual
   * link back to the original messages being analyzed.
   *
   * @param content The raw string content from the LLM response, expected to contain JSON.
   * @param span The MessageSpan object corresponding to the messages being classified.
   *             Used for providing a fallback `best_id` if parsing fails.
   * @returns An `any` object representing the parsed classification. This object
   *          will have a defined structure if parsing is successful, or a default
   *          error structure if all parsing attempts fail.
   */
  parseClassificationResponse(content: string, span: MessageSpan): any {
    let classification = {};
    try {
      classification = JSON.parse(content);
    } catch (err: any) {
      this.logger.info("Error parsing LLM response as JSON:", err.message);
      const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
      if (jsonMatch && jsonMatch[1]) {
        try {
          classification = JSON.parse(jsonMatch[1]);
          this.logger.info("Successfully extracted JSON from markdown block.");
        } catch (parseErr: any) {
          this.logger.info("Failed to parse extracted JSON:", parseErr.message);
          classification = {};
        }
      } else {
        const plainJsonMatch = content.match(/\{[\s\S]*\}/);
        if (plainJsonMatch) {
          try {
            classification = JSON.parse(plainJsonMatch[0]);
            this.logger.info("Successfully extracted JSON using simple match.");
          } catch (parseErr: any) {
            this.logger.info(
              "Failed to parse JSON using simple match:",
              parseErr.message,
            );
            classification = {};
          }
        } else {
          this.logger.info(
            "Could not extract JSON from response: " + JSON.stringify(content),
          );
          classification = {};
        }
      }
      if (Object.keys(classification).length === 0) {
        classification = {
          label: "Parsing Error - Unclassified",
          category: "Other",
          emotionalTone: "neutral",
          sentiment: "neutral",
          significance: 0.1,
          quotes: [],
          best_id: span.startId, // Use span from the calling context
        };
      }
    }
    return classification;
  }
}


/**
 * Example function demonstrating how to use the SemanticTurningPointDetector
 * Implements an adaptive approach based on conversation complexity
 */
async function runTurningPointDetectorExample() {
  const conversationPariah = fs.readJsonSync("src/conversationPariah.json", {
    encoding: "utf-8",
  }) as Message[];

  const startTime = new Date().getTime();

  // Example endpoint configuration - replace with your actual endpoint
  const endpoint = "http://api.openrouter.ai/v1";

  const config: Partial<TurningPointDetectorConfig> = {
    /**
     * API Key Configuration:
     * The detector will automatically select the appropriate API key based on your endpoint:
     * - For OpenRouter endpoints: Uses OPENROUTER_API_KEY environment variable
     * - For OpenAI endpoints: Uses OPENAI_API_KEY environment variable
     * - For custom endpoints: Falls back to LLM_API_KEY environment variable
     * 
     * You can also set the key directly here instead of using environment variables.
     */
    apiKey: endpoint.includes("openrouter.ai")
      ? process.env.OPENROUTER_API_KEY
      : process.env.OPENAI_API_KEY,

    // Core semantic detection parameters
    semanticShiftThreshold: 0.7, // Start with 0.55-0.7 range, then adjust based on your conversation type
    minTokensPerChunk: 1024,
    maxTokensPerChunk: 8192,
    significanceThreshold: 0.3,

    /**
     * ARC Framework Recursion Control:
     * This controls how deep the dimensional analysis goes (n → n+1 → n+2...).
     * Higher values detect more complex meta-patterns but increase processing time exponentially.
     * - Start with 3-5 for most use cases
     * - Lower values (1-2) reduce escalation and keep analysis at base dimensions
     * - Higher values (5+) enable detection of very complex narrative structures
     */
    maxRecursionDepth: 5,

    /**
     * Result Filtering Strategy:
     * This fundamentally changes how results are processed and returned:
     * - true: Apply significance filtering and respect maxTurningPoints limit (focused analysis)
     * - false: Return all detected turning points regardless of significance (comprehensive analysis)
     * 
     * For production use, set to true. For exploration and debugging, set to false.
     */
    onlySignificantTurningPoints: true,

    /**
     * Chunk Size Control for LLM Requests:
     * This affects request efficiency but not the actual turning point detection results.
     * Controls how many messages are grouped together when sending to the LLM:
     * - Smaller values = more requests with smaller payloads
     * - Larger values = fewer requests with bigger payloads
     * 
     * The actual request size is primarily governed by maxTokensPerChunk and minTokensPerChunk.
     * This provides secondary control for fine-tuning request patterns.
     */
    minMessagesPerChunk: 22,

    /**
     * Maximum Results Configuration:
     * For demonstration purposes, we're using a fixed value instead of dynamic scaling.
     * Alternative dynamic approach would be:
     * maxTurningPoints: Math.max(6, Math.round(conversationPariah.length / 20))
     */
    maxTurningPoints: 16,

    /**
     * Experimental Phi Feature - Enhanced Merging:
     * When enableExperimentalPhi is true, this multiplier controls how aggressively
     * phi-enhanced turning points are merged based on thematic similarity.
     * - Higher values (1.5+): More permissive merging of related turning points
     * - Lower values (0.8-1.2): Stricter separation of turning points
     */
    phiMergeThresholdMultiplier: 1.5,

    /**
     * CRA Framework - Complexity Saturation Threshold:
     * This triggers dimensional escalation (n → n+1) when turning point complexity
     * reaches this threshold. Controls when meta-analysis begins:
     * - Lower values (2-3): More aggressive escalation to higher dimensions
     * - Higher values (4-5): Conservative escalation, stays in current dimension longer
     */
    complexitySaturationThreshold: 3,

    /**
     * Concurrency Control:
     * Limits parallel LLM requests to prevent rate limiting and manage resource usage.
     * Recommended settings:
     * - OpenAI API: 4 or higher (good rate limits)
     * - Custom/local endpoints: 1-2 (may have stricter limits)
     * - OpenRouter: 2-4 (varies by provider)
     */
    concurrency: 2,

    /**
     * Message Content Truncation:
     * Prevents extremely long messages from causing context window overflow.
     * Messages longer than this will be truncated while preserving structure.
     */
    max_character_length: 4000,

    // ARC Framework - Convergence Analysis
    measureConvergence: true, // Tracks how turning point detection stabilizes across dimensions

    // Turning Point Overlap Detection
    overlapThreshold: 0.3, // Controls how much overlap is allowed between adjacent turning points

    // Model Configuration
    classificationModel: "google/gemini-2.5-flash",
    endpoint: endpoint,

    /**
     * Dynamic Threshold Adjustment:
     * When enabled, automatically lowers semantic shift thresholds in higher dimensions
     * to capture more potential turning points for meta-analysis.
     * - true: Adaptive thresholds that change based on dimension level
     * - false: Fixed threshold across all dimensions
     */
    dynamicallyAdjustSemanticShiftThreshold: true,

    /**
     * Experimental Phi Processing:
     * Enables the phi (φ) significance field for enhanced turning point analysis.
     * This creates emergent significance scoring that enhances base LLM classifications.
     * 
     * Note: This feature was added but the constructor wasn't initially updated to use it
     * until August 3rd, 1:49 PM CDT, yet earlier results showed improvement - this suggests
     * the base framework was already quite effective.
     */
    enableExperimentalPhi: true,

    /**
     * Custom Endpoint Configuration:
     * Setting a custom endpoint overrides the default api.openai.com/v1 endpoint.
     * This allows usage of other LLM providers that follow the OpenAI API structure.
     * 
     * IMPORTANT: The Semantic Turning Point Detector uses advanced parameters,
     * specifically the 'response_format' parameter for JSON Schema objects.
     * Not all OpenAI-compatible endpoints support this feature.
     * 
     * Verified compatible providers:
     * - Ollama (with JSON format support)
     * - OpenRouter (most models)
     * - vLLM (with structured output)
     * - LM Studio (with response formatting)
     * - Text Generation API (with JSON mode)
     * 
     * The detector does NOT use function/tool calls, only structured JSON responses.
     */

    // Embedding Configuration
    embeddingModel: "text-embedding-3-large",
    debug: false,

    /**
     * Dynamic Complexity Saturation:
     * When disabled (false), uses the fixed complexitySaturationThreshold value.
     * When enabled (true), calculates threshold dynamically based on the actual
     * complexity distribution of detected turning points.
     */
    enableDynamicComplexitySaturation: false,

    /**
     * Counterfactual Analysis:
     * Experimental feature that performs "what-if" analysis to validate turning points
     * by considering alternative interpretations and narrative paths.
     * Helps strengthen turning point selection by filtering out weak candidates.
     */
    enableCounterfactualAnalysis: true,
  };

  // Create detector with ARC/CRA framework configuration
  const detector = new SemanticTurningPointDetector(config);

  try {
    // Execute the ARC/CRA framework turning point detection
    const tokensInConvoFile = await detector.getMessageArrayTokenCount(conversationPariah);
    const turningPointResult = await detector.detectTurningPoints(conversationPariah);

    const turningPoints = turningPointResult.points;
    const confidenceScore = turningPointResult.confidence;
    const necessityScore = turningPointResult.necessity; // Fixed typo: was "neccesityScore"

    const endTime = new Date().getTime();
    const difference = endTime - startTime;
    const formattedTimeDateDiff = new Date(difference).toISOString().slice(11, 19);

    console.info(
      `\nTurning point detection completed in ${formattedTimeDateDiff} (MM:SS) for ${tokensInConvoFile} tokens`,
    );

    // Display results with ARC framework complexity scores
    console.info("\n=== DETECTED TURNING POINTS (ARC/CRA Framework) ===\n");
    console.info(
      `Detected ${turningPoints.length} turning points using model ${detector.getModelName()}`,
    );

    // Display aggregate confidence and necessity scores
    console.info(
      `Confidence Score (softmax): ${confidenceScore?.toFixed(3)} | ${config.enableExperimentalPhi ? "Necessity Score: " + necessityScore?.toFixed(3) : ""
      }`,
    );

    turningPoints.forEach((tp, i) => {
      detector.logger.info(`\n${i + 1}. ${tp.label} (${tp.category})`);
      detector.logger.info(`   Messages: "${tp.span.startId}" → "${tp.span.endId}"`);
      detector.logger.info(`   Dimension: n=${tp.detectionLevel}`);
      detector.logger.info(`   Complexity Score: ${tp.complexityScore.toFixed(2)} of 5`);
      detector.logger.info(`   Emotional Tone: ${tp.emotionalTone || "unknown"}`);
      detector.logger.info(`   Semantic Shift Magnitude: ${tp.semanticShiftMagnitude.toFixed(2)}`);
      detector.logger.info(`   Sentiment: ${tp.sentiment || "unknown"}`);
      detector.logger.info(`   Significance: ${tp.significance.toFixed(2)}`);
      detector.logger.info(`   Quotes: ${tp.quotes?.join(", ") || "none"}`);
    });

    // Display ARC framework convergence analysis
    const convergenceHistory = detector.getConvergenceHistory();
    detector.logger.info("\n=== ARC/CRA FRAMEWORK CONVERGENCE ANALYSIS ===\n");

    convergenceHistory.forEach((state, i) => {
      detector.logger.info(`Iteration ${i + 1}:`);
      detector.logger.info(`  Dimension: n=${state.dimension}`);
      detector.logger.info(`  Convergence Distance: ${state.distanceMeasure.toFixed(3)}`);
      detector.logger.info(`  Dimensional Escalation: ${state.didEscalate ? "Yes" : "No"}`);
      detector.logger.info(`  Turning Points: ${state.currentTurningPoints.length}`);
    });

    // Save results for further analysis
    fs.writeJSONSync("results/turningPoints.json", turningPoints, {
      spaces: 2,
      encoding: "utf-8",
    });

    fs.writeJSONSync("results/convergence_analysis.json", convergenceHistory, {
      spaces: 2,
      encoding: "utf-8",
    });

    detector.logger.info("Results saved to files.");
  } catch (err) {
    detector.logger.error(
      `Error detecting turning points: ${err.message}\nStack trace: ${(err as Error)?.stack}`,
    );
  }
}
if (require.main === module) {
  runTurningPointDetectorExample().finally(() => {
    process.exit(0);
  });
}
