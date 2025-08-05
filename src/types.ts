// -----------------------------------------------------------------------------
// Embedding Generation
// -----------------------------------------------------------------------------

import type winston from "winston";
import type { Message, MessageSpan } from "./Message";

// -----------------------------------------------------------------------------
// Core Interfaces
// -----------------------------------------------------------------------------

/**
 * Represents a single message in a conversation
 * This is used to track the content, author, and metadata of each message
 */

/**
 * Represents a semantic turning point in a conversation
 * This corresponds to a significant semantic shift detected by the system
 */
export interface TurningPoint {
  /** Unique identifier for this turning point */
  id: string;
  /** Human-readable short description of what this turning point represents */
  label: string;
  /** The type of semantic shift this turning point represents */
  category: string;
  /** The span of messages covered by this turning point */
  span: MessageSpan;
  /** Legacy span format no longer utilized due to new class instantiations for MetaMessages */
  deprecatedSpan?: {
    startIndex: number;
    endIndex: number;
    startMessageId: string;
    endMessageId: string;
  };
  /** The semantic distance/shift that triggered this turning point */
  semanticShiftMagnitude: number;
  /** Key terms that characterize this turning point */
  keywords?: string[];
  /** Notable quotes from the messages in this turning point's span */
  quotes?: string[];
  /** The emotionality of this turning point if applicable */
  emotionalTone?: string;
  /**
   * The dimension at which this turning point was detected.
   * If detectionLevel > 0, it indicates that this turning point was analyzed based on a span of turning points, rather than a span of messages.
   */
  detectionLevel: number;
  /** Significance score (higher = more significant) */
  significance: number;

  /** An assessed best point representing the turning point */
  /** The complexity score (1-5) representing saturation in current dimension */
  complexityScore: number;

  /**
   * A potential label assigned by the LLM, which can be either 'positive' or 'negative'.
   * However, this label is not definitive and may be improved using a zero-shot model,
   * based on the classification provided by the LLM.
   */
  sentiment?: string;

  /**
   * @experimental
   * Used for Non-Additive Fusion for Dimensional Escalation
   */
  supportScore?: number;

  /**
   * @experimental
   */
  phi?: number; // Experimental significance score for the turning point, used in the Choquet integral necessity calculation, only used if `enableExperimentalPhi` is set to true in the configuration.

  /**
   * An embedding vector representing the semantic content of this turning point, in which is th emebedding of the messages in the span of this turning point, up to the max length supported by the embedding model. By default this is 8000 tokens.
   */
  embedding?: Float32Array;

  /* @experimental
   * The epistemic primitives for this turning point.
   */
  epistemicPrimitives?: EpistemicPrimitives;
}

/**
 * Type defintion for a single category utilized as part of instructions for the LLM in analyzing turning points.
 */
export type TurningPointCategory = {
  /**
   * A single word, recommended to be one-word, that describes the category of the turning point, and is distinct from any other category.
   */
  category: string;
  /**
   * A short one sentence description of the category of what this represents, usually a definition suffices.
   */
  description: string;
};
/**
 * Configuration options for the turning point detector.
 * - Detailed descriptions for each option are provided below.
 */
/**
 * Configuration options for the turning point detector.
 *
 * This interface controls all aspects of the ARC/CRA semantic analysis framework,
 * from basic LLM settings to advanced experimental features like phi-aware processing
 * and epistemic primitives calculation.
 */
export interface TurningPointDetectorConfig {
  /**
   * Configurable turning point categories with descriptions.
   *
   * These categories guide the LLM in classifying detected semantic shifts.
   * Well-defined categories significantly improve classification accuracy and consistency.
   *
   * @remarks
   * - Maximum of 15 categories allowed (default: 11 categories)
   * - Categories should be mutually exclusive and clearly defined
   * - Each category needs both a concise name and descriptive definition
   * - More categories provide finer granularity but may confuse the LLM
   * - Fewer categories ensure cleaner classification but lose nuance
   *
   * @example
   * ```typescript
   * turningPointCategories: [
   *   { category: "Topic", description: "A shift to a new subject or theme" },
   *   { category: "Insight", description: "A realization, discovery, or moment of understanding" },
   *   { category: "Decision", description: "A choice or resolution being made" }
   * ]
   * ```
   *
   * @recommendation Use default categories initially, then customize based on your domain
   */
  turningPointCategories: TurningPointCategory[];

  /**
   * API Key for LLM requests.
   *
   * Required for all LLM-based analysis operations. The detector will first check this value,
   * then fall back to environment variables based on your endpoint configuration.
   *
   * @environment
   * - For custom endpoints: Set `LLM_API_KEY` environment variable
   * - For OpenAI (default): Set `OPENAI_API_KEY` environment variable
   *
   * @security Store API keys securely and never commit them to version control
   */
  apiKey: string;

  /**
   * The LLM model used for turning point analysis.
   *
   * Must be available on your configured endpoint and support structured JSON output.
   * Model choice significantly affects analysis quality, speed, and cost.
   *
   * @recommendations
   * - For balanced performance: "gpt-4o-mini" or "claude-3-haiku"
   * - For highest quality: "gpt-4" or "claude-3-opus"
   * - For speed/cost: "gpt-3.5-turbo" (lower accuracy)
   * - For local deployment: Compatible Ollama models with JSON support
   *
   * @performance Larger models provide better semantic understanding but increase latency and cost
   */
  classificationModel: string;

  /**
   * Temperature setting for LLM model (0.0-1.0).
   *
   * Controls randomness in LLM responses. Lower values produce more consistent,
   * deterministic outputs, which is crucial for reliable JSON schema compliance.
   *
   * @default 0.6
   * @range 0.0-1.0
   * @warning Values >0.8 may cause JSON parsing errors
   * @recommendation Keep at 0.6 or lower for production use
   */
  temperature?: number;

  /**
   * Top-p (nucleus sampling) setting for LLM model (0.0-1.0).
   *
   * Controls the diversity of token selection by limiting the cumulative probability mass.
   * Lower values increase consistency but may reduce semantic richness.
   *
   * @default 1.0
   * @range 0.0-1.0
   * @warning Values <0.8 may cause repetitive or incomplete JSON responses
   * @recommendation Use 0.9-1.0 for best balance of diversity and reliability
   */
  top_p?: number;

  /**
   * Model for generating embeddings.
   *
   * Embeddings are crucial for semantic distance calculations that drive turning point detection.
   * Model choice affects both accuracy and computational cost.
   *
   * @recommendations
   * - For OpenAI: "text-embedding-3-small" (balanced), "text-embedding-3-large" (accuracy)
   * - For custom endpoints: Any OpenAI-compatible embedding model
   * - Ensure model supports sufficient context length for your message chunks
   *
   * @performance Larger embedding models provide better semantic representation but slower processing
   */
  embeddingModel: string;

  /**
   * Custom endpoint for embedding generation.
   *
   * Allows use of alternative embedding providers while maintaining OpenAI API compatibility.
   * Useful for cost optimization, privacy requirements, or local deployment.
   *
   * @examples
   * - Local: "http://localhost:1234/v1" (LM Studio)
   * - Hosted: "https://api.openrouter.ai/api/v1"
   * @environment Set `EMBEDDINGS_API_KEY` for external providers
   * @default undefined (uses OpenAI)
   */
  embeddingEndpoint?: string;

  /**
   * RAM limit for embedding cache in MB.
   *
   * Controls memory usage by caching embeddings to avoid recomputation.
   * Higher values improve performance but consume more memory.
   *
   * @default 256
   * @recommendation
   * - For large conversations: 512-1024MB
   * - For memory-constrained environments: 128-256MB
   * - For batch processing: 1024MB+
   * @note Increase Node.js memory limit with --max-old-space-size if needed
   */
  embeddingCacheRamLimitMB?: number;

  /**
   * Threshold for detecting semantic shifts between messages (0.0-1.0).
   *
   * This is the primary control for turning point sensitivity. Higher values detect
   * only major semantic changes, lower values capture subtle shifts.
   *
   * @range 0.0-1.0
   * @recommendations
   * - Technical/focused discussions: 0.5-0.7 (detect major shifts)
   * - Casual conversations: 0.3-0.5 (capture subtle changes)
   * - Exploratory analysis: 0.2-0.4 (maximum sensitivity)
   *
   * @behavior
   * - Automatically scaled down in higher dimensions (meta-analysis)
   * - Can be dynamically adjusted if `dynamicallyAdjustSemanticShiftThreshold` is enabled
   *
   * @tuning Start with 0.4 and adjust based on results - too high misses shifts, too low creates noise
   */
  semanticShiftThreshold: number;

  /**
   * Minimum token count for conversation chunks.
   *
   * Prevents chunks too small for meaningful semantic analysis while balancing
   * processing efficiency and context preservation.
   *
   * @recommendations
   * - General use: 300-500 tokens
   * - Technical content: 400-600 tokens (preserve complex context)
   * - Short messages: 200-350 tokens (avoid over-chunking)
   *
   * @relationship Works with `minMessagesPerChunk` - both conditions must be met
   * @scaling Automatically reduced in higher dimensions for meta-message analysis
   */
  minTokensPerChunk: number;

  /**
   * Maximum token count for conversation chunks.
   *
   * Prevents context window overflow while maximizing available context for analysis.
   * Should account for prompt overhead and model limitations.
   *
   * @recommendations
   * - GPT-4: 6000-8000 tokens (allows prompt overhead)
   * - GPT-3.5: 3000-4000 tokens
   * - Claude: 8000-12000 tokens
   * - Custom models: Check context window and adjust accordingly
   *
   * @relationship Must be significantly larger than `minTokensPerChunk`
   * @scaling Automatically reduced proportionally in higher dimensions
   */
  maxTokensPerChunk: number;

  /**
   * Maximum dimension level for ARC framework recursion.
   *
   * Controls the depth of meta-analysis performed on conversations.
   * Higher dimensions detect complex narrative patterns but increase processing time exponentially.
   *
   * @dimensions
   * - Dimension 0: Direct message analysis
   * - Dimension 1: Analysis of turning point patterns
   * - Dimension 2: Meta-patterns in turning point relationships
   * - Dimension 3+: Higher-order thematic and structural analysis
   *
   * @recommendations
   * - Most conversations: 2-3 levels
   * - Complex narratives: 3-4 levels
   * - Simple analysis: 1-2 levels
   *
   * @performance Each dimension roughly doubles processing time and cost
   * @note Actual escalation only occurs when complexity saturation is reached
   */
  maxRecursionDepth: number;

  /**
   * Minimum significance score for final results (0.0-1.0).
   *
   * Filters turning points based on computed Choquet integral necessity scores.
   * Acts as a quality gate to ensure only meaningful turning points are returned.
   *
   * @range 0.0-1.0 (probability-like scale)
   * @interpretation
   * - >0.7: High significance (major narrative moments)
   * - 0.5-0.7: Moderate significance (notable shifts)
   * - 0.3-0.5: Low significance (subtle changes)
   * - <0.3: Minimal significance (may be noise)
   *
   * @recommendations
   * - Strict filtering: 0.6-0.7
   * - Balanced analysis: 0.4-0.5
   * - Exploratory analysis: 0.2-0.3
   *
   * @interaction Only applies when `onlySignificantTurningPoints` is true
   */
  significanceThreshold: number;

  /**
   * Controls turning point filtering strategy and result prioritization.
   *
   * Fundamentally changes how results are filtered, ordered, and returned.
   * Choose based on your analysis goals and downstream processing requirements.
   *
   * @mode true (Focused Analysis)
   * - Enforces `significanceThreshold` filtering
   * - Limits results to `maxTurningPoints`
   * - Orders by significance score (highest first)
   * - Best for: Comparative analysis, key moment extraction, summarization
   *
   * @mode false (Comprehensive Analysis)
   * - Returns ALL detected turning points
   * - Ignores `maxTurningPoints` limit
   * - Orders chronologically by conversation position
   * - Best for: Detailed exploration, conversation mapping, research analysis
   *
   * @recommendation Use `true` for production applications, `false` for analysis and debugging
   */
  onlySignificantTurningPoints: boolean;

  /**
   * Minimum messages required per chunk.
   *
   * Ensures chunks contain sufficient conversational context for semantic analysis.
   * Prevents over-fragmentation of short message sequences.
   *
   * @recommendations
   * - Standard conversations: 3-5 messages
   * - Long-form discussions: 2-3 messages
   * - Chat-style rapid exchanges: 5-8 messages
   *
   * @relationship Works with `minTokensPerChunk` - both conditions must be met
   */
  minMessagesPerChunk: number;

  /**
   * Maximum turning points returned in final results.
   *
   * Controls output size for focused analysis mode. Helps manage downstream
   * processing load and maintains focus on most significant shifts.
   *
   * @recommendations
   * - Conversation summaries: 5-10 turning points
   * - Detailed analysis: 15-25 turning points
   * - Comprehensive mapping: 30+ turning points
   *
   * @note Only applies when `onlySignificantTurningPoints` is true
   * @interaction Combined with `significanceThreshold` for quality control
   */
  maxTurningPoints: number;

  /**
   * Enable verbose logging for debugging and monitoring.
   *
   * Provides detailed insights into the analysis process, performance metrics,
   * and decision points. Essential for tuning and troubleshooting.
   *
   * @output
   * - Chunking statistics and decisions
   * - Embedding generation and caching
   * - Semantic distance calculations
   * - Turning point detection rationale
   * - Dimensional escalation triggers
   * - Performance timing and memory usage
   *
   * @recommendation Enable during development and tuning, disable in production
   */
  debug: boolean;

  /**
   * Custom LLM endpoint for analysis requests.
   *
   * Enables use of alternative LLM providers, local models, or specialized endpoints
   * while maintaining OpenAI API compatibility.
   *
   * @requirements
   * - Must support OpenAI-compatible chat completions API
   * - Must support structured output (JSON schema format parameter)
   * - Should handle temperature and top_p parameters
   *
   * @examples
   * - "http://localhost:1234/v1" (LM Studio)
   * - "https://api.openrouter.ai/api/v1" (OpenRouter)
   * - "http://localhost:11434/v1" (Ollama with OpenAI compatibility)
   *
   * @testing Verify JSON schema support before production use
   */
  endpoint?: string;

  /**
   * Complexity saturation threshold for dimensional escalation (1.0-5.0).
   *
   * Triggers escalation to higher dimensions when turning points show high complexity.
   * Controls the sensitivity of the ARC framework's dimensional transitions.
   *
   * @range 1.0-5.0 (complexity score scale)
   * @recommendations
   * - Conservative escalation: 4.0-4.5
   * - Balanced analysis: 3.5-4.0
   * - Aggressive escalation: 3.0-3.5
   *
   * @mechanism When average complexity exceeds threshold, meta-analysis begins
   * @dynamic Can be automatically adjusted if `enableDynamicComplexitySaturation` is enabled
   */
  complexitySaturationThreshold: number;

  /**
   * Enable convergence measurement across ARC iterations.
   *
   * Tracks how turning point detection stabilizes across dimensional escalations.
   * Useful for understanding analysis convergence and optimizing parameters.
   *
   * @metrics
   * - Distance between turning point sets across iterations
   * - Stability of semantic patterns
   * - Dimensional escalation effectiveness
   *
   * @overhead Minimal performance impact, recommended for analysis workflows
   */
  measureConvergence: boolean;

  /**
   * Custom system instruction injection.
   *
   * Allows fine-tuning LLM behavior for domain-specific analysis requirements.
   * Inserted after contextual aid text but before main system prompt.
   *
   * @warning Advanced feature - improper use can break JSON schema compliance
   * @examples
   * - "Focus on technical terminology shifts"
   * - "Prioritize emotional tone changes"
   * - "Consider domain-specific context"
   *
   * @recommendation Test thoroughly before production use
   */
  customSystemInstruction?: string;

  /**
   * Custom user message injection.
   *
   * Modifies the user prompt structure for specialized analysis approaches.
   * Allows customization of how conversation content is presented to the LLM.
   *
   * @warning Advanced feature - can break analysis if misused
   * @use_case Domain-specific prompting strategies
   * @recommendation Only use if default prompting is insufficient
   */
  customUserInstruction?: string;

  /**
   * Maximum characters per message for analysis context.
   *
   * Truncates very long messages to prevent context window overflow while
   * preserving essential semantic content.
   *
   * @default 8000 characters (approximately 2000 tokens)
   * @recommendations
   * - Standard analysis: 6000-8000 characters
   * - Long-form content: 10000-12000 characters
   * - Memory-constrained: 4000-6000 characters
   *
   * @behavior Truncation preserves message structure and key content
   */
  max_character_length?: number;

  /**
   * Custom logger instance for output control.
   *
   * Allows integration with existing logging infrastructure and custom output formatting.
   *
   * @accepts winston.Logger or Console interface
   * @default console (when debug=true)
   * @integration Useful for structured logging and monitoring systems
   */
  logger?: winston.Logger | Console;

  /**
   * Control error handling behavior during analysis.
   *
   * Determines whether to halt processing on LLM errors or continue with graceful degradation.
   *
   * @mode true: Fail fast on errors (debugging/development)
   * @mode false: Continue with degraded results (production resilience)
   *
   * @considerations
   * - Single chunk failures don't invalidate entire analysis
   * - Failed analyses are treated as "no turning point detected"
   * - Enable for development to catch configuration issues
   */
  throwOnError?: boolean;

  /**
   * Parallel processing concurrency for LLM analysis requests.
   *
   * Controls how many analysis requests are processed simultaneously.
   * Balances speed against rate limits and resource consumption.
   *
   * @recommendations
   * - OpenAI API: 3-8 concurrent requests
   * - OpenRouter/commercial: 5-10 concurrent requests
   * - Custom endpoints: 1-3 concurrent requests
   * - Local models: 1-2 concurrent requests
   *
   * @limits Most APIs have rate limits around 60 RPM (requests per minute)
   * @scaling Higher concurrency increases speed but may hit rate limits
   */
  concurrency?: number;

  /**
   * Parallel processing concurrency for embedding generation.
   *
   * Embeddings are typically faster and less resource-intensive than LLM analysis,
   * allowing higher concurrency levels.
   *
   * @default 5
   * @recommendations
   * - OpenAI embeddings: 8-15 concurrent requests
   * - Custom embedding endpoints: 3-8 concurrent requests
   * - Local embedding models: 2-5 concurrent requests
   *
   * @performance Embeddings have lower latency than chat completions
   */
  embeddingConcurrency?: number;

  /**
   * @experimental
   * Enable phi-aware significance scoring.
   *
   * Integrates an experimental third epistemic axis (φ) into significance calculations.
   * Enhances turning point selection with emergent thematic importance weighting.
   *
   * @mechanism
   * - Computes phi scores based on thematic coherence and narrative position
   * - Uses phi as fuzzy capacity in Choquet integral necessity calculation
   * - Balances structural complexity with existential significance
   *
   * @stability Experimental - subject to algorithmic changes
   * @recommendation Use false for production, true for research/experimentation
   * @performance Adds ~15-20% computational overhead
   */
  enableExperimentalPhi?: boolean;

  /**
   * Enable dynamic semantic shift threshold adjustment.
   *
   * Automatically lowers semantic shift threshold in lower dimensions to capture
   * more potential turning points for higher-dimensional meta-analysis.
   *
   * @behavior
   * - Dimension 0: Uses configured threshold
   * - Dimension 1+: Progressively lowers threshold
   * - Facilitates richer meta-pattern detection
   *
   * @recommendation Enable for complex narrative analysis, disable for focused detection
   */
  dynamicallyAdjustSemanticShiftThreshold?: boolean;

  /**
   * Enable statistical complexity saturation adjustment.
   *
   * Dynamically adjusts complexity saturation threshold based on observed
   * turning point complexity distribution rather than fixed values.
   *
   * @mechanism
   * - Analyzes complexity score distribution
   * - Adjusts threshold to target specific percentile
   * - Improves adaptive behavior across different conversation types
   *
   * @benefit Reduces manual parameter tuning for diverse content
   */
  enableDynamicComplexitySaturation?: boolean;

  /**
   * Target percentile for dynamic complexity saturation (0.0-1.0).
   *
   * When dynamic adjustment is enabled, sets the target percentage of turning points
   * that should trigger complexity saturation and dimensional escalation.
   *
   * @default 0.15 (15% of turning points trigger escalation)
   * @range 0.05-0.30
   * @recommendations
   * - Conservative escalation: 0.10-0.15
   * - Balanced escalation: 0.15-0.20
   * - Aggressive escalation: 0.20-0.30
   */
  dynamicSaturationTargetPercentile?: number;

  /**
   * Minimum samples required for dynamic complexity adjustment.
   *
   * Ensures sufficient data before statistical adjustment kicks in.
   * Prevents premature optimization from small sample sizes.
   *
   * @default 10
   * @recommendation 8-15 for most use cases
   * @behavior Falls back to fixed threshold until minimum reached
   */
  dynamicSaturationMinSamples?: number;

  /**
   * Phi merge threshold multiplier for experimental phi processing.
   *
   * Controls how aggressively phi-enhanced turning points are merged or separated.
   * Higher values require stronger phi coherence for merging operations.
   *
   * @experimental Used only when `enableExperimentalPhi` is true
   * @range 0.8-2.0
   * @default 1.5
   * @effect Higher values = fewer, more distinct turning points
   */
  phiMergeThresholdMultiplier?: number;

  /**
   * @experimental
   * Enable counterfactual analysis for turning point validation.
   *
   * Performs "what-if" analysis to strengthen turning point selection by
   * considering alternative interpretations and narrative paths.
   *
   * @mechanism
   * - Generates counterfactual scenarios for detected turning points
   * - Evaluates robustness of turning point significance
   * - Filters points that don't survive counterfactual testing
   *
   * @stability Experimental feature under active development
   * @performance Approximately doubles analysis time
   */
  enableCounterfactualAnalysis?: boolean;

  /**
   * Overlap threshold for turning point boundary detection (0.0-1.0).
   *
   * Controls how much overlap is allowed between adjacent turning point spans
   * before they are considered for merging or boundary adjustment.
   *
   * @range 0.0-1.0
   * @recommendations
   * - Strict separation: 0.1-0.2
   * - Balanced overlap: 0.3-0.4
   * - Permissive overlap: 0.5-0.6
   *
   * @interaction Works with epistemic filtering for boundary refinement
   */
  overlapThreshold?: number;

  /**
   * Epistemic threshold for primitive-based filtering (0.0-1.0).
   *
   * Minimum epistemic score required for turning points to pass
   * compatibility and necessity assessment in the possibilistic framework.
   *
   * @range 0.0-1.0
   * @interpretation Probability-like measure of epistemic validity
   * @recommendation 0.01-0.03 for balanced filtering (removes ~5-15% of weak points)
   */
  epistemicThreshold?: number;

  /**
   * Epistemic primitives configuration for advanced filtering.
   *
   * Defines the epistemic assessment criteria used in possibilistic
   * turning point evaluation and selection.
   *
   * @components
   * - compatibility: Evidence alignment scoring
   * - necessity: Non-falsifiability assessment
   * - possibility: Maximum plausibility calculation
   * - surprisal: Unexpectedness measurement
   *
   * @advanced Used by epistemic primitive calculation engine
   */
  epistemicPrimitives?: EpistemicPrimitives;

  /**
   * Support score threshold for experimental features.
   *
   * Used in conjunction with experimental epistemic and phi processing
   * to control the minimum support required for turning point validation.
   *
   * @experimental Subject to change as epistemic framework evolves
   * @range 0.0-1.0
   * @interaction Combined with other experimental thresholds
   */
  supportScore?: number;
}

/**
 * Default turning point categories with descriptions
 */
export const turningPointCategories: TurningPointCategory[] = [
  {
    category: "Topic",
    description:
      "This category is for content that is primarily focused on a specific area, domain, or subject. Use this when the content warrants categorization by topic.",
  },
  {
    category: "Insight",
    description:
      "This category applies to content that provides a unique insight or perspective. Use this when the content warrants categorization by insight.",
  },
  {
    category: "Emotion",
    description:
      "This category is for content that holds significant emotional impact. Use this when the content warrants categorization by emotion.",
  },
  {
    category: "Meta-Reflection",
    description:
      "This category applies to content that reflects on the conversation or interaction. Use this when the content warrants categorization by meta-reflection.",
  },
  {
    category: "Decision",
    description:
      "This category is for content that involves a decision or choice that has been made. Use this when the content warrants categorization by decision.",
  },
  {
    category: "Question",
    description:
      "This category applies to content that poses a question or inquiry. Use this when the content warrants categorization by question.",
  },
  {
    category: "Problem",
    description:
      "This category is for content that presents a problem or issue. Use this when the content warrants categorization by problem.",
  },
  {
    category: "Action",
    description:
      "This category applies to content that involves an action or activity, or serves as a call to action. Use this when the content warrants categorization by action.",
  },
  {
    category: "Clarification",
    description:
      "This category is for content that seeks or provides clarification. Use this when the content warrants categorization by clarification.",
  },
  {
    category: "Objection",
    description:
      "This category applies to content that expresses an objection or disagreement. Use this when the content warrants categorization by objection.",
  },
  {
    category: "Other",
    description:
      "This category applies to any other significant conversational shift that doesn't fit the above categories.",
  },
];

/**
 * Chunking result with message segments and metrics
 */
export interface ChunkingResult {
  /** Array of message chunks */
  chunks: Message[][];
  /** Total number of chunks created */
  numChunks: number;
  /** Average tokens per chunk */
  avgTokensPerChunk: number;
}

/**
 * Embedding with associated message data
 */
export interface MessageEmbedding {
  /** The message ID */
  id: string;
  /** The message index in original array */
  index: number;
  /** The embedding vector */
  embedding: Float32Array;
}

/**
 * Tracks state changes across iteration for convergence measurement
 */
export interface ConvergenceState {
  /** Previous state turning points */
  previousTurningPoints: TurningPoint[];
  /** Current state turning points */
  currentTurningPoints: TurningPoint[];
  /** Current dimension */
  dimension: number;
  /** Convergence measure between states (lower = more converged) */
  distanceMeasure: number;
  /** Whether the state has converged */
  hasConverged: boolean;
  /** Whether dimension escalation occurred */
  didEscalate: boolean;
}

export type EpistemicPrimitives = {
  compatibility: number; // How well TP aligns with evidence
  necessity: number; // How non-falsifiable the TP is
  possibility: number; // Maximum plausibility
  surprisal: number; // Unexpectedness given evidence
};

export type EpistemicSupportSet = {
  boundedRegion: Float32Array[]; // Smolyak sparse grid points
  supportPoints: MessageEmbedding[];
  rejectionThreshold: number;
};
