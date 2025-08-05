import { ResponseFormatJSONSchema } from "openai/resources/shared";
import { Message, MetaMessage } from "./Message";
import { returnFormattedMessageContent } from "./stripContent";
import type { TurningPointDetectorConfig, TurningPointCategory } from "./types";

export const emotionToNumber: Record<string, number> = {
  joyful: 0,
  excited: 1,
  surprised: 2,
  worried: 3,
  anxious: 4,
  angry: 5,
  furious: 6,
  skeptical: 7,
  disgusted: 8,
  sad: 9,
  discouraged: 10,
  hopeful: 11,
};



export function formSystemMessage(
  dimension
): string {
  const isMeta = dimension > 1;

  const contextType = isMeta
    ? "a collection of previously identified turning points, that may each represent and span either messages, or turning points themselves, up to four levels deep"
    : "two adjacent messages, in which are being assessed as a potential turning point of the conversation";

  const analysisFocus = isMeta
    ? "Determine the overarching shift or structural transformation represented by this group of turning points."
    : "Determine whether a significant semantic or emotional shift occurs between these two specific messages.";

  return `
You are a conversation analysis expert tasked with identifying, summarizing, and classifying conversational turning points.

### 📝 Your Goal:
Analyze ${contextType} provided in the user message. ${analysisFocus}

### 🧭 How to Approach:
- Carefully examine the provided content for notable changes in meaning, intent, emotional tone, or topic structure.
- If available, use the 'Contextual Aid' from the system message to situate your understanding of where this turning point occurs within the broader conversation.
- Ground your response in the actual message content provided below the instructions. Do not make inferences beyond what is visible in the message content.

## 🧠 Terminology
- **Message:** The single message within the dialogue processed, that is from an actual person. Given that the provided input is a chain of messages between two people.
- **Turning Point:** A point within a conversation that signifies a shift in the dialogue, in which encompasses and spans at least two messages (in which minimually two messages are required in realizing a 'point' or turning point within the chain of message).
  - **"Meta" Turning Point:** Just like how turning points spans at least two messages, it may also span turning points themselves. This can be done up to five levels of hierarchy (dimension 0 to 4). Though, it is always possible to derive the messages spanned, regardless of a turning point's level of dimension, moreover, when returning the detected turnign points, what is returned is always a turning point that spans messages, or in other words, 

You are to analyze the following user messages, in which will contain, 1) conversation_context or surrounding messages that are not part of the candidate turning point, as well as, a final user message containing the content in which forms this candidate turning point, in which its content comprises of ${isMeta ? 'turning points, up to four levels deep, however is still analyzable as a span of messages, in which, accordingly, would be greater in number as the level or dimension increases' : 'two messages, in which are being assessed as a candidate turning point of the conversation'}`
    + `\nRemember, to refer to the defintions of Terminology provided above for full defintions of what a turning point is, and what its comprised of, among other directions accordingly.`
}





export const circularToneSimilarity = (tone1: string, tone2: string): number => {
  const N = Object.keys(emotionToNumber).length;
  const i1 = emotionToNumber[tone1.toLowerCase()];
  const i2 = emotionToNumber[tone2.toLowerCase()];
  if (i1 === undefined || i2 === undefined) return 0;

  const delta = Math.abs(i1 - i2);
  const dist = Math.min(delta, N - delta); // shortest path around the circle
  const normalized = 1 - (dist / Math.floor(N / 2)); // scaled 0–1

  return Math.max(0, normalized); // prevent negative rounding error
};

export function computeSignificance(
  factors: {

    //  "certainty",
    //         "novelty",
    //         "affectiveDelta",
    //         "impact"
    certainty: number;
    novelty: number;
    affectiveDelta: number;
    impact: number;
  },
  emotionalTone: string,
  context?: {
    averageDistance?: number;
    dimension?: number;
    totalMessages?: number;
    previousEmotionalTone?: string; // Add for transition analysis
  }
): number {
  // Helper function for circular tone similarity (using your exact logic)


  // Get emotional context and intensity
  const getEmotionalContext = (currentTone: string, previousTone?: string) => {
    // Map emotions to intensity using your emotionToNumber as reference
    const emotionIntensity: { [key: string]: number } = emotionToNumber

    const intensity = emotionIntensity[currentTone.toLowerCase()] || 0.3;

    let transitionBoost = 0;
    if (previousTone) {
      const toneSimilarity = circularToneSimilarity(currentTone, previousTone);
      // Bigger boost for dramatic emotional shifts
      transitionBoost = (1 - toneSimilarity) * 0.2; // Max 20% boost
    }

    return {
      intensityBoost: intensity * 0.1, // Max 10% weight boost for intense emotions
      transitionBoost: transitionBoost
    };
  };

  const emotionData = getEmotionalContext(emotionalTone, context?.previousEmotionalTone);

  // Dynamic weight adjustment based on emotional intensity
  const baseWeights = {
    structural: 0.35,
    novelty: 0.25,
    affective: 0.2 + emotionData.intensityBoost, // Boost for intense emotions
    impact: 0.2
  };

  // Adjust weights based on dimension (higher dimensions get more structural weight)
  const dimensionAdjustment = context?.dimension
    ? Math.min(0.15 * context.dimension, 0.3)
    : 0;

  const adjustedWeights = {
    structural: baseWeights.structural + dimensionAdjustment,
    novelty: baseWeights.novelty - (dimensionAdjustment * 0.5),
    affective: baseWeights.affective - (dimensionAdjustment * 0.3),
    impact: baseWeights.impact - (dimensionAdjustment * 0.2)
  };

  // Normalize weights to sum to 1
  const totalWeight = Object.values(adjustedWeights).reduce((sum, w) => sum + w, 0);
  const normalizedWeights = {
    structural: adjustedWeights.structural / totalWeight,
    novelty: adjustedWeights.novelty / totalWeight,
    affective: adjustedWeights.affective / totalWeight,
    impact: adjustedWeights.impact / totalWeight
  };

  // Apply emotional transition scaling
  const emotionalScale = 1 + emotionData.transitionBoost;

  // Apply distance-based scaling if context available
  const distanceScale = context?.averageDistance
    ? Math.min(1, context.averageDistance * 1.2)
    : 1;

  // Calculate weighted score with emotional and distance scaling
  const score = (
    (factors.certainty * normalizedWeights.structural) +
    (factors.novelty * normalizedWeights.novelty) +
    (factors.affectiveDelta * normalizedWeights.affective) +
    ((factors.impact / 10) * normalizedWeights.impact)
  ) * emotionalScale * distanceScale;

  // Final normalization to ensure 0-1 range
  const maxPossibleScore = (
    1 * normalizedWeights.structural +
    1 * normalizedWeights.novelty +
    1 * normalizedWeights.affective +
    1 * normalizedWeights.impact
  ) * emotionalScale * distanceScale;

  const normalizedScore = score / maxPossibleScore;

  return Math.min(1, Math.max(0, Math.round(normalizedScore * 100) / 100));
}

/**
 * Simplified Choquet integral calculation for turning point significance.
 * Uses only the four core significance factors without additional context.
 */

/**
 * Calculates significance using a Choquet integral approach with robust input validation
 * to prevent NaN/undefined errors.
 */
/**
 * Calculates significance using a Choquet integral approach with optional phi enhancement.
 * Handles all edge cases and provides robust fallbacks.
 * 
 * TODO: we updated focator fields, so must update the key names in the factors object
 */
export function computeSignificanceWithChoquet(
  factors: {
    certainty: number;
    novelty: number;
    affectiveDelta: number;
    impact: number;
  },
  emotionalTone: string,
  options?: {
    enableExperimentalPhi?: boolean;
    phiScore?: number;
    dimension?: number;
    averageDistance?: number;
  }
): number {
  try {
    // STEP 1: VALIDATE AND NORMALIZE INPUTS
    if (!factors || typeof factors !== 'object') {
      console.warn('Invalid factors provided to computeSignificanceWithChoquet, using defaults');
      factors = {
        certainty: 0.3,
        novelty: 0.2,
        affectiveDelta: 0.2,
        impact: 2.0
      };
    }

    // Normalize factors with proper validation
    const normalizedFactors = {
      certainty: validateAndClamp(factors.certainty, 0, 1, 0.3),
      novelty: validateAndClamp(factors.novelty, 0, 1, 0.2),
      affectiveDelta: validateAndClamp(factors.affectiveDelta, 0, 1, 0.2),
      impact: validateAndClamp(factors.impact, 0, 10, 2.0) / 10
    };

    // Validate emotional tone
    const validEmotionalTone = typeof emotionalTone === 'string' ?
      emotionalTone.toLowerCase().trim() : 'neutral';

    // STEP 2: CALCULATE EMOTIONAL INTENSITY BOOST
    const emotionParts = validEmotionalTone.split('-');
    const maxEmotionValue = Math.max(...Object.values(emotionToNumber)); // 11

    let intensity = 0.3; // default neutral baseline

    if (emotionParts.length === 1) {
      // Single emotion
      const emotionPosition = emotionToNumber[emotionParts[0]];
      if (emotionPosition !== undefined) {
        intensity = 0.3 + (emotionPosition / maxEmotionValue) * 0.65;
      }
    } else {
      // Composite emotion - average the positions on the wheel
      const validPositions = emotionParts
        .map(part => emotionToNumber[part.trim()])
        .filter(pos => pos !== undefined);

      if (validPositions.length > 0) {
        const avgPosition = validPositions.reduce((sum, pos) => sum + pos, 0) / validPositions.length;
        intensity = 0.3 + (avgPosition / maxEmotionValue) * 0.65;
      }
    }

    const emotionalBoost = intensity * 0.15; // 15% max boost

    // STEP 3: PHI-AWARE ENHANCEMENTS (if enabled)
    let phiMultiplier = 1.0;
    let phiCapacityBoost = 0.0;

    if (options?.enableExperimentalPhi &&
      typeof options.phiScore === 'number' &&
      isFinite(options.phiScore)) {

      const phi = Math.max(0, Math.min(1, options.phiScore));

      // Phi multiplier: high phi amplifies, low phi dampens
      if (phi > 0.7) {
        phiMultiplier = 1.0 + (phi - 0.7) * 0.6; // Up to 18% boost
      } else if (phi < 0.3) {
        phiMultiplier = 0.9 + (phi / 0.3) * 0.1; // Down to 90%
      }

      // Phi capacity boost affects synergy weights
      phiCapacityBoost = (phi - 0.5) * 0.08; // Range: -0.04 to +0.04

      // Apply phi-based factor adjustments
      if (phi > 0.6) {
        // High phi amplifies structural and affective components
        normalizedFactors.certainty = Math.min(1.0,
          normalizedFactors.certainty * (1.0 + (phi - 0.6) * 0.25)
        );
        normalizedFactors.affectiveDelta = Math.min(1.0,
          normalizedFactors.affectiveDelta * (1.0 + (phi - 0.6) * 0.3)
        );
      } else if (phi < 0.4) {
        // Low phi dampens novelty and impact
        normalizedFactors.novelty = Math.max(0.0,
          normalizedFactors.novelty * (0.8 + phi * 0.5)
        );
        normalizedFactors.impact = Math.max(0.0,
          normalizedFactors.impact * (0.85 + phi * 0.375)
        );
      }
    }

    // STEP 4: DEFINE CHOQUET CAPACITIES WITH PHI ADJUSTMENTS
    const baseCapacities = {
      // Individual criterion capacities
      "c": 0.35, // certainty (was "s")
      "n": 0.25, // novelty
      "a": 0.20, // affective delta
      "i": 0.20, // impact (was "c")

      // Pairwise interactions (synergistic effects)
      "cn": 0.65, "ca": 0.60, "ci": 0.60,    // Updated combinations
      "na": 0.50, "ni": 0.50, "ai": 0.45,

      // Triple interactions
      "cna": 0.85, "cni": 0.85, "cai": 0.80, "nai": 0.70,

      // All criteria
      "cnai": 1.0
    };


    // Apply emotional and phi boosts to capacities
    const capacities = {
      "c": Math.min(1.0, baseCapacities.c + phiCapacityBoost * 0.8),
      "n": Math.min(1.0, baseCapacities.n + phiCapacityBoost * 0.6),
      "a": Math.min(1.0, baseCapacities.a + emotionalBoost + phiCapacityBoost * 1.2),
      "i": Math.min(1.0, baseCapacities.i + phiCapacityBoost * 0.4),

      "cn": Math.min(1.0, baseCapacities.cn + phiCapacityBoost * 1.0),
      "ca": Math.min(1.0, baseCapacities.ca + phiCapacityBoost * 1.2),
      "ci": Math.min(1.0, baseCapacities.ci + phiCapacityBoost * 0.8),
      "na": Math.min(1.0, baseCapacities.na + phiCapacityBoost * 1.0),
      "ni": Math.min(1.0, baseCapacities.ni + phiCapacityBoost * 0.6),
      "ai": Math.min(1.0, baseCapacities.ai + phiCapacityBoost * 1.4),

      "cna": Math.min(1.0, baseCapacities.cna + phiCapacityBoost * 1.5),
      "cni": Math.min(1.0, baseCapacities.cni + phiCapacityBoost * 1.2),
      "cai": Math.min(1.0, baseCapacities.cai + phiCapacityBoost * 1.8),
      "nai": Math.min(1.0, baseCapacities.nai + phiCapacityBoost * 1.6),

      "cnai": 1.0 // Always 1.0 for full set
    };

    // STEP 5: PREPARE FACTORS FOR CHOQUET INTEGRAL
    const sortedFactors = [
      { key: "c", value: normalizedFactors.certainty },
      { key: "n", value: normalizedFactors.novelty },
      { key: "a", value: normalizedFactors.affectiveDelta },
      { key: "i", value: normalizedFactors.impact }
    ].sort((a, b) => a.value - b.value);

    // STEP 6: CALCULATE CHOQUET INTEGRAL
    let choquetScore = 0;
    let prevValue = 0;

    for (let i = 0; i < sortedFactors.length; i++) {
      const currentValue = sortedFactors[i].value;
      const increment = currentValue - prevValue;

      if (increment > 0) {
        // Create subset key (sorted criteria at this level and above)
        const subset = sortedFactors
          .slice(i)
          .map(f => f.key)
          .sort()
          .join("");

        // Get capacity with fallback
        const capacity = capacities[subset as keyof typeof capacities] ??
          (subset.length / sortedFactors.length);

        choquetScore += increment * capacity;
      }

      prevValue = currentValue;
    }

    // STEP 7: APPLY FINAL ADJUSTMENTS
    let finalScore = choquetScore * phiMultiplier;

    // Apply dimension and distance scaling if available
    if (options?.dimension !== undefined && options.dimension > 0) {
      const dimensionScale = 1.0 + (options.dimension * 0.03); // Small boost for higher dimensions
      finalScore *= dimensionScale;
    }

    if (options?.averageDistance !== undefined && options.averageDistance > 0) {
      const distanceScale = Math.min(1.15, 1.0 + options.averageDistance * 0.15);
      finalScore *= distanceScale;
    }

    // STEP 8: FINAL VALIDATION AND NORMALIZATION
    if (!isFinite(finalScore) || isNaN(finalScore)) {
      console.warn('Choquet calculation produced invalid result, using fallback');
      return 0.3;
    }

    // Clamp to [0, 1] range and round to 2 decimal places
    return Math.min(1, Math.max(0, Math.round(finalScore * 100) / 100));

  } catch (error) {
    console.error('Error in Choquet calculation:', error);
    return 0.3; // Safe fallback
  }
}

/**
 * Helper function to validate and clamp numeric values
 */
function validateAndClamp(value: any, min: number, max: number, fallback: number): number {
  if (typeof value !== 'number' || !isFinite(value) || isNaN(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, value));
}




// REQUEST 1: Analysis & Classification (Simpler JSON)
export const formAnalysisResponseFormat = (
  dimension: number,
  config: TurningPointDetectorConfig
): ResponseFormatJSONSchema => ({
  type: "json_schema",
  json_schema: {
    description: "Provide a concise analysis and classification of the turning point",
    name: "turning_point_analysis",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      required: ["label", "category", "sentiment", "quotes"],
      properties: {
        label: {
          type: "string",
          description: "Form a descriptive title for this turning point"
        },
        category: {
          type: "string",
          enum: config.turningPointCategories.map(tp => tp.category.toLocaleLowerCase()).sort(() => Math.random() - 0.5),
          description: "Classify the turning point into one of the provided categories"
        },

        sentiment: {
          type: "string",
          enum: ["positive", "negative"],
          description: "Ascribe a negative or positive sentiment that wholly represents the turning point."
        },

        quotes: {
          type: "array",
          items: { type: "string" },
          maxItems: 5,
          minItems: 1,
          description: "Find notable quotes from the content found in the potential turning point, as 'source: <quote>' that represent the turning point, and are to be used in the final summary of the turning point. The quotes should be concise and relevant to the turning point's significance.",
        },

      }
    }
  }
});

// REQUEST 2: Significance Scoring (Numbers only)
export const formScoringResponseFormat = (): ResponseFormatJSONSchema => ({
  type: "json_schema",
  json_schema: {
    description: "Determine the significance of the provided possible turning point",
    name: "significance_scoring",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      required: ["emotionalTone", "certainty", "novelty", "affectiveDelta", "impact"],
      properties: {
   
        certainty: {
          type: "number",
          minimum: 0,
          maximum: 1,
          multipleOf: 0.01,
          description: "How clear/definitive is this turning point boundary"
        },
        novelty: {
          type: "number",
          minimum: 0,
          maximum: 1,
          multipleOf: 0.01,
          description: "How novel/unique is this shift in context"
        },
        affectiveDelta: {
          type: "number",
          minimum: 0,
          maximum: 1,
          multipleOf: 0.01,
          description: "Magnitude of emotional change"
        },
        impact: {
          type: "number",
          minimum: 0,
          maximum: 10,
          multipleOf: 0.1,
          description: "Conversational importance (0-10 scale)"
        },

        emotionalTone: {
          type: "string",
          enum: Object.keys(emotionToNumber).map(t => t.toLowerCase()).sort(() => Math.random() - 0.5),
          description: "Emotional tone of the shift"
        },
      }
    }
  }
});

// PROMPT 1: Analysis & Classification (Shorter, focused)
export function formSystemPrompt(
  dimension: number,
): string {
  const itemType = dimension === 0 ? "messages" : "turning points";
  const contextType = dimension > 1
    ? "a collection of previously identified turning points, that may each represent and span either messages, or turning points themselves, up to four levels deep"
    : "two adjacent messages, in which are being assessed as a potential turning point of the conversation";

  const analysisFocus = dimension > 1
    ? "Determine the overarching shift or structural transformation represented by this group of turning points."
    : "Determine whether a significant semantic or emotional shift occurs between these two specific messages.";

  return `
  
You are an expert analyst in lingustic shifts concerning dialogue and other types of written language that is usually a linear sequence of messages, in which you are to analyze the shift between two ${itemType}.

- As you are providing the means in allowing for the identification of significant points, or better known as turning point(s) within such a linear correspondence. Your goal is to analyze the presented content, in which this case is of two ${itemType}, and determine whether a significant semantic or emotional shift occurs between these two specific ${itemType}.

- The provided ${itemType} are adjacent and part of a larger single written document or conversation. In the case that the provided two items are turning points, this means you are now discerning the lelvel of shift at a high level of abstraction, in which, at most can reach four levels. Thus a turning point may be of a span of messages, or span of turning points themselves, that is also to say, high levels of turning points would indaverntly comprise of more semantic length, and turning points of any dimension are always viewable into the most basic level, of the otehr item type, which are thus messages, or actual, chunks of the provided document or conversation that is being analyzed.


Suggested process:
- As you are providing the means in allowing for the identification of significant points, or better known as turning point(s) within such a linear correspondence. Your goal is to analyze the presented content, in which this case is of two ${itemType}, and determine whether a significant semantic or emotional shift occurs between these two specific ${itemType}.
- Carefully examine the provided content for notable changes in meaning, intent, emotional tone, or topic structure.
- Ground your response in the actual message content provided below the instructions. Do not make inferences beyond what is visible in the message content.

Important Notes:
- Ensure that you primarily analyze the provided content to anlyze, in which case will be presented as the content within the following user message. 
- In additoin, you are to also, at a secondary level, utilize the provided conversation_context, which neighboring content to the presented primary content to analzye. 

🧠 Terminology
- **Message:** The single message within the dialogue processed, that is from an actual person. Given that the provided input is a chain of messages between two people.
- **Turning Point:** A point within a conversation that signifies a shift in the dialogue, in which encompasses and spans at least two messages (in which minimally two messages are required in realizing a 'point' or turning point within the chain of message).
  - **"Meta" Turning Point:** Just like how turning points spans at least two messages, it may also span turning points themselves. This can be done up to five levels of hierarchy (dimension 0 to 4). Though, it is always possible to derive the messages spanned, regardless of a turning point's level of dimension.

Last remarks:

- The content to analyze as well as context is provided then in th efollowing user mesasges, in which will contain, a first user message providng conversation_context, as well as second user mesasge containing two ${itemType} that are being assessed as a potential turning point of the conversation.

Please provide your analysis as instructed in the user messages and present your response in the expected structured format.
`


}



// ANALYSIS PROMPT: Focused on classification and labeling
export const formAnalysisSystemPromptEnding = (dimension: number, _config: TurningPointDetectorConfig) => {
  const itemType = dimension === 0 ? "messages" : "turning points";

  return `As a rememinder of your task at hand, ensure that you adhere to the prior instructions and framing, as well as the presented conversation_context, and content in providing a impactful analysis of a potential turning point presented as two adjacent ${itemType}.

**Context Examination**:
   - Ensure you review the provided content as primary means of your analysis.
   - Understand how such content fits within the broader written context.
   - Disern then your analysis into the required response format defined.

Output Requirements:
- **category**: Cateogrize the potential turning point using one of the provided labels.
- **sentiment**: positive/negative (overall sentiment that represents the turning point)
- **label**: A distinct and creative title based on the content of the turning point.
- **quotes**: Extract notable quotes from the messages of the turning point, or content of the turning point.

Ensure you adhere to all other guidlines in the expected response format, as a structured json object comprised of valid fields as well as values.
`
};

// SCORING PROMPT: Focused on quantitative assessment
export const formScoringSystemPromptEnding = (dimension: number) => {
  const itemType = dimension === 0 ? "messages" : "turning points";

  return `As a reminder of your task at hand, ensure that you adhere to the prior instructions and framing, as well as the presented conversation_context, and content in providing a quantitative significance assessment of a potential turning point presented as two adjacent ${itemType}.

 As you are providing the means in allowing for the numerical evaluation of significant points, or better known as turning point(s) within such a linear correspondence. Your goal is to analyze the presented content, in which this case is of two ${itemType}, and determine the quantitative significance factors that measure the magnitude and importance of any semantic or emotional shift that occurs between these two specific ${itemType}.

- The provided ${itemType} are adjacent and part of a larger single written document or conversation. In the case that the provided two items are turning points, this means you are now scoring the level of shift at a high level of abstraction, in which, at most can reach four levels. Thus a turning point may be of a span of messages, or span of turning points themselves, that is also to say, high levels of turning points would inadvertently comprise of more semantic length, and turning points of any dimension are always viewable into the most basic level, of the other item type, which are thus messages, or actual, chunks of the provided document or conversation that is being analyzed.

Score the significance factors for the shift between these two ${itemType}:

**First ${dimension === 0 ? 'Message' : 'Turning Point'}:**
${dimension > 0 ? "Source" : "Author"}: [Will be provided in content]
Content: [Will be provided in content]

**Second ${dimension === 0 ? 'Message' : 'Turning Point'}:**
${dimension > 0 ? "Source" : "Author"}: [Will be provided in content]  
Content: [Will be provided in content]

What to measure? Focus on:
- Structural certainty of the boundary
- Novelty and uniqueness of the shift
- Emotional change magnitude
- Overall conversational impact

Ensure you provide the following fields in your response:

**certainty**: How clear is the turning point boundary?
- 0.0-0.3: Subtle/unclear shift
- 0.4-0.7: Moderate clarity  
- 0.8-1.0: Very clear boundary

**novelty**: How new/unique is this shift?
- 0.0-0.3: Similar to previous patterns
- 0.4-0.7: Somewhat novel
- 0.8-1.0: Completely new direction

**affectiveDelta**: Emotional change magnitude?
- 0.0-0.3: Minor emotional shift
- 0.4-0.7: Moderate emotional change
- 0.8-1.0: Major emotional transformation  

**impact**: Conversational importance (0-10)?
- 0-3: Minor importance
- 4-6: Moderate importance
- 7-10: Critical to conversation flow

**emotionalTone**: Select from: ${Object.keys(emotionToNumber).slice(0, 8).map(t => t.toLowerCase()).join(", ")}
`
};