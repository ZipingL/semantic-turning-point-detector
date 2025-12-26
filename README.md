# Semantic Turning Point Detector: From Discovery to Direction


[![GitHub](https://img.shields.io/github/stars/gaiaverseltd/semantic-turning-point-detector?style=social)](


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15616960.svg)](https://doi.org/10.5281/zenodo.15616960)

[![DOI](https://img.shields.io/badge/DOI-10.21203/rs.3.rs--6605714/-blue.svg)](https://www.researchsquare.com/article/rs-6605714/)

The **Semantic Turning Point Detector** is a practical instantiation of a bold theoretical ambition: to make AI aware not just of *what* has changed in a conversation, but *why* that change matters. Built atop the **ARC/CRA/DAO** triadic framework, this tool identifies, scores, and clusters **semantic turning points** in dialogues—moments where insight, contradiction, revelation, or emotional upheaval reshape the conversation’s trajectory.

The Zenodo submission for this project is available [here](https://zenodo.org/record/15616960). It offers a comprehensive and accessible background article that explains what the detector is and why it is important. The article is designed for readers without prior knowledge of the underlying theory, architecture, or technical concepts, yet it provides a strong analogical foundation to help understand the detector's purpose and functionality.

## 🔍 What Makes a Turning Point Semantic?

Semantic turning points aren’t topic changes—they are **inflection moments**. They mark emotional ruptures, intellectual pivots, or narrative fractures. These aren’t obvious unless viewed through an epistemic lens tuned to

* **significance (φ)**
* **necessity (Choquet-based fusion)**
* **structural recursion** across dimensions

These are *not* surface-level detections. They emerge from deep alignment between:

* **internal coherence** (ARC),
* **contextual correspondence** (CRA), and
* **intentional salience** (DAO).

---


## 🧠 The Tri-Axial Framework

### ARC — Adaptive Recursive Convergence

ARC recursively refines messages in a conversation, contracting them through semantic clustering. Each recursion layer is only allowed if complexity exceeds a saturation threshold. Contraction ensures convergence and dimensional stabilization.

### CRA — Cascading Re-dimensional Attention

CRA identifies when complexity cannot be resolved in the current dimension. It triggers expansion into higher-order meta-messages. These are abstractions over message spans—like narrative paragraphs formed from message sets.

### DAO — Dimensional Attention Optimization *(New Third Pillar)*

**DAO is the newly introduced third pillar** that transforms the detector from a structural analyzer into an epistemically-aware system. It introduces:

* **φ-field generation**: a scalar significance field from emotion, sentiment, and LLM-rated importance.
* **Choquet fusion of epistemic signals**, ensuring turning points are supported across multiple dimensions.
* **Counterfactual admissibility checks**, which challenge and refine candidate turning points.
* **Possibilistic gating networks** that replace simple filtering with necessity-based selection.

**Together, ARC structures. CRA contextualizes. DAO prioritizes through epistemic awareness.**

---

## 🔁 Why This Third Pillar Matters

The introduction of DAO as the third pillar represents a fundamental shift from confidence-based to **necessity-based** turning point evaluation. Traditional approaches collapse meaning into size, assuming bigger models = deeper insight. Our **φ-aware framework** reveals that **small models, like Qwen3:1.7B**, often outperform larger ones in semantic precision because they are epistemically stricter—they refuse to hallucinate.

**Before DAO (ARC/CRA only):**
- Structural detection based on complexity and dimensional escalation
- Simple significance filtering
- Confidence scores based on model certainty

**After DAO (Complete Tri-Axial Framework):**
- Epistemic primitives: compatibility, necessity, possibility, surprisal
- φ-aware Choquet integral fusion across dimensions
- Counterfactual analysis for robustness
- Necessity scores that capture epistemically essential shifts

Through rigorous testing on real philosophical dialogues like *Strindberg's Pariah*, the **DAO-enhanced detector** achieved:

---

## 📦 Install

```bash
npm install @zipingl/semantic-turning-point-detector
```

---

## ✨ Usage Example

```ts
import { SemanticTurningPointDetector } from '@zipingl/semantic-turning-point-detector';
import fs from 'fs-extra';

const conversation = fs.readJsonSync([
  {
    "author": "system",
    "message": "You are a helpful assistant that identifies semantic turning points in conversations.",
  },
  {
    "author": "user",
    "message": "...",
  },
  {
    "author": "assistant",
    "message": "...",
  },
  {
    "author": "user",
    "message": "...",
  },
  {
    "author": "assistant",
    "message": "...",
  },
]);
const detector = new SemanticTurningPointDetector({
  apiKey: process.env.OPENAI_API_KEY,
  maxRecursionDepth: 5,
  enableExperimentalPhi: true,
  enableCounterfactualAnalysis: true,
  complexitySaturationThreshold: 3,
  dynamicallyAdjustSemanticShiftThreshold: true,
  classificationModel: 'gpt-4.1-nano',
  embeddingModel: 'text-embedding-3-large',
  endpoint: 'https://api.openai.com/v1',
  debug: false,
});

const result = await detector.detectTurningPoints(conversation);
console.log("Detected", result.points.length, "turning points.");
console.log("Confidence:", result.confidence.toFixed(3));
console.log("Necessity:", result.necessity?.toFixed(3));
```

---

## 📈 Scoring Meaning

| Score   | Meaning              | Action                               |
| ------- | -------------------- | ------------------------------------ |
| 0.0–0.2 | Flat or redundant    | Skip / combine                       |
| 0.3–0.4 | Moderate transitions | Acceptable summary                   |
| 0.4–0.6 | Sharp turns          | Ideal detection range                |
| 0.6–1.0 | Fragmented / chaotic | Clean transcript or lower thresholds |

Necessity (ϕ-aware Choquet score) now complements Confidence. It captures not just how different a section is—but how **epistemically essential** that shift was. Models with higher necessity scores exhibit sharper, consensus-backed recognition of meaning.

---

## 🧪 Theoretical Foundations

The detector is grounded in:

* **Banach Fixed-Point Theorem** for convergence
* **Choquet Integral Fusion** over fuzzy φ capacities
* **Counterfactual patch admissibility** for pruning weak turning points
* **Possibilistic Gating**: selecting salient over spurious points

All tied together under the triadic equation:

```
dE/dt = Φ_arc(E) + Ψ_cra(E) + α ∇φ_dao(E)
```



## Key Components

Our implementation directly translates the tri-axial epistemic architecture—ARC, CRA, and DAO—into a functional system.

### 1\. Adaptive Recursive Convergence (ARC) & Complexity

ARC's goal is to achieve internal coherence. A key part of this is the **Complexity Function (χ)**, which determines if the current representational dimension is saturated. When complexity is too high, it signals the need for dimensional escalation.

```typescript
// A simplified complexity score based on significance.
// When phi-awareness (DAO) is enabled, this score is further modulated.
private calculateComplexityScore(significance: number): number {
  // Maps [0,1] significance to a [1,5] complexity scale.
  const complexity = 1 + significance * 4;
  return Math.max(1, Math.min(5, complexity));
}
```

### 2\. Cascading Re-Dimensional Attention (CRA) & Dimensional Expansion

CRA acts as the epistemic auditor. When the **Transition Operator (Ψ)** detects that complexity has saturated (`maxComplexity >= complexitySaturationThreshold`), it triggers dimensional escalation. This is handled within our `multiLayerDetection` loop, which recursively calls itself with a higher dimension.

The expansion from dimension *n* to *n+1* is achieved by creating **meta-messages**—higher-order abstractions that group related turning points. Our framework now generates these using multiple strategies:

  * **Categorical Grouping:** Bundles turning points by their assigned category.
  * **Thematic Clustering (φ-Aware):** When the third pillar (DAO) is enabled, it groups turning points based on thematic resonance, using a similarity score derived from emotional tone, sentiment, and significance.
  * **Temporal Sectioning:** Groups turning points into chronological phases or sections.


```typescript
// Snippet from createMetaMessagesFromTurningPoints, showcasing multiple strategies
private createMetaMessagesFromTurningPoints(
  turningPoints: TurningPoint[],
  originalMessages: Message[],
): Message[] {
  // ... logic to group by category ...

  if (this.config.enableExperimentalPhi) {
    // ... logic for thematic clustering and temporal sectioning ...
  } else {
    // ... fallback to simpler chronological sectioning ...
  }
  return metaMessages;
}
```

### 3\. Differentiating Attentional Orientation (DAO) & The φ-Field

DAO introduces the third epistemic axis of **significance**. It is operationalized through a **φ-Significance Field**, which assigns a salience score to each turning point based on its emotional intensity and thematic importance. This field directly influences merging, filtering, and significance calculations.

```typescript
// Computes the φ-field by interpreting LLM-derived data.
private computePhiSignificanceField(
  turningPoints: TurningPoint[],
): Map<string, number> {
  const phiMap = new Map<string, number>();
  // ... logic to calculate phi from emotional intensity and significance ...
  for (const tp of turningPoints) {
    const normSignificance = Math.min(1.0, tp.significance || 0);
    const toneIntensity = emotionalIntensity[tp.emotionalTone.toLowerCase()] || 0.1;
    const sentimentModifier = tp.sentiment === "negative" ? 1.1 : 1.0;
    const phi = normSignificance * 0.7 + toneIntensity * sentimentModifier * 0.3;
    phiMap.set(tp.id, Math.max(0, Math.min(1, phi)));
  }
  return phiMap;
}
```

### 4\. Possibilistic Fusion & Counterfactual Analysis

Instead of simple filtering, the framework now uses a sophisticated **Possibilistic Gating Network**. This uses a **Choquet integral** to fuse turning points from different dimensions, prioritizing those that are not only significant but also epistemically necessary and compatible with the broader evidence. Finally, an optional **Counterfactual Analysis** layer further refines the selection by assessing the structural importance of each turning point.

```typescript
// The final filtering stage combines multiple advanced techniques
private filterSignificantTurningPoints(
  turningPoints: TurningPoint[],
  phiMap: Map<string, number> = new Map(),
): TurningPoint[] {
    // 1. Calculate Epistemic Primitives (compatibility, necessity, etc.)
    // ...
    // 2. Fuse candidates using a Possibilistic Gating Network (Choquet Integral)
    const fusedTurningPoints = this.createPossibilisticGatingNetwork(byDimension);
    // 3. (Optional) Enhance selection with Counterfactual Analysis
    const enhancedTurningPoints = this.counterfactualAnalyzer.enhanceTurningPointSelection(fusedTurningPoints);
    // 4. Enforce hard cap and return sorted results
    // ...
}
```

## Project Structure

This semantic turning point detector is organized into modular components that implement the ARC/CRA/DAO framework for conversation analysis. Below is an overview of the key files:

```
src/
├── conversation.ts                    # Sample sci-fi conversation for testing/demo
├── conversationPariah.json           # Test dataset: Strindberg's "The Stronger" dialogue
├── counterfactual.ts                  # Experimental robustness testing via perturbation analysis
├── index.ts                          # Main export file exposing public API
├── Message.ts                        # Core message types and MetaMessage class for dimensional escalation
├── prompt.ts                         # LLM prompt templates and response formatting
├── resultsV4.md                      # Analysis results comparing different models (GPT-4, Gemini, Qwen, etc.)
├── semanticTurningPointDetector.ts   # Main detector class implementing ARC/CRA framework
├── stripContent.ts                   # Content preprocessing utilities
├── tokensUtil.ts                     # Token counting and text processing utilities
└── types.ts                          # TypeScript interfaces and configuration types
```

### Core Files

- **semanticTurningPointDetector.ts** - The main implementation featuring:
  - Multi-dimensional ARC analysis (n → n+1 escalation)
  - Experimental phi-field enhancement for significance scoring
  - Possibilistic gating networks for robust turning point fusion
  - Configurable LLM integration (OpenAI, OpenRouter, Ollama, etc.)

- **types.ts** - Comprehensive type definitions including:
  - `TurningPointDetectorConfig` with 30+ configuration options
  - `TurningPoint` interface with semantic metadata
  - Epistemic primitives for advanced filtering
  - Convergence tracking types

- **`Message.ts`** - Message handling with:
  - Base `Message` interface for conversation data
  - `MetaMessage` class for representing higher-dimensional turning point clusters
  - Span management and indexing utilities

### Analysis & Testing

- **resultsV4.md** - Real analysis results showing:
  - Performance across different LLM models (GPT-4.1, Gemini-2.5-flash, Qwen3)
  - Convergence patterns in dimensional escalation
  - Confidence vs. necessity scoring comparisons

- **counterfactual.ts** - Experimental feature providing:
  - Robustness testing through perturbation analysis
  - Enhanced turning point selection via "what-if" scenarios
  - Quality assurance for detected semantic shifts

### Utilities & Support

- **`prompt.ts`** - Advanced prompting system with:
  - Modular prompt architecture for LLM clarity
  - JSON schema response formatting
  - Context management for dimensional analysis

- **`tokensUtil.ts`** - Text processing including:
  - Accurate token counting for various models
  - Text truncation with ratio estimation
  - Memory-efficient caching strategies


For detailed configuration options, see the comprehensive JSDoc documentation in semanticTurningPointDetector.ts and type definitions in types.ts.

## Results

As current there is a scattering of results, so unless otherwise noted, the latest results are found in [README.RESULTS.V4.md](results/README.RESULTS.V4.md).
This file contains detailed analysis of the detector's performance across various models, including:
- **Model comparisons**: GPT-4.1, Gemini-2.5-flash:thinking, Qwen3:1.7B, Qwen3:30B-2507, and GPT-4.1-nano.
- **Turning point detection accuracy**: How many turning points were detected and their significance.
- **Convergence patterns**: How the detector's performance changes with different configurations and model choices.
- **Confidence vs. necessity**: Analysis of how the detector's confidence scores align with the necessity of detected turning points.


These results are from analysis of the well known Pariah dialogue by Strindberg, which is a rich source of semantic turning points. The analysis demonstrates the detector's ability to identify key moments of insight, contradiction, and emotional upheaval in the conversation. The version of how it was used can be found in [conversationPariah.json](src/conversationPariah.json). 


Below is a brief comparison that visualizes and highlights the key improvements in the detector's functionality compared to the previous version, v3.3.6.




#### Original (pre-DAO, confidence-based)
```cypher
Message   0   10   20   30   40   50   60   70   80   90  100  110  120  130  140  150
Index     │   │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
          ▼   ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
gpt4.1nano ●   ●        ●         ●         ●              ●         ●         ●    ●    ●
Gemini         ●    ●                  ●         ●    ●    ●    ●         ●         ●
gpt-4.1   ●   ●             ●                              ●         ●          ●    ●
qwen1.7b  ●   ●                  ●              ●    ●    ●    ●    ●    ●         ●
qwen30b   ●                        ●   ●              ●    ●         ●    ●    ●    ●
```

#### After DAO (v4.0.0, necessity/φ-aware, same binning)
```cypher
Message   0   10   20   30   40   50   60   70   80   90  100  110  120  130  140  150
Index     │   │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
          ▼   ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
Qwen-1.7b ●                                   ● ● ● ● ● ● ● ●
GPT-4.1-n ● ●                                       ●       ●     ● ●
GPT-4.1   ● ● ● ● ●     ●   ●   ● ●       ● ●     ● ●
Qwen-30B  ●                   ● ●         ●     ● ●
Gemini    ●           ● ●   ●     ●       ●     ● ● ● ●
```

## 📖 Citation

Liu, Z., & Jah, M. (2025). *Adaptive Recursive Convergence and Semantic Turning Points: A Self-Verifying Architecture for Progressive AI Reasoning.* Research Square Preprint. [https://doi.org/10.21203/rs.3.rs-6605714/v1](https://doi.org/10.21203/rs.3.rs-6605714/v1)



## Miscellaneous

[![wakatime](https://wakatime.com/badge/user/e012350f-8b4a-4ec4-ae89-56e558bfec5d/project/0bc83f4e-e5ca-423c-bee6-b2b9b49f4965.svg)](https://wakatime.com/badge/user/e012350f-8b4a-4ec4-ae89-56e558bfec5d/project/0bc83f4e-e5ca-423c-bee6-b2b9b49f4965)