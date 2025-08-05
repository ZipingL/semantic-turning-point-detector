import { TurningPoint } from './types';

export interface PerturbationTest {
  factor: number;    // How much to change (0.1 = 10% change)
  direction: 'up' | 'down' | 'random';
}

/**
 * CounterfactualAnalyzer - ONE PURPOSE: Better turning point selection via robustness testing
 * 
 * MAIN METHOD: enhanceTurningPointSelection()
 */
export class CounterfactualAnalyzer {

  /**
   * THE MAIN METHOD - Enhance turning point selection using counterfactual analysis
   */
  public enhanceTurningPointSelection(turningPoints: TurningPoint[]): TurningPoint[] {
    if (turningPoints.length === 0) return [];

    // Test each turning point's robustness and score them
    const scoredTPs = turningPoints.map(tp => {
      const robustnessScore = this.testRobustness(tp);
      const enhancedScore = (tp.significance * 0.6) + (robustnessScore * 0.4);

      return {
        tp,
        enhancedScore,
        robustnessScore
      };
    });

    // Sort by enhanced score and return
    return scoredTPs
      .sort((a, b) => b.enhancedScore - a.enhancedScore)
      .map(item => item.tp);
  }

  /**
   * Test turning point robustness with simple perturbations
   */
  private testRobustness(tp: TurningPoint): number {
    const tests = this.getPerturbationTests();
    let totalRobustness = 0;

    for (const test of tests) {
      const modifiedTP = this.applyPerturbation(structuredClone(tp), test);
      const robustness = this.calculateRobustness(tp, modifiedTP);
      totalRobustness += robustness;
    }

    return totalRobustness / tests.length;
  }

  private getPerturbationTests(): PerturbationTest[] {
    return [
      { factor: 0.1, direction: 'down' },    // -10% significance
      { factor: 0.2, direction: 'down' },    // -20% significance  
      { factor: 0.1, direction: 'up' },      // +10% significance
      { factor: 0.15, direction: 'random' }, // ±15% random
      { factor: 0.25, direction: 'random' }  // ±25% random
    ];
  }

  private applyPerturbation(tp: TurningPoint, test: PerturbationTest): TurningPoint {
    let multiplier = 1;

    switch (test.direction) {
      case 'up':
        multiplier = 1 + test.factor;
        break;
      case 'down':
        multiplier = 1 - test.factor;
        break;
      case 'random':
        multiplier = 1 + (Math.random() - 0.5) * test.factor * 2;
        break;
    }

    tp.significance = Math.max(0.1, Math.min(1.0, tp.significance * multiplier));
    return tp;
  }

  private calculateRobustness(original: TurningPoint, modified: TurningPoint): number {
    const change = Math.abs(modified.significance - original.significance);
    // Lower change = higher robustness
    return Math.max(0, 1 - (change * 2));
  }
}