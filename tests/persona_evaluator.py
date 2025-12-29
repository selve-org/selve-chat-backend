"""
Automated Persona Evaluation System

Tests SELVE chatbot's ability to accurately assess personalities through conversation.

Uses LLM-as-judge pattern from Google ADK to evaluate:
1. Scoring accuracy (ground truth vs. chatbot assessment)
2. Conversation quality (rapport, follow-ups, depth)
3. Detection of key psychological patterns
4. Handling of edge cases (manipulation, inconsistency)

Target: 9.5/10 accuracy ("DECIPHER ACCURATELY like a detective")
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from openai import AsyncOpenAI

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.thinking_engine import ThinkingEngine
from app.services.user_state_service import UserStateService, UserState, AssessmentStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PersonaEvaluationResult:
    """Results from evaluating a single persona."""
    persona_id: str
    persona_name: str

    # Scoring accuracy
    ground_truth_scores: Dict[str, int]
    chatbot_scores: Dict[str, int]
    dimension_deviations: Dict[str, int]
    avg_deviation: float
    max_deviation: int
    within_5_points: int  # How many dimensions within 5 points
    within_10_points: int  # How many dimensions within 10 points

    # Detection quality
    key_patterns_detected: List[str] = field(default_factory=list)
    key_patterns_missed: List[str] = field(default_factory=list)

    # Conversation quality
    conversation_transcript: List[Dict[str, str]] = field(default_factory=list)
    conversation_quality_score: Optional[float] = None

    # LLM-as-judge evaluation
    judge_assessment: Optional[str] = None
    judge_score: Optional[float] = None

    # Overall
    passed_excellent: bool = False  # Within 5 points on 7/8 dimensions
    passed_good: bool = False  # Within 10 points on 6/8 dimensions
    passed_acceptable: bool = False  # Within 15 points on 5/8 dimensions


@dataclass
class BatchEvaluationResults:
    """Results from evaluating all personas."""
    timestamp: str
    total_personas: int

    # Overall accuracy metrics
    avg_deviation_across_all: float
    max_deviation_across_all: int

    # Pass rates
    excellent_count: int
    good_count: int
    acceptable_count: int
    poor_count: int

    # Per-dimension accuracy
    dimension_avg_deviations: Dict[str, float]

    # Individual results
    persona_results: List[PersonaEvaluationResult]

    # Final grade
    overall_grade: str  # "A+" to "F"
    overall_score: float  # 0-10 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_personas": self.total_personas,
            "avg_deviation_across_all": self.avg_deviation_across_all,
            "max_deviation_across_all": self.max_deviation_across_all,
            "excellent_count": self.excellent_count,
            "good_count": self.good_count,
            "acceptable_count": self.acceptable_count,
            "poor_count": self.poor_count,
            "dimension_avg_deviations": self.dimension_avg_deviations,
            "overall_grade": self.overall_grade,
            "overall_score": self.overall_score,
            "persona_results": [
                {
                    "persona_id": r.persona_id,
                    "persona_name": r.persona_name,
                    "avg_deviation": r.avg_deviation,
                    "max_deviation": r.max_deviation,
                    "within_5_points": r.within_5_points,
                    "within_10_points": r.within_10_points,
                    "passed_excellent": r.passed_excellent,
                    "passed_good": r.passed_good,
                    "judge_score": r.judge_score,
                    "key_patterns_detected": r.key_patterns_detected,
                    "key_patterns_missed": r.key_patterns_missed,
                }
                for r in self.persona_results
            ],
        }


class PersonaSimulator:
    """Simulates a persona in conversation with the chatbot."""

    def __init__(self, persona: Dict[str, Any]):
        """Initialize simulator with persona data."""
        self.persona = persona
        self.openai = AsyncOpenAI()
        self.conversation_history: List[Dict[str, str]] = []

    async def generate_response(self, chatbot_message: str) -> str:
        """
        Generate how this persona would respond to chatbot's message.

        Uses GPT-4o to simulate persona's conversational style.
        """
        system_prompt = f"""You are simulating a conversation as this person:

Name: {self.persona['name']}
Age: {self.persona['age']}
Background: {self.persona['background']}

Personality Scores (0-100 scale):
{json.dumps(self.persona['ground_truth_scores'], indent=2)}

Behavioral Patterns:
{json.dumps(self.persona['behavioral_markers'], indent=2)}

IMPORTANT Instructions:
1. Respond EXACTLY as this person would - use their voice, tone, vocabulary
2. Include specific behavioral examples from your life
3. Be authentic to the personality scores (e.g., low KAEL = passive language)
4. Reference actual situations from the behavioral markers
5. Do NOT explicitly state your scores or dimensions
6. Let your personality show through natural conversation
7. If chatbot asks about feelings/behaviors, give specific examples
8. Match the emotional tone described in the persona

Example responses this persona has given:
{json.dumps(self.persona['expected_chat_samples'][:2], indent=2)}
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)

        # Add chatbot's new message
        messages.append({"role": "user", "content": f"Chatbot says: {chatbot_message}"})

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.8,  # Some creativity for natural conversation
            max_tokens=300,
        )

        persona_response = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": chatbot_message})
        self.conversation_history.append({"role": "assistant", "content": persona_response})

        return persona_response


class PersonaEvaluator:
    """Evaluates chatbot's personality assessment accuracy using test personas."""

    DIMENSIONS = ["LUMEN", "AETHER", "ORPHEUS", "ORIN", "LYRA", "VARA", "CHRONOS", "KAEL"]

    def __init__(self, personas_file: str = "tests/test_personas.json"):
        """Initialize evaluator."""
        self.personas_file = Path(personas_file)
        self.openai = AsyncOpenAI()

        # Load personas
        with open(self.personas_file, 'r') as f:
            data = json.load(f)
            self.personas = data['personas']
            self.evaluation_criteria = data['evaluation_criteria']

        logger.info(f"Loaded {len(self.personas)} test personas")

    async def run_conversation_with_persona(
        self,
        persona: Dict[str, Any],
        num_turns: int = 8,
    ) -> List[Dict[str, str]]:
        """
        Simulate a conversation between chatbot and persona.

        Args:
            persona: Persona data
            num_turns: Number of back-and-forth exchanges

        Returns:
            Conversation transcript
        """
        logger.info(f"Starting conversation with {persona['name']}")

        # Create persona simulator
        simulator = PersonaSimulator(persona)

        # Create chatbot instance
        thinking_engine = ThinkingEngine()

        # Simulate user session
        user_id = f"test_persona_{persona['id']}"
        session_id = f"eval_session_{datetime.utcnow().timestamp()}"

        conversation_transcript = []

        # Initial chatbot greeting
        chatbot_message = "Hi! I'm here to help you explore your personality. What's been on your mind lately?"
        conversation_transcript.append({
            "role": "assistant",
            "content": chatbot_message,
        })

        # Conversation loop
        for turn in range(num_turns):
            # Persona responds
            persona_response = await simulator.generate_response(chatbot_message)
            conversation_transcript.append({
                "role": "user",
                "content": persona_response,
            })

            logger.info(f"Turn {turn + 1}/{num_turns} - Persona: {persona_response[:100]}...")

            # Chatbot responds (using thinking engine)
            try:
                # Get conversation history in right format
                history = []
                for msg in conversation_transcript[:-1]:  # Exclude the message we just added
                    if msg['role'] in ('user', 'assistant'):
                        history.append(msg)

                # Create minimal user state for testing
                user_state = UserState(
                    user_id=user_id,
                    clerk_user_id=user_id,
                    user_name=persona['name'],
                    assessment_status=AssessmentStatus.NOT_TAKEN,
                    has_assessment=False,
                )

                # System prompt for the chatbot
                system_prompt = """You are a SELVE chatbot designed to understand personality through conversation.
Your goal is to have a natural conversation while learning about the user's personality across 8 dimensions:
LUMEN (social energy), AETHER (emotional stability), ORPHEUS (empathy), ORIN (organization),
LYRA (openness/creativity), VARA (honesty), CHRONOS (patience), KAEL (assertiveness).

Ask thoughtful, probing questions. Build rapport. Be curious. Detect patterns in their behavior."""

                # Get chatbot response
                chatbot_result = await thinking_engine.think_and_respond_sync(
                    message=persona_response,
                    user_state=user_state,
                    conversation_history=history,
                    system_prompt=system_prompt,
                )

                chatbot_message = chatbot_result.response
                conversation_transcript.append({
                    "role": "assistant",
                    "content": chatbot_message,
                })

                logger.info(f"Turn {turn + 1}/{num_turns} - Chatbot: {chatbot_message[:100]}...")

            except Exception as e:
                logger.error(f"Error getting chatbot response: {e}")
                chatbot_message = "Tell me more about that."
                conversation_transcript.append({
                    "role": "assistant",
                    "content": chatbot_message,
                })

        return conversation_transcript

    async def extract_chatbot_assessment(
        self,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Use LLM to extract chatbot's implicit personality assessment from conversation.

        Since chatbot doesn't give explicit scores in chat, we use GPT-4o to analyze
        what the chatbot's questions and responses reveal about its assessment.
        """
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation
        ])

        prompt = f"""Analyze this conversation between a personality assessment chatbot and a user.

CONVERSATION:
{conversation_text}

Based on the chatbot's questions, follow-ups, and observations, estimate what personality scores (0-100) the chatbot would assign for these 8 dimensions:

1. LUMEN (Social Energy): 0 = Very Introverted, 100 = Very Extroverted
2. AETHER (Emotional Stability): 0 = Very Anxious/Unstable, 100 = Very Stable/Calm
3. ORPHEUS (Empathy): 0 = Low Empathy, 100 = High Empathy
4. ORIN (Organization): 0 = Chaotic, 100 = Highly Organized
5. LYRA (Creativity): 0 = Conventional, 100 = Highly Creative
6. VARA (Honesty): 0 = Strategic/Manipulative, 100 = Direct/Honest
7. CHRONOS (Patience): 0 = Impulsive, 100 = Patient
8. KAEL (Assertiveness): 0 = Passive, 100 = Assertive

Look for:
- What behaviors/patterns the chatbot focused on
- What follow-up questions it asked (reveals what it's detecting)
- Any explicit observations about personality traits
- Implicit assumptions in its responses

Return ONLY a JSON object with dimension scores:
{{"LUMEN": 45, "AETHER": 22, "ORPHEUS": 78, ...}}"""

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temp for more consistent extraction
            response_format={"type": "json_object"},
        )

        scores_json = response.choices[0].message.content
        scores = json.loads(scores_json)

        # Validate all dimensions present
        for dim in self.DIMENSIONS:
            if dim not in scores:
                logger.warning(f"Missing dimension {dim} in chatbot assessment, defaulting to 50")
                scores[dim] = 50

        return scores

    async def judge_conversation_quality(
        self,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use LLM-as-judge to evaluate conversation quality.

        Assesses:
        - Rapport building
        - Question depth and relevance
        - Follow-up quality
        - Detection of key patterns
        """
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation
        ])

        expected_patterns = persona['assessment_expectations']['should_detect']

        prompt = f"""Evaluate this personality assessment conversation.

CONVERSATION:
{conversation_text}

PERSONA GROUND TRUTH:
The user in this conversation has these actual characteristics:
{json.dumps(persona['behavioral_markers'], indent=2)}

KEY PATTERNS THE CHATBOT SHOULD DETECT:
{json.dumps(expected_patterns, indent=2)}

Evaluate the chatbot on:

1. **Rapport Building** (0-10): Did it establish trust before deep questions?
2. **Question Quality** (0-10): Were questions specific, behavioral, and probing?
3. **Follow-up Depth** (0-10): Did it ask follow-ups on key revelations?
4. **Pattern Detection** (0-10): Did it pick up on the important personality markers?
5. **Avoiding Assumptions** (0-10): Did it ask rather than assume?

For each category, provide:
- Score (0-10)
- Justification (2-3 sentences)
- Key patterns detected vs. missed

Return JSON:
{{
  "rapport_building": {{"score": X, "justification": "..."}},
  "question_quality": {{"score": X, "justification": "..."}},
  "follow_up_depth": {{"score": X, "justification": "..."}},
  "pattern_detection": {{"score": X, "justification": "..."}},
  "avoiding_assumptions": {{"score": X, "justification": "..."}},
  "overall_score": X.X,
  "patterns_detected": ["pattern1", "pattern2"],
  "patterns_missed": ["pattern1", "pattern2"],
  "summary": "Overall assessment in 2-3 sentences"
}}"""

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        judgment = json.loads(response.choices[0].message.content)
        return judgment

    def calculate_deviations(
        self,
        ground_truth: Dict[str, int],
        chatbot_scores: Dict[str, int],
    ) -> Dict[str, Any]:
        """Calculate deviation metrics between ground truth and chatbot scores."""
        deviations = {}

        for dim in self.DIMENSIONS:
            gt = ground_truth.get(dim, 50)
            cb = chatbot_scores.get(dim, 50)
            deviations[dim] = abs(gt - cb)

        dev_values = list(deviations.values())

        return {
            "dimension_deviations": deviations,
            "avg_deviation": statistics.mean(dev_values),
            "max_deviation": max(dev_values),
            "within_5_points": sum(1 for d in dev_values if d <= 5),
            "within_10_points": sum(1 for d in dev_values if d <= 10),
            "within_15_points": sum(1 for d in dev_values if d <= 15),
        }

    def grade_persona_result(self, result: PersonaEvaluationResult) -> None:
        """Determine pass/fail grades for persona evaluation."""
        # Excellent: Within 5 points on 7/8 dimensions
        result.passed_excellent = result.within_5_points >= 7

        # Good: Within 10 points on 6/8 dimensions
        result.passed_good = result.within_10_points >= 6

        # Acceptable: Within 15 points on 5/8 dimensions
        within_15 = sum(1 for d in result.dimension_deviations.values() if d <= 15)
        result.passed_acceptable = within_15 >= 5

    async def evaluate_persona(self, persona: Dict[str, Any]) -> PersonaEvaluationResult:
        """Run complete evaluation for a single persona."""
        logger.info(f"\\n{'='*60}")
        logger.info(f"Evaluating: {persona['name']} ({persona['id']})")
        logger.info(f"{'='*60}")

        # Run conversation
        conversation = await self.run_conversation_with_persona(persona, num_turns=8)

        # Extract chatbot's assessment
        chatbot_scores = await self.extract_chatbot_assessment(conversation, persona)

        # Calculate deviations
        deviation_metrics = self.calculate_deviations(
            persona['ground_truth_scores'],
            chatbot_scores,
        )

        # Judge conversation quality
        judgment = await self.judge_conversation_quality(conversation, persona)

        # Create result
        result = PersonaEvaluationResult(
            persona_id=persona['id'],
            persona_name=persona['name'],
            ground_truth_scores=persona['ground_truth_scores'],
            chatbot_scores=chatbot_scores,
            dimension_deviations=deviation_metrics['dimension_deviations'],
            avg_deviation=deviation_metrics['avg_deviation'],
            max_deviation=deviation_metrics['max_deviation'],
            within_5_points=deviation_metrics['within_5_points'],
            within_10_points=deviation_metrics['within_10_points'],
            conversation_transcript=conversation,
            conversation_quality_score=judgment['overall_score'],
            judge_assessment=judgment['summary'],
            judge_score=judgment['overall_score'],
            key_patterns_detected=judgment['patterns_detected'],
            key_patterns_missed=judgment['patterns_missed'],
        )

        # Grade result
        self.grade_persona_result(result)

        # Log summary
        logger.info(f"\\n📊 Results for {persona['name']}:")
        logger.info(f"  Avg Deviation: {result.avg_deviation:.1f} points")
        logger.info(f"  Max Deviation: {result.max_deviation} points")
        logger.info(f"  Within 5 pts: {result.within_5_points}/8 dimensions")
        logger.info(f"  Within 10 pts: {result.within_10_points}/8 dimensions")
        logger.info(f"  Grade: {'✅ EXCELLENT' if result.passed_excellent else '✅ GOOD' if result.passed_good else '⚠️  ACCEPTABLE' if result.passed_acceptable else '❌ POOR'}")
        logger.info(f"  Conversation Quality: {result.conversation_quality_score:.1f}/10")
        logger.info(f"  Patterns Detected: {len(result.key_patterns_detected)}")
        logger.info(f"  Patterns Missed: {len(result.key_patterns_missed)}")

        return result

    async def evaluate_all_personas(
        self,
        persona_ids: Optional[List[str]] = None,
    ) -> BatchEvaluationResults:
        """
        Evaluate all personas (or subset) and generate comprehensive report.

        Args:
            persona_ids: Optional list of persona IDs to evaluate. If None, evaluates all.
        """
        logger.info("\\n" + "="*60)
        logger.info("🚀 Starting Persona Evaluation Batch")
        logger.info("="*60)

        # Filter personas if IDs provided
        if persona_ids:
            personas_to_eval = [p for p in self.personas if p['id'] in persona_ids]
        else:
            personas_to_eval = self.personas

        logger.info(f"Evaluating {len(personas_to_eval)} personas...")

        # Evaluate each persona
        results = []
        for persona in personas_to_eval:
            result = await self.evaluate_persona(persona)
            results.append(result)

        # Calculate batch metrics
        all_deviations = [r.avg_deviation for r in results]
        all_max_deviations = [r.max_deviation for r in results]

        # Per-dimension average deviations
        dimension_deviations = {dim: [] for dim in self.DIMENSIONS}
        for result in results:
            for dim, dev in result.dimension_deviations.items():
                dimension_deviations[dim].append(dev)

        dimension_avg_deviations = {
            dim: statistics.mean(devs)
            for dim, devs in dimension_deviations.items()
        }

        # Count pass rates
        excellent_count = sum(1 for r in results if r.passed_excellent)
        good_count = sum(1 for r in results if r.passed_good and not r.passed_excellent)
        acceptable_count = sum(1 for r in results if r.passed_acceptable and not r.passed_good)
        poor_count = len(results) - excellent_count - good_count - acceptable_count

        # Calculate overall score (0-10 scale)
        # Target: 9.5/10 = "excellent" grade
        # Formula: Higher is better, lower deviation is better
        avg_deviation = statistics.mean(all_deviations)

        # Score based on deviation:
        # 0-5 avg deviation = 10/10
        # 6-10 = 9/10
        # 11-15 = 8/10
        # 16-20 = 7/10
        # 21+ = decreasing
        if avg_deviation <= 5:
            score = 10.0
        elif avg_deviation <= 10:
            score = 10.0 - (avg_deviation - 5) * 0.2
        elif avg_deviation <= 15:
            score = 9.0 - (avg_deviation - 10) * 0.2
        elif avg_deviation <= 20:
            score = 8.0 - (avg_deviation - 15) * 0.2
        else:
            score = max(1.0, 7.0 - (avg_deviation - 20) * 0.1)

        # Adjust score based on pass rates
        excellent_rate = excellent_count / len(results)
        if excellent_rate >= 0.8:
            score += 0.5
        elif excellent_rate >= 0.6:
            score += 0.25

        score = min(10.0, score)

        # Assign letter grade
        if score >= 9.5:
            grade = "A+"
        elif score >= 9.0:
            grade = "A"
        elif score >= 8.5:
            grade = "A-"
        elif score >= 8.0:
            grade = "B+"
        elif score >= 7.5:
            grade = "B"
        elif score >= 7.0:
            grade = "B-"
        elif score >= 6.5:
            grade = "C+"
        elif score >= 6.0:
            grade = "C"
        else:
            grade = "F"

        # Create batch results
        batch_results = BatchEvaluationResults(
            timestamp=datetime.utcnow().isoformat(),
            total_personas=len(results),
            avg_deviation_across_all=statistics.mean(all_deviations),
            max_deviation_across_all=max(all_max_deviations),
            excellent_count=excellent_count,
            good_count=good_count,
            acceptable_count=acceptable_count,
            poor_count=poor_count,
            dimension_avg_deviations=dimension_avg_deviations,
            persona_results=results,
            overall_grade=grade,
            overall_score=score,
        )

        # Log final summary
        logger.info("\\n" + "="*60)
        logger.info("📈 FINAL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"\\n🎯 Overall Score: {score:.2f}/10 (Grade: {grade})")
        logger.info(f"\\n📊 Pass Rates:")
        logger.info(f"  ✅ Excellent: {excellent_count}/{len(results)} ({excellent_rate*100:.1f}%)")
        logger.info(f"  ✅ Good: {good_count}/{len(results)}")
        logger.info(f"  ⚠️  Acceptable: {acceptable_count}/{len(results)}")
        logger.info(f"  ❌ Poor: {poor_count}/{len(results)}")
        logger.info(f"\\n📏 Deviation Metrics:")
        logger.info(f"  Average Deviation: {batch_results.avg_deviation_across_all:.1f} points")
        logger.info(f"  Maximum Deviation: {batch_results.max_deviation_across_all} points")
        logger.info(f"\\n📐 Per-Dimension Average Deviations:")
        for dim, avg_dev in sorted(dimension_avg_deviations.items(), key=lambda x: x[1]):
            logger.info(f"  {dim}: {avg_dev:.1f} points")

        # Determine if target met
        if score >= 9.5:
            logger.info(f"\\n🎉 TARGET MET! Score {score:.2f}/10 >= 9.5/10")
            logger.info("SELVE is DECIPHERING ACCURATELY like a detective! ✅")
        else:
            logger.info(f"\\n⚠️  Target not yet met. Score {score:.2f}/10 < 9.5/10")
            logger.info(f"Need to improve by {9.5 - score:.2f} points")

        logger.info("\\n" + "="*60)

        return batch_results

    def save_results(
        self,
        results: BatchEvaluationResults,
        output_file: str = "tests/evaluation_results.json",
    ):
        """Save evaluation results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        logger.info(f"\\n💾 Results saved to {output_path}")


async def main():
    """Run persona evaluation."""
    evaluator = PersonaEvaluator()

    # Option 1: Evaluate all personas
    # results = await evaluator.evaluate_all_personas()

    # Option 2: Evaluate specific personas for testing
    results = await evaluator.evaluate_all_personas(
        persona_ids=[
            "persona_001_anxious_perfectionist",
            "persona_002_charismatic_entrepreneur",
            "persona_013_balanced_therapist",
        ]
    )

    # Save results
    evaluator.save_results(results)

    # Return exit code based on grade
    if results.overall_score >= 9.5:
        print("\\n✅ Evaluation PASSED - Target accuracy achieved!")
        return 0
    else:
        print("\\n❌ Evaluation needs improvement")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
