# SELVE Persona Evaluation System

Automated testing framework to evaluate SELVE chatbot's personality assessment accuracy.

**Target:** 9.5/10 accuracy - "DECIPHER ACCURATELY like a detective"

---

## 📋 Overview

This system tests whether SELVE can accurately assess personalities through conversation by:

1. **Simulating realistic personas** with ground truth personality scores
2. **Running conversations** between chatbot and simulated personas
3. **Extracting chatbot's assessment** from conversation patterns
4. **Comparing to ground truth** and calculating accuracy metrics
5. **Using LLM-as-judge** to evaluate conversation quality

---

## 🎭 Test Personas

**15 diverse personas** covering wide range of personality profiles:

| Persona | Key Traits | Edge Cases Tested |
|---------|-----------|-------------------|
| Anxious Perfectionist | Low AETHER, High ORIN | Anxiety + perfectionism |
| Charismatic Entrepreneur | High LUMEN, Low ORIN | Manic energy, chaos |
| Introverted Artist | Low LUMEN, High LYRA | Deep introversion |
| Type-A Executive | High ORIN, Low ORPHEUS | Low empathy leader |
| Anxious People-Pleaser | High ORPHEUS, Low KAEL | Boundary issues, burnout |
| Cynical Academic | Low ORPHEUS, High LYRA | Intellectual detachment |
| Chaotic Creative | High LYRA, Low ORIN | ADHD traits |
| Stoic Veteran | High ORIN, Low ORPHEUS | PTSD, emotional suppression |
| Manic Salesperson | High LUMEN, Low CHRONOS | Compulsive, manipulative |
| Burned-out Nurse | Low AETHER | Compassion fatigue |
| Grandiose Narcissist | High KAEL, Low ORPHEUS | Pathological narcissism |
| Depressed Grad Student | Low AETHER | Clinical depression |
| Balanced Therapist | Balanced across all | Healthy baseline |
| Rebellious Teenager | Low AETHER, High LYRA | Adolescent turmoil |
| Workaholic Immigrant | High ORIN, High CHRONOS | Cultural factors |

Each persona includes:
- ✅ Ground truth scores for all 8 SELVE dimensions
- ✅ Behavioral markers (conversation patterns, work behaviors, stress responses)
- ✅ Expected chat samples showing their voice
- ✅ Key insights the chatbot should detect

---

## 🎯 Evaluation Criteria

### Scoring Accuracy Grades

| Grade | Criteria |
|-------|----------|
| **Excellent** | Within 5 points on 7/8 dimensions |
| **Good** | Within 10 points on 6/8 dimensions |
| **Acceptable** | Within 15 points on 5/8 dimensions |
| **Poor** | Larger deviations |

### Overall Score (0-10 scale)

- **9.5-10.0**: A+ (Target met! "Detective-level accuracy")
- **9.0-9.4**: A (Excellent, nearly there)
- **8.5-8.9**: A- (Very good)
- **8.0-8.4**: B+ (Good)
- **< 8.0**: Needs improvement

### Conversation Quality Metrics

LLM-as-judge evaluates:
1. **Rapport Building** - Establishes trust before deep questions
2. **Question Quality** - Specific, behavioral, probing
3. **Follow-up Depth** - Explores key revelations
4. **Pattern Detection** - Identifies important personality markers
5. **Avoiding Assumptions** - Asks rather than assumes

---

## 🚀 Usage

### Run Full Evaluation (All 15 Personas)

```bash
cd /home/chris/selve-org/selve-chat-backend
source .venv/bin/activate
python tests/persona_evaluator.py
```

**Note:** This will take ~30-45 minutes (15 personas × 8 turns × LLM calls)

### Run Quick Test (3 Personas)

Edit `persona_evaluator.py` and modify the `main()` function:

```python
# Option 2: Quick test with 3 personas
results = await evaluator.evaluate_all_personas(
    persona_ids=[
        "persona_001_anxious_perfectionist",
        "persona_002_charismatic_entrepreneur",
        "persona_013_balanced_therapist",
    ]
)
```

Then run:
```bash
python tests/persona_evaluator.py
```

### Run Single Persona

```python
results = await evaluator.evaluate_all_personas(
    persona_ids=["persona_001_anxious_perfectionist"]
)
```

---

## 📊 Output

### Console Output

```
============================================================
🚀 Starting Persona Evaluation Batch
============================================================
Evaluating 15 personas...

============================================================
Evaluating: Sarah Chen (persona_001_anxious_perfectionist)
============================================================
Turn 1/8 - Persona: I've been feeling really overwhelmed at work...
Turn 1/8 - Chatbot: Tell me more about what's overwhelming you...
...

📊 Results for Sarah Chen:
  Avg Deviation: 8.2 points
  Max Deviation: 15 points
  Within 5 pts: 5/8 dimensions
  Within 10 pts: 7/8 dimensions
  Grade: ✅ GOOD
  Conversation Quality: 8.5/10
  Patterns Detected: 12
  Patterns Missed: 3

...

============================================================
📈 FINAL EVALUATION RESULTS
============================================================

🎯 Overall Score: 8.7/10 (Grade: A-)

📊 Pass Rates:
  ✅ Excellent: 8/15 (53.3%)
  ✅ Good: 5/15
  ⚠️  Acceptable: 2/15
  ❌ Poor: 0/15

📏 Deviation Metrics:
  Average Deviation: 7.4 points
  Maximum Deviation: 18 points

📐 Per-Dimension Average Deviations:
  ORPHEUS: 5.2 points
  LUMEN: 6.1 points
  LYRA: 6.8 points
  AETHER: 7.3 points
  ORIN: 7.9 points
  VARA: 8.4 points
  CHRONOS: 9.1 points
  KAEL: 9.7 points

⚠️  Target not yet met. Score 8.7/10 < 9.5/10
Need to improve by 0.8 points
============================================================
```

### JSON Output

Results saved to `tests/evaluation_results.json`:

```json
{
  "timestamp": "2025-12-29T12:34:56.789Z",
  "total_personas": 15,
  "avg_deviation_across_all": 7.4,
  "max_deviation_across_all": 18,
  "excellent_count": 8,
  "good_count": 5,
  "acceptable_count": 2,
  "poor_count": 0,
  "overall_grade": "A-",
  "overall_score": 8.7,
  "persona_results": [
    {
      "persona_id": "persona_001_anxious_perfectionist",
      "persona_name": "Sarah Chen",
      "avg_deviation": 8.2,
      "max_deviation": 15,
      "within_5_points": 5,
      "within_10_points": 7,
      "passed_excellent": false,
      "passed_good": true,
      "judge_score": 8.5,
      "key_patterns_detected": [
        "Low emotional stability evident from anxiety mentions",
        "High organization through perfectionism behaviors",
        ...
      ],
      "key_patterns_missed": [
        "Empathy towards others not fully explored",
        ...
      ]
    },
    ...
  ]
}
```

---

## 🔍 How It Works

### 1. Persona Simulation

Uses GPT-4o to simulate each persona's conversational style:
- Matches personality scores (e.g., low KAEL → passive language)
- Uses behavioral markers (stress responses, relationship patterns)
- Maintains consistent voice throughout conversation
- Gives specific examples from their life

### 2. Conversation Simulation

- 8 turns of back-and-forth dialogue
- Chatbot uses full ThinkingEngine (memory search, RAG, etc.)
- Persona simulator responds authentically based on personality
- Captures full transcript

### 3. Assessment Extraction

Since chatbot doesn't give explicit scores during chat, we use GPT-4o to analyze:
- What questions the chatbot asked (reveals what it's detecting)
- What patterns it followed up on
- Implicit observations in responses
- Extracts estimated scores for all 8 dimensions

### 4. LLM-as-Judge Evaluation

Separate GPT-4o call evaluates conversation quality:
- Compares chatbot's approach to best practices
- Checks if key personality markers were detected
- Assesses rapport building and question depth
- Returns detailed qualitative feedback

### 5. Metrics Calculation

- Deviation per dimension (absolute difference)
- Average deviation across all dimensions
- Pass/fail grades (excellent/good/acceptable/poor)
- Overall score (0-10 scale)
- Letter grade (A+ to F)

---

## 📈 Interpreting Results

### What "9.5/10" Means

**Target:** SELVE should detect personalities "like a detective" - accurate, specific, insightful.

- **9.5-10.0**: Production-ready. Trust chatbot assessments.
- **9.0-9.4**: Almost there. Minor refinements needed.
- **8.5-8.9**: Good but needs improvement in specific dimensions.
- **< 8.5**: Significant issues. Don't trust assessments yet.

### Common Issues to Watch For

1. **High CHRONOS deviation**: Chatbot confuses patience with passivity
2. **High KAEL deviation**: Assertiveness detection needs work
3. **Low conversation quality scores**: Not building rapport or asking deep questions
4. **Patterns missed**: Key behaviors not being detected

### Improvement Strategies

If score < 9.5, check:
1. **Worst-performing dimensions**: Focus improvements there
2. **Patterns consistently missed**: Add detection logic
3. **Conversation quality issues**: Improve prompts and question strategy
4. **Edge cases failing**: Add training examples for complex personas (narcissism, depression, ADHD)

---

## 🛠️ Customization

### Add New Personas

Edit `tests/test_personas.json`:

```json
{
  "id": "persona_016_new_profile",
  "name": "New Person",
  "age": 30,
  "occupation": "Job",
  "background": "Life story...",
  "ground_truth_scores": {
    "LUMEN": 50,
    "AETHER": 50,
    ...
  },
  "behavioral_markers": { ... },
  "expected_chat_samples": [ ... ],
  "assessment_expectations": { ... }
}
```

### Adjust Evaluation Parameters

In `persona_evaluator.py`:

```python
# Change number of conversation turns
conversation = await self.run_conversation_with_persona(persona, num_turns=10)

# Change grading thresholds
if score >= 9.5:  # Change target
    grade = "A+"
```

---

## 🎯 Success Criteria

**Before production deployment:**

✅ Overall score >= 9.5/10
✅ At least 80% "excellent" pass rate (12/15 personas)
✅ No "poor" grades
✅ Average deviation <= 5 points
✅ Max deviation <= 15 points on any dimension
✅ Conversation quality >= 8.0/10 average
✅ Key psychological patterns detected (depression, narcissism, ADHD, burnout)

---

## 🚨 Known Limitations

1. **Simulated personas aren't real users**: GPT-4o simulation approximates but doesn't perfectly replicate human complexity
2. **Chatbot doesn't give explicit scores**: Extraction via LLM-as-judge adds uncertainty
3. **Cultural bias**: Personas skew Western/English-speaking
4. **Age range**: Limited adolescent and elderly representation
5. **No severe mental illness**: Personas don't include schizophrenia, bipolar (type 1), severe trauma

---

## 📚 Related Documentation

- `/home/chris/selve-org/goal.md` - Project goals and requirements
- `/home/chris/selve-org/selve/backend/app/routes/assessment/constants.py` - SELVE dimensions
- `/home/chris/selve-org/selve/backend/app/narratives/dimensions/README.md` - Dimension descriptions

---

## 🎉 Target Achievement

When you see this:

```
🎉 TARGET MET! Score 9.52/10 >= 9.5/10
SELVE is DECIPHERING ACCURATELY like a detective! ✅
```

**You're ready for production!** 🚀

The chatbot can accurately assess personalities through conversation and provide insights that are:
- ✅ Uncomfortably accurate
- ✅ Specific and behavioral
- ✅ Detecting mental health patterns
- ✅ Useful for personal growth

---

**Happy Testing! Let's get to 9.5/10!** 🎯
