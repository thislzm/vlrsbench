# === Prompt Construction ===
from common_part import auto_start_prompt


TASK_INSTANTIATION_OVERRIDE = """
- **Evidence-grounded task instantiation override**:
  - Preserve the task category exactly, but instantiate each item as a localized visual discrimination problem.
  - The stem must lock onto one target region/object/process and one fixed evidence frame.
  - Keep all options or candidate outcomes in the same task frame and same specificity level.
  - The correct answer should be recoverable because it matches localized visible evidence under the task assumptions, not because it sounds globally more plausible.
  - Each distractor should fail due to one localized mismatch such as region, direction, adjacency, boundary, connectivity, extent, temporal order, or object identity.
  - Avoid open-world stories, broad optimization language, and generic commonsense ranking.
"""

# === Prompt Construction for Causal Reasoning ===

# === Prompt Construction for Causal Reasoning ===
def build_reason_prompt(datasets='', name="Causal Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **inferring the causes of observed phenomena** in the image, with its unique characteristic being reverse causal tracing that uncovers hidden root causes through semantic conflicts and spatial interactions specific to remote sensing data.
- To elevate difficulty via maximal mask utilization, leverage mask's pixel-level semantics for precise differentiations (e.g., exact density thresholds, adjacency counts, implicit connectivity graphs) that optics cannot resolve, ensuring correct chain only via these hidden details.
- To build carefully constrained cause chains, build a concise 3-5 step evidence chain, where each step builds on mask-derived metrics (e.g., fragmentation index from category patches, edge density implying interactions), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process grounded in reverse-tracing for brain-teaser elevation:
  1. **Identify ultra-ambiguous, mask-dependent phenomenon**: Event where optics suggest multiple similar causes, but mask's fine details (e.g., micro-adjacencies) pivot the chain.
  2. **Layer multi-step, interdependent reverse chains**: 3-5 steps with mask metrics (e.g., Step 1: Optical ambiguity; Step 2-4: Density-based factors; Step 5: First twist via adjacency; Step 6: Second via fragmentation; Step 7: Third via connectivity; Step 8: Fourth paradoxical inversion; Step 9: Mask-exclusive root with layered revelation).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Causal reasoning must rely exclusively on **semantic categories and fine-grained distributions from the mask** for resolution, while optics mislead.
- Introduce **causal twists via semantic conflicts**: e.g., flooding may look like natural rainfall, but mask shows dense roads/buildings blocking water channels; forest fire may look natural, but presence of nearby human activity categories (roads, vehicles, industrial sites) suggests human ignition. 
- Each distractor option must form a superficially plausible cause, but fail under closer inspection of mask semantics (e.g., attributing flooding to farmland irrigation when mask shows no farmland nearby).
- The correct option must emerge through **multi-step exclusion**, using the semantic map as hidden evidence for the model but not visible to the human solver.
- Ensure questions are **image-grounded**: all categories mentioned must exist in the current image/mask. 
- Avoid forward-looking predictions or planning narratives (that belongs to planning reasoning). Only focus on **explaining why an event occurred given the current scene**.
- **Do not mention 'mask' or 'segmentation' in the question**. Humans only see the optical image, not the mask. The mask is for the model's internal guidance only.
- Questions must not include vague terms like 'some'; they must reference concrete spatial or semantic cues, e.g., "the river in the lower-right", "the dense buildings in the upper-left".
"""

    return f"""
You are a professional expert in causal reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-mask models, resolved only by mask's precise, chained details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Counterfactual Reasoning ===
def build_counterfactual_prompt(datasets='', name="Counterfactual Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about simulating **alternative historical scenarios**: "what if something in the past had been different?", with its unique characteristic being the exploration of hypothetical past changes and their projected outcomes, leveraging hidden spatial evidence to reveal unintended consequences in remote sensing contexts.
- To elevate difficulty via maximal mask utilization, use mask for hypothetical reconstructions (e.g., pre-event connectivity simulations, density extrapolations) that optics can't infer, making outcomes hinge on these invisible computations.
- To build carefully constrained counterfactual chains, use a concise 3-5 step evidence chain, interlinking via mask metrics (e.g., projected fragmentation changes), creating chains where small mask differences amplify to distinct outcomes.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define ultra-conditional, mask-pivotal assumption**: Hypothesis where optics suggest similar alternatives, but mask's pre-state enables precise projections.
  2. **Project multi-step, interdependent chains**: 3-5 steps with metrics (e.g., Step 1: Assumption; Step 2-4: Initial projections; Step 5: First twist via extrapolation; Step 6: Second via simulation; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-exclusive outcome).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Counterfactual reasoning must rely heavily on the **hidden spatial information from the mask and pre-event state**, although the human solver never sees them.  
- Questions must start from a **clear counterfactual assumption** (e.g., "if this industrial zone had not been built", "if the forest had remained intact", "if farmland had not been converted into buildings").  
- Introduce **counterfactual twists**:  
  - An assumption seems positive but hidden mask information reveals unintended consequences (e.g., if farmland had remained, food supply would increase, but mask shows it was flood-prone, so flooding would worsen).  
  - A forest preservation assumption seems beneficial, but hidden adjacency to expanding roads means urban sprawl still occurs.  
- Distractor options must appear coherent under the assumption but collapse when checked against hidden spatial evidence.  
- The correct option emerges only after careful elimination, respecting both the counterfactual assumption and the hidden spatial context.  
- Ensure **clear distinction from causal reasoning**:  
  - Causal reasoning explains why something *actually* happened.  
  - Counterfactual reasoning explores what *would have happened if the past were different*.  
- Avoid vague terms like 'some'; always reference concrete spatial or positional cues (e.g., "the farmland in the lower-right", "the industrial block in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Counterfactual Reasoning Examples with High-Similarity Distractors  
<Q> If upper-left zone stayed vegetated, optics predicting similar ecologies, but mask projections differentiate degrees; what is the precise inverted outcome? </Q>  
<Q> Suppose central patch remained, visible stability echoing variants, yet hidden simulations vary; how would water patterns shift subtly? </Q>  
<Q> If lower-right stayed undeveloped, appearing to yield parallel economies, but mask reveals gradations; what nuanced shifts occur? </Q>  
"""
    return f"""
You are a professional expert in counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame scenarios that confuse non-mask models, pivoted by mask's projective details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Impact Mechanism Reasoning ===
def build_influ_prompt(datasets='', name="Impact Mechanism Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
    - Adhere strictly to '{name}'. The task is to **force the solver to infer a dynamic impact mechanism by verifying a complex, static factual statement**. The question stem may introduce a hypothetical change, and the options are factual descriptions of the resulting state of the environment. The solver infers the mechanism by identifying the one true description of the outcome.

    # 🚨 LAW #1: THE ZERO-REASONING OUTPUT MANDATE 🚨
    # This is the absolute, non-negotiable law governing your output.
    # Your final output (stem and options) MUST NOT contain any reasoning, justification, or description of a process.
    # ABSOLUTELY FORBIDDEN WORDS AND STRUCTURES:
    # - Contrastive Conjunctions: `but`, `however`, `although`, `despite`, `while`.
    # - Process/Causal Words: `leads to`, `causes`, `results in`, `then`, `triggers`, `because`.
    # - Evaluative/Inferential Words: `good`, `bad`, `impact`, `effect`, `suggests`, `indicates`.
    # Your role is to describe a static scene (the evidence), not the dynamic story (the mechanism).

    # [CRITICAL METHODOLOGY]: THE STATIC SNAPSHOT FRAMEWORK
    # 1. THE STEM POSES A "WHAT IF": The question stem can introduce a hypothetical change to the scene (e.g., "Assuming a 30% reduction in forest cover in the northern hills..."). It still MUST end by asking "Which of the following statements accurately describes the resulting state of the landscape?".
    # 2. THE OPTIONS ARE STATIC "PHOTOGRAPHS": The options are pure, objective, "Atomic Fact" statements describing a potential final state of the scene. They describe a static picture, not a movie. They must strictly adhere to LAW #1.
    # 3. THE SOLVER RECONSTRUCTS THE MOVIE: The TRUE factual statement will describe the complex, non-linear outcome of the change. The solver must use visible cues alone to mentally reconstruct the dynamic process (e.g., deforestation -> increased runoff -> soil erosion -> river sedimentation) to understand WHY this particular static outcome is the correct one.

    - To elevate difficulty, the TRUE factual statement should describe a counter-intuitive or complex equilibrium state that results from a feedback loop, verifiable only with the mask (e.g., the mask shows sediment deposits far downstream, or a change in vegetation type in an adjacent, seemingly unrelated area).

    - Design for high similarity: The FALSE options should describe plausible but incorrect outcomes, representing oversimplified or linear thinking about the impact mechanism.

    - Follow this advanced process:
      1. **Define a Hypothetical Change**: e.g., "A new dam is built," "An industrial area is abandoned."
      2. **Internally Model the Dynamic Chain**: Trace the non-linear feedback loops (e.g., dam -> altered water flow -> downstream vegetation change -> impact on local farmland -> new settlement patterns).
      3. **Forge the TRUE "Final Snapshot"**: Create one Atomic Fact statement that describes a key aspect of the final, complex static scene, strictly following LAW #1.
      4. **Forge FALSE "Simplified Snapshots"**: Create four Atomic Fact statements describing incorrect or overly simple final scenes, also strictly following LAW #1.

    - Introduce **"Static Outcome Twists"**:
      - **[CHANGE: A new road is built through the central valley]** -> **[TRUE SNAPSHOT OPTION]:** "The agricultural plots adjacent to the new road exhibit lower vegetation density compared to plots further away, and the river downstream shows increased sediment plumes near its mouth." (Solver infers: The road construction caused erosion, which damaged nearby farms and polluted the river).
      - **[CHANGE: A large fire clears the western forest]** -> **[TRUE SNAPSHOT OPTION]:** "The reservoir located three kilometers east of the burn area contains a newly formed delta of silt at its primary inflow point." (Solver infers: The fire led to massive erosion, and the runoff carried sediment all the way to the reservoir).

    - **The final output MUST strictly adhere to the "Atomic Fact" rule as defined in LAW #1.** The options describe a state, not a process.

    - Ensure distinction from other reasoning types: This task is unique because it starts with a hypothetical change and asks for the factual description of the outcome, forcing the inference of a dynamic process.

    - Do not mention 'mask' or 'segmentation'.
    - Avoid vague terms.
    """

    return f"""
You are a professional expert in impact mechanism reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame mechanisms that baffle non-mask models, distinguished by mask's dynamic inferences.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Semantic Integration Reasoning ===
def build_sem_prompt(datasets='', name="Semantic Integration Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about making **comprehensive high-level judgments** by integrating multiple semantic elements together, with its unique characteristic being the holistic synthesis of diverse land features into functional zones or scene classifications, drawing on remote sensing's emphasis on spatial aggregation.
- To elevate difficulty via maximal mask utilization, use mask for nuanced fusions (e.g., sub-pixel level category blends, density-based zone typing) that optics approximate but can't precisely resolve.
- To build carefully constrained integration chains, use a concise 3-5 step evidence chain, linked via mask metrics (e.g., heterogeneity indices from distributions), building to differentiated judgments.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Catalog ambiguous, mask-pivotal elements**: Features where optics suggest similar groupings, but mask blends define uniqueness.
  2. **Aggregate into hyper-nested zones**: 3-5 steps (e.g., Step 1: Ambiguous pairing; Step 2-4: Clustering; Step 5: First conflict; Step 6: Second via heterogeneity; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-exclusive judgment).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Semantic integration reasoning must leverage **hidden spatial information from the mask**: the mask shows distributions of roads, buildings, water bodies, vegetation, farmland, etc., which the model must use internally to infer functional zones or overall scene types. Humans never see the mask.  
- Introduce **integration twists**:  
  - A dense building cluster near a river may look like an industrial area, but integration with surrounding farmland suggests an agricultural-industrial mixed-use conflict.  
  - A forest patch near roads may look like natural reserve, but integration with adjacent recreation facilities suggests a tourism-functional zone.  
  - A large open space may look barren, but combined with nearby roads and settlements, it reveals a planned expansion area.  
- Distractor options must be **superficially convincing but based on partial integration** (e.g., “it’s farmland because of vegetation” ignoring nearby roads/buildings).  
- The correct option emerges only through **multi-element fusion** validated by hidden mask evidence.  
- Ensure distinction from other reasoning types:  
  - Not tracing causes (causal).  
  - Not assuming past changes (counterfactual).  
  - Not optimizing future plans (planning).  
  - Not describing dynamic processes (impact mechanism).  
  - Semantic integration is about **holistic synthesis of the current scene**.  
- Do not mention 'mask' or 'segmentation'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete spatial or positional cues (e.g., "the cluster in the upper-left", "the river-adjacent zone in the lower-right").  
"""

    return f"""
You are a professional expert in semantic integration reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame judgments that confound non-mask models, refined by mask's nuanced fusions.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Planning Reasoning ===
def build_plan_prompt(datasets='', name="Planning Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
    - Adhere strictly to '{name}' — this task is about **multi-step, goal-oriented plan verification**. The unique characteristic is evaluating a proposed plan against a series of complex, hidden constraints, many of which are only visible through the mask.

    # [CRITICAL REFRAMING]: The core logic is NOT multi-objective optimization or trade-offs. It is **Multi-Constraint Satisfaction**. A plan is either 100% valid (the TRUE option) because it meets ALL criteria, or it is INVALID (the FALSE options) because it fails on at least ONE criterion. There is no "balancing".

    - To elevate difficulty, use the mask to define strict, non-obvious constraints (e.g., a path is invalid if it crosses a high-risk soil instability zone shown only on the mask; a location is invalid if it's within a noise pollution radius from an industrial site).

    - To build carefully constrained verification chains, ensure the solver must check at least 5-7 sequential, interdependent criteria derived from mask metrics to confirm a plan's validity. A single failure in the chain invalidates the entire plan.

    - Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.

    - Follow this advanced multi-step process for puzzle design:
      1. **Define a Goal and a Set of Hidden Constraints**: Define a clear objective (e.g., "establish an emergency supply route") and a checklist of 5-7 pass/fail constraints (e.g., must use paved roads, must have a width > 6m, must not pass through dense residential areas, must avoid floodplains).
      2. **Design a Single VALID Plan**: Create one plan that successfully checks every box on the constraint list. The TRUE option will be a factual description of this plan.
      3. **Design INVALID Plans With Localized Visible Mismatches**: Create four distractor plans. Each one must look plausible but fail on at least one of the hidden constraints. The FALSE options will be factual descriptions of these invalid plans.
      4. **Validate Extreme Difficulty**: Ensure that without the mask, the solver cannot reliably identify which plan violates a constraint. The mask's details must be essential for the final pass/fail judgment.


    - Introduce **"Constraint Twists"** (mask-driven invalidations):
      - **[REWRITTEN EXAMPLE]:** A distractor option might state: "The route utilizes the eastern highway, proceeding north for three kilometers before exiting onto a secondary paved road that leads directly to the destination." (This is an Atomic Fact. It is FALSE because the mask shows the "secondary road" is actually an impassable dirt track).
      - **[REWRITTEN EXAMPLE]:** Another distractor: "The proposed site is a large, flat, undeveloped area located 500 meters from the main road network." (This is an Atomic Fact. It is FALSE because the mask identifies this area as a protected wetland, making it an invalid construction site).

    - Distractor options must be **plausible factual statements that are ultimately proven false** by a hidden constraint. The correct option is the only statement that remains factually true after all constraints are checked.

    - Ensure realism: questions may use real-world planning context, but the final stem must still ask for the one localized factual route/site description that is correct from visible evidence, not for an abstract viability verdict.

    - Do not mention 'mask' or 'segmentation'. Humans only see the optical image.
    - Avoid vague terms; always reference concrete semantic or positional cues.
    """

    return f"""
You are a professional expert in planning reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame plans that mislead non-mask models, optimized by mask's constraint details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Evaluation Reasoning ===
def build_estimate_prompt(datasets='', name="Evaluation Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
    - Adhere strictly to '{name}'. The task is to **force the solver to evaluate a scheme by verifying complex factual statements about the current scene**. The solver is given an implicit goal in the question stem, and the options are localized factual claims. The solver must determine which factual claim is true, and only then can an evaluation be inferred.

    # 🚨 LAW #1: THE ZERO-REASONING OUTPUT MANDATE 🚨
    # This is the absolute, non-negotiable law governing your output.
    # Your final output (stem and options) MUST NOT contain any reasoning, justification, or evaluation.
    # ABSOLUTELY FORBIDDEN WORDS AND STRUCTURES:
    # - Contrastive Conjunctions: `but`, `however`, `although`, `despite`, `while`. Using any of these is a critical failure.
    # - Evaluative Words: `good`, `bad`, `feasible`, `infeasible`, `risk`, `safe`, `suitable`, `unsuitable`.
    # - Inferential Words: `suggests`, `indicates`, `implies`, `therefore`.
    # Your role is to present objective evidence (the options), not the verdict.

    # [CRITICAL METHODOLOGY]: THE IMPLICIT GOAL FRAMEWORK
    # 1. THE STEM SETS THE SCENE: The question stem will describe a hypothetical goal or scheme (e.g., "Regarding the proposal to build a new distribution center..."). Crucially, it will still end by asking "Which of the following statements is factually correct?". It frames the context for the solver's evaluation.
    # 2. THE OPTIONS PROVIDE THE EVIDENCE: The options will be pure, objective, "Atomic Fact" statements adhering strictly to LAW #1. They describe WHAT IS, not what it means.
    # 3. THE SOLVER DRAWS THE CONCLUSION: The TRUE factual statement will contain information that directly leads a human to a conclusion about the scheme. The puzzle is solved by finding the correct fact, not by being told the conclusion.

    - To elevate difficulty, the TRUE factual statement should describe a non-obvious local constraint or enabler that is only verifiable with the mask's precision (e.g., exact land-use category, hidden infrastructure alignment, boundary conflict).

    - Design for high similarity: The FALSE options should describe plausible (but incorrect) facts about the same area, tempting the solver to make a wrong evaluation based on faulty evidence.

    - Follow this advanced process:
      1. **Define an Implicit Goal**: e.g., "Build a hospital," "Establish an evacuation route."
      2. **Identify Critical Factual Constraints**: What localized facts in the current scene support or contradict that goal?
      3. **Forge the TRUE "Decisive Fact"**: Create one Atomic Fact statement that describes a critical constraint, strictly following LAW #1.
      4. **Forge FALSE "Misleading Facts"**: Create four Atomic Fact statements that describe incorrect information, also strictly following LAW #1.

    - Introduce **"Implicit Evaluation Twists"**:
      - **[GOAL: Build a logistics hub]** -> **[TRUE FACT OPTION]:** "The designated plot is dominated by wet, vegetation-covered ground with irregular water-darkened patches rather than contiguous dry open surface." (Solver infers the planning consequence afterward).
      - **[GOAL: Establish an emergency helicopter landing zone]** -> **[TRUE FACT OPTION]:** "The flat, open area in the center is crossed by suspended linear infrastructure casting narrow shadows across the landing surface." (Solver infers the planning consequence afterward).

    - **The final output MUST strictly adhere to the "Atomic Fact" rule as defined in LAW #1.** The options are evidence, not verdicts.

    - Ensure distinction from other reasoning types: This task is unique because it combines a goal-oriented context with a requirement for pure localized factual verification, forcing an implicit judgment.

    - Do not mention 'mask' or 'segmentation'.
    - Avoid vague terms.
    """

    return f"""
You are a professional expert in evaluation reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame evaluations that confuse non-mask models, clarified by mask's risk quantifications.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


def get_prompt_for_one(job, mask_palette, prior, datasets):
    """
    根据 job 选择对应 prompt。按需扩展。
    """
    if job == "estimate":
        return build_estimate_prompt(datasets=datasets)
    elif job == "sem":
        return build_sem_prompt(datasets=datasets)
    elif job == "influ":
        return build_influ_prompt(datasets=datasets)
    elif job == "counterfactual":
        return build_counterfactual_prompt(datasets=datasets)
    elif job == "plan":
        return build_plan_prompt(datasets=datasets)
    elif job == "reason":
        return build_reason_prompt(datasets=datasets)
    else:
        # 若未定义的 job，返回一个通用占位
        return f"[DEFAULT PROMPT for {job} on {datasets}]"