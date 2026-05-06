# 预测能力
################################################# 类别状态预测推理（对象级别）（对象级别，三时相）
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

SPACENET7_REASONING_HARDENING = """
- **spaceNet7 building-only hardening**:
  - Because this dataset only changes in the building category, do **not** reduce the item to a bare statement about whether a building appeared, disappeared, or remained unchanged.
  - Every claim must combine the temporal change with at least two extra RGB-verifiable constraints chosen from: attachment vs isolated infill, gap closure between persistent roofs, extension direction relative to a road/canal/lot edge, continuity or interruption along a row/corridor, boundary crossing, count threshold, or asymmetry between two nearby building groups.
  - Prefer two candidate building subregions that look similar at first glance, so the solver must compare local structure across frames rather than answer from a single obvious change cue.
  - If the claim can be judged from one coarse endpoint glance alone, reject it and rewrite it with a stable landmark and a second local constraint.
"""


# === Prompt Construction for Multi-Temporal Class State Prediction Reasoning ===

# === Prompt Construction for Multi-Temporal Class State Prediction Reasoning ===
def build_multitemporal_class_state_prediction_prompt(
    datasets="",
    name="Multi-Temporal Class State Prediction Reasoning",
    interval_1to2="one year",
    interval_2to3="one year"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to the **building category only**.  
  - Therefore, all predictions must focus exclusively on **building-related state changes**  
    (e.g., new construction, demolition, persistence, densification) over the time interval of {interval_2to3} after im2.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple land-cover categories (farmland, forest, water, residential, industrial, etc.).  
  - Predictions must therefore consider **multi-class state changes** for objects over the time interval of {interval_2to3} after im2.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about predicting the **future category state** of a given object {interval_2to3} after im2, using multi-temporal hidden evidence to uncover trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im1 ({interval_1to2} before im2) and multi-mask (mask1, mask2, mask3), leverage temporal mask differences for precise trend extrapolations (e.g., category transition rates over {interval_1to2}, density change thresholds, implicit evolution graphs across phases) that visible im1/im2 alone cannot resolve, using these hidden details only to mine ambiguous regions, construct close alternatives, and verify that the final item remains solvable from visible RGB evidence.
- To build carefully constrained prediction chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal fragmentation index from mask1 to mask2 over {interval_1to2}, projected density shifts for {interval_2to3}), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{interval_1to2} before im2 (current time), the state was X; predict Y {interval_2to3} after im2"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trend**: Trajectory where visible im1/im2 suggest multiple similar states, but hidden im1/mask1 ({interval_1to2} prior) pivots the extrapolation.
  2. **Layer multi-step prediction chains**: 3-5 steps with sub-layers and temporal twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {interval_1to2}; Step 5: First twist via transition rates; Step 6: Second via density shifts; Step 7: Third via evolution graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future state for {interval_2to3}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the **future category state** of a given object {interval_2to3} after the current visible image (im2).  
- Solver sees im1 ({interval_1to2} earlier) and im2 (current visible image).  
- Hidden im3 ({interval_2to3} after im2) is used internally, together with masks, to validate the future state.  

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Detect how the object changed from hidden im1 ({interval_1to2} earlier) to visible im2.  
  2. Examine surrounding regions (e.g., proximity to roads, rivers, urban edges).  
  3. Combine trajectory + context → predict im3 category {interval_2to3} later.  
  4. Ensure distractor categories are plausible but contradicted by spatial or semantic context.  

- Example twists:  
  - A farmland shrinking in im2, surrounded by residential expansion, is unlikely to remain farmland → im3 becomes residential or industrial {interval_2to3} later.  
  - A forest thinning in im2, adjacent to new infrastructure, is unlikely to stay forest → im3 becomes bare land or farmland {interval_2to3} later.  
  - A wetland shrinking in im2, adjacent to urbanization, is unlikely to recover → im3 becomes urban {interval_2to3} later.  
"""
    examples = f"""
### Example Questions  

<Q> From im1 to im2 over {interval_1to2}, the central zone changed from intact farmland to scattered houses.  
Now predict what the category will be {interval_2to3} after im2. </Q>  

<Q> From im1 to im2 over {interval_1to2}, the upper-left forest thinned.  
Now predict what the category will be {interval_2to3} after im2. </Q>  
"""
    return f"""
You are a professional expert in multi-temporal class state prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im1 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Multi-Temporal Object Shape Prediction Reasoning ===
def build_multitemporal_shape_prediction_prompt(
    datasets="",
    name="Multi-Temporal Object Shape Prediction Reasoning",
    interval_1to2="one year",
    interval_2to3="one year"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all predictions must focus exclusively on **building-related shape changes**  
    (e.g., expansion of building clusters, demolition leading to shrinkage, densification patterns).  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple land-cover categories (farmland, forest, water, residential, industrial, etc.).  
  - Predictions must therefore consider **multi-class object shape changes** (expansion, shrinkage, densification, fragmentation).  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about predicting the **future spatial shape** of a given object {interval_2to3} after im2, using multi-temporal hidden evidence to project geometric trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im1 ({interval_1to2} before im2) and multi-mask, leverage temporal mask geometries for precise shape extrapolations (e.g., boundary change rates over {interval_1to2}, fragmentation thresholds, implicit morphology graphs across phases) that visible im1/im2 alone cannot resolve.
- To build carefully constrained shape prediction chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal shape index from mask1 to mask2 over {interval_1to2}, projected boundary shifts for {interval_2to3}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{interval_1to2} before im2 (current time), the shape was X; predict Y {interval_2to3} after im2"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trajectory**: Geometry where visible im1/im2 suggest multiple similar shapes, but hidden im1/mask1 ({interval_1to2} prior) pivots the projection.
  2. **Layer multi-step prediction chains**: 3-5 steps with sub-layers and geometric twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial shapes from mask1-2 over {interval_1to2}; Step 5: First twist via boundary rates; Step 6: Second via fragmentation; Step 7: Third via morphology; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future shape for {interval_2to3}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the **future spatial shape** of a given object {interval_2to3} after im2.  
- Solver sees im1 ({interval_1to2} earlier) and im2 (current visible image).  
- Hidden im3 ({interval_2to3} after im2) is used internally, together with masks, to validate the future trajectory.  

- Reasoning chain:  
  1. Track object geometry from hidden im1 ({interval_1to2} earlier) → visible im2 (e.g., forest patch splitting, river widening).  
  2. Assess external pressures (urban sprawl, agricultural expansion, hydrological changes).  
  3. Predict how the object’s spatial shape evolves into im3 {interval_2to3} later (expansion, shrinkage, fragmentation, merging).  
  4. Distractors: plausible but geometrically or contextually inconsistent.  

- Example twists:  
  - A forest patch continuous in im1 ({interval_1to2} earlier), fragmented in im2, near urban expansion → im3 likely further fragmented into isolated patches {interval_2to3} later.  
  - A water body shrinking in im2 due to farmland drainage → im3 likely further reduced, not suddenly expanding {interval_2to3} later.  
  - A residential cluster growing denser in im2, near a highway → im3 likely expands further, not shrinks {interval_2to3} later.  
"""
    examples = f"""
### Example Questions  

<Q> From im1 to im2 over {interval_1to2}, a continuous forest patch became fragmented.  
Now predict what the spatial shape will be {interval_2to3} after im2. </Q>  

<Q> From im1 to im2 over {interval_1to2}, the river narrowed and farmland expanded.  
Now predict how the river and its banks will evolve {interval_2to3} after im2. </Q>  
"""
    return f"""
You are a professional expert in multi-temporal object shape prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im1 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Scenario Uncertainty Prediction Reasoning ===
def build_multitemporal_uncertainty_prediction_prompt(
    datasets="",
    name="Scenario Uncertainty Prediction Reasoning",
    interval_1to2="one year",
    interval_2to3="one year"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all uncertainty predictions must focus exclusively on **building-related scenario outcomes**  
    (e.g., new construction patterns, demolition vs persistence, densification trends) over the future {interval_2to3}.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Uncertainty predictions must therefore consider **multi-class scenario outcomes** (e.g., farmland urbanization vs ecological protection, forest regrowth vs clearance, water expansion vs stability) over the future {interval_2to3}.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about selecting the **future scenario best supported by the visible trajectory** among multiple fixed alternatives {interval_2to3} after im2, using multi-temporal hidden evidence to resolve uncertainties in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im3 ({interval_2to3} after im2) and multi-mask, leverage temporal mask uncertainties for precise scenario branching (e.g., probabilistic transition matrices over {interval_1to2}, density uncertainty thresholds, implicit trend graphs across phases) that visible im1/im2 alone cannot resolve.
- To build carefully constrained uncertainty chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal uncertainty index from mask1 to mask2 over {interval_1to2}, projected branching probabilities for {interval_2to3}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{interval_1to2} before im2 (now), the trend was X; predict Y scenario {interval_2to3} after im2"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trajectory**: Trends where visible im1/im2 suggest multiple similar scenarios, but hidden mask3 ({interval_2to3} later) pivots the probabilities.
  2. **Layer multi-step uncertainty chains**: 3-5 steps with sub-layers and probabilistic twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {interval_1to2}; Step 5: First twist via transition probabilities; Step 6: Second via density uncertainties; Step 7: Third via trend graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive scenario for {interval_2to3}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the most plausible **future scenario** among multiple alternatives {interval_2to3} after im2.  
- Solver sees im1 ("{interval_1to2} earlier") and im2 ("now").  
- im3 ("{interval_2to3} later") hidden, used for validation.  

- How to use masks:  
  - Use mask1/mask2 to identify multiple evolving trends (e.g., farmland shrinking, forest thinning, water expansion).  
  - Use mask3 to validate which scenario materializes.  
  - Distractors must be scenarios that seem plausible but contradict hidden evidence.  

- Reasoning chain:  
  1. Analyze im1→im2 to detect diverging trends.  
  2. Generate multiple plausible im3 scenarios.  
  3. Select the one consistent with hidden semantics + evolution logic.  
"""
    examples = f"""
### Example Questions  

<Q> From im1 to im2 over {interval_1to2}, the central farmland was fragmented by scattered houses, northern forest thinned, southern river widened.  
Now predict which future scenario is best supported {interval_2to3} after im2: full urban expansion, ecological restoration, or mixed agro-urban land use? </Q>  

<Q> From im1 to im2 over {interval_1to2}, the eastern industrial zone expanded toward central farmland with highway construction.  
Now predict which future scenario is best supported {interval_2to3} after im2: stabilized industry, farmland persistence, or full industrial takeover? </Q>  
"""
    return f"""
You are a professional expert in scenario uncertainty prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im3 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Spatio-Temporal Sequence Prediction Reasoning ===
def build_multitemporal_sequence_prediction_prompt(
    datasets="",
    name="Spatio-Temporal Sequence Prediction Reasoning",
    interval_1to2="one year",
    interval_2to3="one year"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all sequence predictions must focus exclusively on **building-related temporal trends**  
    (e.g., continuous expansion of building clusters, ongoing densification, demolition-reconstruction cycles) projected {interval_2to3} after im2.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Sequence predictions must therefore consider **multi-class temporal trends** (e.g., farmland → scattered houses → dense residential, forest → bare land → agriculture, water body shrinkage → further shrinkage) projected {interval_2to3} after im2.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about extrapolating deterministic **temporal sequences** im1→im2→im3, using multi-temporal hidden evidence to project sustained trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im3 ({interval_2to3} after im2) and multi-mask, leverage temporal mask sequences for precise trend continuations (e.g., consistent change vectors over {interval_1to2}, density evolution thresholds, implicit sequence graphs across phases) that visible im1/im2 alone cannot resolve.
- To build carefully constrained sequence chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal trend index from mask1 to mask2 over {interval_1to2}, projected vector extensions for {interval_2to3}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{interval_1to2} before im2 (now), the sequence was X; predict Y {interval_2to3} after im2"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trend**: Sequence where visible im1/im2 suggest multiple similar continuations, but hidden mask3 ({interval_2to3} later) pivots the projection.
  2. **Layer multi-step sequence chains**: 3-5 steps with sub-layers and trend twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {interval_1to2}; Step 5: First twist via change vectors; Step 6: Second via density evolutions; Step 7: Third via sequence graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future state for {interval_2to3}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: extrapolate deterministic **temporal sequences** im1→im2→im3.  
- Solver sees im1 ("{interval_1to2} earlier") and im2 ("now").  
- im3 ("{interval_2to3} later") hidden, used for validation.  

- How to use masks:  
  - Use mask1/mask2 to detect sustained trends (steady shrinkage, continuous expansion).  
  - Use mask3 to ensure deterministic continuation.  
  - Distractors = trend reversals without cause.  

- Reasoning chain:  
  1. Detect im1→im2 consistent trends.  
  2. Extrapolate deterministically into im3.  
  3. Eliminate trend reversals as implausible.  
"""
    examples = f"""
### Example Questions  

<Q> From im1 to im2 over {interval_1to2}, the central farmland shrank and was replaced by new residential blocks.  
Now predict what deterministic state it will reach {interval_2to3} after im2: restored farmland, stabilized mixed use, or fully dense residential? </Q>  

<Q> From im1 to im2 over {interval_1to2}, the upper-left forest thinned continuously.  
Now predict what deterministic state it will reach {interval_2to3} after im2: dense forest, bare land, or cultivated farmland? </Q>  
"""
    return f"""
You are a professional expert in spatio-temporal sequence prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im3 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Spatio-Temporal Causal Chain Traceback Reasoning ===
def build_multitemporal_evolution_prompt(
    datasets="",
    name="Multi-Temporal Evolution Reasoning",
    interval_1to2="one year",
    interval_2to3="one year",
    interval_total="two years"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to the **building category only**.  
  - Therefore, all evolution reasoning must focus exclusively on **building-related changes**  
    (e.g., construction patterns, densification) over the total interval of {interval_total} from im1 to im3.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Evolution reasoning must therefore consider **multi-class changes** over the total interval of {interval_total} from im1 to im3.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about describing the **evolution** of the scene from the initial image (im1) to the final image over the total {interval_total}, using multi-temporal hidden evidence to uncover intermediate trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im2 (after {interval_1to2} from im1 and before {interval_2to3} to im3) and multi-mask (mask1, mask2, mask3), leverage temporal mask evolutions for precise intermediate insights (e.g., change rates over {interval_1to2}, transition thresholds over {interval_2to3}, implicit evolution graphs across phases) that visible im1/im3 cannot resolve, ensuring correct description only via these hidden details.
- To build carefully constrained evolution chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal change index from mask1 to mask2 over {interval_1to2}, projected evolutions for {interval_2to3} to mask3), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the total time interval (e.g., "From the initial image over {interval_total} to the final image, the evolution was X reaching Y"), ensuring solvers reason with temporal context, but do not mention hidden im2 or sub-intervals in the question text itself—reserve for internal logic and options only.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent evolution**: Trajectory where visible im1/im3 suggest multiple similar changes over {interval_total}, but hidden im2/mask2 pivots the description.
  2. **Layer multi-step evolution chains**: 3-5 steps with sub-layers and temporal twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial changes from mask1-2 over {interval_1to2}; Step 5: First twist via rate changes; Step 6: Second via thresholds; Step 7: Third via graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive full evolution over {interval_total}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: describe the **evolution** of a given object or scene from the initial image to the final image over the total {interval_total}.  
- Solver sees the initial image (im1) and the final image (im3).  
- Hidden im2 (intermediate) is used internally to ground the evolution.  

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Analyze visible changes from the initial image to the final image over {interval_total}.  
  2. Infer hidden intermediate states using masks.  
  3. Describe the full evolution path, ensuring consistency with total interval.  
  4. Ensure distractor descriptions are plausible but contradicted by hidden context.  

- Example twists:  
  - From initial to final over {interval_total}, farmland appears urbanized, but hidden intermediate shows gradual residential growth then acceleration.  
  - From initial to final over {interval_total}, forest appears thinned, but hidden intermediate reveals clearance followed by partial regrowth.  
"""
    examples = f"""
### Example Questions  

<Q> The initial image shows the central zone as mostly farmland with open fields, and the final image over {interval_total} reveals a dense residential area with buildings and roads. How did this area evolve during that period? </Q>  

<Q> In the initial image, the upper-left is a dense forest, and the final image over {interval_total} shows it thinned with bare patches and nearby development. How did this forest change over time? </Q>  
"""
    return f"""
You are a professional expert in multi-temporal evolution reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im2 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Multi-Temporal Consistency Reasoning ===
def build_multitemporal_consistency_prompt(
    datasets="",
    name="Multi-Temporal Consistency Reasoning",
    interval_1to2="one year",
    interval_2to3="one year",
    interval_total="two years"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all consistency reasoning must focus exclusively on **building-related consistency**  
    (e.g., consistent densification, persistent structures) over the total interval of {interval_total} from im1 to im3.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Consistency reasoning must therefore consider **multi-class consistency** over the total interval of {interval_total} from im1 to im3.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about evaluating the **consistency** of changes from the initial image to the final image over the total {interval_total}, using multi-temporal hidden evidence to verify logical continuity in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im2 (after {interval_1to2} from im1 and before {interval_2to3} to im3) and multi-mask, leverage temporal mask consistencies for precise verification (e.g., continuity rates over {interval_1to2}, stability thresholds over {interval_2to3}, implicit consistency graphs across phases) that visible im1/im3 cannot resolve, ensuring correct evaluation only via these hidden details.
- To build carefully constrained consistency chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal stability index from mask1 to mask2 over {interval_1to2}, projected consistencies for {interval_2to3} to mask3), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the total time interval (e.g., "From the initial image to the final image over {interval_total}, the consistency was X maintaining Y"), ensuring solvers reason with temporal context, but do not mention hidden im2 or sub-intervals in the question text itself—reserve for internal logic and options only.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent consistency**: Patterns where visible im1/im3 suggest multiple similar stabilities over {interval_total}, but hidden im2/mask2 pivots the evaluation.
  2. **Layer multi-step consistency chains**: 3-5 steps with sub-layers and temporal twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial stabilities from mask1-2 over {interval_1to2}; Step 5: First twist via rate stabilities; Step 6: Second via thresholds; Step 7: Third via graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive full consistency over {interval_total}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: evaluate the **consistency** of a given object or scene from the initial image to the final image over the total {interval_total}.  
- Solver sees the initial image (im1) and the final image (im3).  
- Hidden im2 (intermediate) is used internally to verify consistency.  

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Analyze visible patterns from the initial image to the final image over {interval_total}.  
  2. Infer hidden intermediate consistency using masks.  
  3. Evaluate the full consistency level, ensuring alignment with total interval.  
  4. Ensure distractor evaluations are plausible but contradicted by hidden context.  

- Example twists:  
  - From initial to final over {interval_total}, buildings appear denser, but hidden intermediate shows consistent construction then stabilization.  
  - From initial to final over {interval_total}, water body appears stable, but hidden intermediate reveals fluctuation followed by recovery.  
"""
    examples = f"""
### Example Questions  

<Q> The initial image has farmland and open fields in the central zone, and the final image over {interval_total} shows dense residential with buildings and roads. How consistent was this transformation? </Q>  

<Q> The initial image features a dense forest in the upper-left, and the final image over {interval_total} has it thinned with bare patches and development nearby. How consistent was this shift? </Q>  
"""
    return f"""
You are a professional expert in multi-temporal consistency reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im2 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Spatio-Temporal Causal Chain Reasoning ===
def build_spatiotemporal_causal_chain_prompt(
    datasets="",
    name="Spatio-Temporal Causal Chain Reasoning",
    interval_1to2="one year",
    interval_2to3="one year",
    interval_total="two years"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all causal chain reasoning must focus exclusively on **building-related causes**  
    (e.g., initial sparse structures leading to densification over {interval_total}).  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Causal chain reasoning must therefore consider **multi-class causes** over {interval_total} from im1 to im3.  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about explaining **why the initial image evolved into the final image** over {interval_total}, identifying the real causal factors using hidden evidence in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im2 (after {interval_1to2} from im1 and before {interval_2to3} to im3) and multi-mask (mask1, mask2, mask3), leverage temporal mask causals for precise cause identification (e.g., causal rates over {interval_1to2}, factor thresholds over {interval_2to3}, implicit causal graphs across phases) that visible im1/im3 cannot resolve, ensuring correct explanation only via these hidden details.
- To build carefully constrained causal chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal causal index from mask1 to mask2 over {interval_1to2}, projected factors for {interval_2to3} to mask3), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the total time interval (e.g., "From the initial image to the final image over {interval_total}, the cause was X leading to Y"), ensuring solvers reason with temporal context, but do not mention hidden im2 or sub-intervals in the question text itself—reserve for internal logic and options only.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent cause**: Explanations where visible im1/im3 suggest multiple similar causes over {interval_total}, but hidden im2/mask2 pivots the true cause.
  2. **Layer multi-step causal chains**: 3-5 steps with sub-layers and causal twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial factors from mask1-2 over {interval_1to2}; Step 5: First twist via rate factors; Step 6: Second via thresholds; Step 7: Third via graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive full cause over {interval_total}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: explain **why the initial image evolved into the final image** over {interval_total} (real causal factors).  
- Solver sees the initial image (im1) and the final image (im3).  
- Hidden im2 (intermediate) is used internally to ground the real causal chain.  

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Analyze visible evolution from the initial image to the final image over {interval_total}.  
  2. Infer hidden intermediate causes using masks.  
  3. Explain the full causal chain, ensuring consistency with total interval.  
  4. Ensure distractor explanations are plausible but contradicted by hidden context.  

- Example twists:  
  - From initial to final over {interval_total}, farmland became urbanized due to infrastructure development, but hidden intermediate reveals specific triggers like road construction.  
  - From initial to final over {interval_total}, forest thinned due to logging, but hidden intermediate shows environmental factors amplifying the cause.  
"""
    examples = f"""
### Example Questions  

<Q> The initial image shows farmland in the central zone, and over {interval_total} the final image has it as dense residential. What caused this change? </Q>  

<Q> The initial image has dense forest in the upper-left, and over {interval_total} the final image shows it thinned to bare land. What caused this transformation? </Q>  
"""
    return f"""
You are a professional expert in spatio-temporal causal chain reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im2 and masks' pivotal roles.  

    """.strip()


# === Prompt Construction for Spatio-Temporal Counterfactual Reasoning ===
def build_spatiotemporal_counterfactual_prompt(
    datasets="",
    name="Spatio-Temporal Counterfactual Reasoning",
    interval_1to2="one year",
    interval_2to3="one year",
    interval_total="two years"
):
    # 针对 spaceNet7 做特殊处理
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all counterfactual reasoning must focus exclusively on **building-related hypothetical outcomes**  
    (e.g., if construction had not occurred, the final image would be sparse).  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Counterfactual reasoning must therefore consider **multi-class hypothetical outcomes** (e.g., if deforestation had not happened, the final image would remain forested).  
"""

    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target claim that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer claims about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion in the same sentence.
- Avoid claims whose truth is determined solely by a binary endpoint difference.
- For long claims, keep every clause tied to the same local target and its nearby reference, not to a broad neighborhood story.
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about hypothesizing **what the final image would look like if a past change had not occurred** (historical assumption), using hidden evidence to simulate alternative outcomes in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden im2 (after {interval_1to2} from im1 and before {interval_2to3} to im3) and multi-mask, leverage temporal mask alternatives for precise hypothetical simulations (e.g., alternative rates over {interval_1to2}, counterfactual thresholds over {interval_2to3}, implicit simulation graphs across phases) that visible im1/im3 cannot resolve, ensuring correct hypothesis only via these hidden details.
- To build carefully constrained counterfactual chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal alternative index from mask1 to mask2 over {interval_1to2}, projected counterfactuals for {interval_2to3} to mask3), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the total time interval (e.g., "If a past change had not occurred from the initial image over {interval_total} to the final image, the outcome would be X"), ensuring solvers reason with temporal context, but do not mention hidden im2 or sub-intervals in the question text itself—reserve for internal logic and options only.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent hypothesis**: Alternatives where visible im1/im3 suggest multiple similar hypotheticals over {interval_total}, but hidden im2/mask2 pivots the simulation.
  2. **Layer multi-step counterfactual chains**: 3-5 steps with sub-layers and hypothetical twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial alternatives from mask1-2 over {interval_1to2}; Step 5: First twist via rate alternatives; Step 6: Second via thresholds; Step 7: Third via graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive counterfactual outcome over {interval_total}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: assume a past change did not occur → predict what the final image would look like (historical hypothesis).  
- Solver sees the initial image (im1) and the final image (im3).  
- Hidden im2 (intermediate) is used internally to simulate alternatives.  

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Analyze visible outcome from the initial image to the final image over {interval_total}.  
  2. Hypothesize alternative paths using hidden masks.  
  3. Predict the counterfactual final state.  
  4. Ensure distractor hypotheticals are plausible but contradicted by hidden context.  

- Example twists:  
  - If urbanization had not occurred from initial to final over {interval_total}, farmland would remain intact, but hidden intermediate shows alternative pressures.  
  - If thinning had not happened from initial to final over {interval_total}, forest would be dense, but hidden intermediate reveals other factors.  
"""
    examples = f"""
### Example Questions  

<Q> The initial image shows farmland in the central zone, and over {interval_total} the final image has dense residential. What would the final image look like if a past change had not occurred? </Q>  

<Q> The initial image has dense forest in the upper-left, and over {interval_total} the final image shows thinned bare land. What would the final image look like if a past change had not occurred? </Q>  
"""
    return f"""
You are a professional expert in spatio-temporal counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden im2 and masks' pivotal roles.  

    """.strip()


JOBS = ["reason", "counterfactual", "influ", "sem", "plan", "estimate", "st_reason", "st_counterfactual",
        "st_evolution", "st_consistency", "state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]


def get_prompt_for_three(job, mask_palette, prior, datasets, interval_1to2="one year", interval_2to3="one year",
                         interval_total="two years"):
    """
    Build prompts for tri-phase jobs.  Time wording constraints (inserted into `prior`) are strict and written
    in fluent English so the LLM will consistently describe temporal references relative to the middle image.
    - interval_1to2 and interval_2to3 are human-readable offsets (e.g., "one month", "nine months", "one year").
    - Use *now* to refer to the middle/current image (the solver's primary viewpoint).
    - Describe the first image as "the scene {interval_1to2} before now" (i.e., now − interval_1to2).
    - Describe the third image as "the scene {interval_2to3} after now" (i.e., now + interval_2to3).
    - Do NOT use vague temporal words such as "earlier", "later", or ordinal labels like "first/second/third" in question text.
    - When a particular image is not visible to the solver, phrase the requirement as a prediction: "predict the scene {interval_2to3} after now" or "infer the scene {interval_1to2} before now".
    - For tasks that require only the earliest and latest images (functional evolution / consistency), require a single explicit interval `interval_total` that represents the elapsed time between the earlier image and the later image; the question text must state that elapsed interval in natural language (e.g., "the interval between the earlier and later images is two years") and must avoid any reference to the middle/current image.
    """

    # shared phrasing for jobs that present im1, im2, im3 and treat im2 as "now"
    im123_phrase = (
        f"\n  **Temporal phrasing rules (use exactly these styles):**\n"
        f"  - The middle image is the current viewpoint and should be described as \"now\" (the solver's primary view).\n"
        f"  - Describe the earlier image as: \"the scene {interval_1to2} before now\" (i.e., now − {interval_1to2}).\n"
        f"  - Describe the later image as: \"the scene {interval_2to3} after now\" (i.e., now + {interval_2to3}).\n"
        f"  - If the later image is not visible, phrase tasks as predictions: e.g., \"predict the scene {interval_2to3} after now.\" \n"
        f"  - If the earlier image is not visible, phrase tasks as reconstructions: e.g., \"infer the scene {interval_1to2} before now.\" \n"
        f"  - Do NOT use vague time words such as 'earlier'/'later' or ordinal words 'first/second/third' in the question text.\n"
        f"  - Use numeric, human-readable offsets (examples: 'nine months', 'four months', 'one year') when stating times.\n"
    )

    # phrasing for jobs that must NOT reference the middle image and instead use interval_total between earliest and latest images
    im1_im3_only_phrase = (
        f"\n  **Temporal constraint for earliest-vs-latest tasks:**\n"
        f"  - The question must explicitly state the elapsed interval `interval_total` describing the time between the earlier image and the later image.\n"
        f"    For example: \"The interval between the earlier image and the later image is {interval_total}.\"\n"
        f"  - In such tasks, do NOT mention or describe the middle/current image at all. Frame all temporal language solely around the earlier/later pair\n"
        f"    (e.g., \"the earlier scene {interval_total} before the later scene\").\n"
    )

    if job == "st_reason":
        prior += im1_im3_only_phrase
        return build_spatiotemporal_causal_chain_prompt(datasets=datasets,
                                                                  interval_1to2=interval_1to2,
                                                                  interval_2to3=interval_2to3, interval_total=interval_total)
    elif job == "st_counterfactual":
        prior += im1_im3_only_phrase
        return build_spatiotemporal_counterfactual_prompt(datasets=datasets,
                                                          interval_1to2=interval_1to2, interval_2to3=interval_2to3, interval_total=interval_total)
    elif job == "st_evolution":
        # this task uses only the earlier (im1) and later (im3) images in the question stem
        prior += im1_im3_only_phrase
        return build_multitemporal_evolution_prompt(datasets=datasets, interval_1to2=interval_1to2, interval_2to3=interval_2to3,
                                                                      interval_total=interval_total)
    elif job == "st_consistency":
        prior += im1_im3_only_phrase
        return build_multitemporal_consistency_prompt(datasets=datasets, interval_1to2=interval_1to2, interval_2to3=interval_2to3,
                                                             interval_total=interval_total)
    elif job == "state_predict":
        prior += im123_phrase
        return build_multitemporal_class_state_prediction_prompt(datasets=datasets,
                                                                 interval_1to2=interval_1to2,
                                                                 interval_2to3=interval_2to3)
    elif job == "shape_predict":
        prior += im123_phrase
        return build_multitemporal_shape_prediction_prompt(datasets=datasets,
                                                           interval_1to2=interval_1to2, interval_2to3=interval_2to3)
    elif job == "scene_uncertainty":
        prior += im123_phrase
        return build_multitemporal_uncertainty_prediction_prompt(datasets=datasets,
                                                                 interval_1to2=interval_1to2,
                                                                 interval_2to3=interval_2to3)
    elif job == "sequence_predict":
        prior += im123_phrase
        return build_multitemporal_sequence_prediction_prompt(datasets=datasets,
                                                              interval_1to2=interval_1to2, interval_2to3=interval_2to3)
    else:
        # undefined job: preserve original fallback behavior
        return f"[DEFAULT PROMPT for {job} on {datasets}]"

