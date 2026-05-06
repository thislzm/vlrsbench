# 预测能力
################################################# 类别状态预测推理（对象级别）（对象级别，多时相扩展）
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
  - Every question and option set must combine the temporal change with at least two extra RGB-verifiable constraints chosen from: attachment vs isolated infill, gap closure between persistent roofs, extension direction relative to a road/canal/lot edge, continuity or interruption along a row/corridor, boundary crossing, count threshold, or asymmetry between two nearby building groups.
  - Prefer two candidate building subregions that look similar at first glance, so the solver must compare local structure across frames rather than answer from a single obvious change cue.
  - If the correct option can be chosen from one coarse endpoint glance alone, reject the item and rewrite it with a stable landmark and a second local constraint.
"""

# 注意：根据用户反馈，修复examples中的病句问题——避免{ past_intervals }导致不自然表述（如"From four months, six months ago to now"），改为根据len(intervals)动态生成自然句子。
# - 如果len(intervals)==2（三时相），使用"From {intervals[0]} ago to now"（自然）。
# - 如果更多，使用"Over the past periods of {', '.join(intervals[:-1]) }"（e.g., "Over the past periods of four months, six months, eight months"），模型可生成合理question。
# - 同时，使用interval_total（总从im1 to im{n_phases}）辅助，如果需要总过去时间，可描述"over the past portion of {interval_total} to now"，但当前仅用于counterfactual；为预测任务添加可选past_description = f"over the past portion of {interval_total} to now" if wanted, but keep simple。
# - 仅调整examples生成，其他逻辑不变。
# - 类似调整所有预测任务函数。

# === Prompt Construction for Multi-Temporal Class State Prediction Reasoning (Multi-Phase) ===
def build_multitemporal_class_state_prediction_prompt(
    datasets="",
    name="Multi-Temporal Class State Prediction Reasoning",
    intervals=["one year", "one year"],
    interval_total="two years"
):
    n_phases = len(intervals) + 1  # 动态计算时相数
    # 校对：保留原change_scope逻辑，使用intervals[-1]作为最后一个间隔。
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to the **building category only**.  
  - Therefore, all predictions must focus exclusively on **building-related state changes**  
    (e.g., new construction, demolition, persistence, densification) over the time interval of {intervals[-1]} after now.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple land-cover categories (farmland, forest, water, residential, industrial, etc.).  
  - Predictions must therefore consider **multi-class state changes** for objects over the time interval of {intervals[-1]} after now.  
"""

    # 校对：添加明确指导段落，然后是原内容。
    last_interval = intervals[-1]
    prior_intervals_str = " ; ".join([f"im{i} to im{i+1} over {intervals[i-1]}" for i in range(1, n_phases)])
    past_intervals = ", ".join(intervals[:-1])  # 用于描述过去间隔链
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target question that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer contrasts about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion across all options.
- Avoid questions or options whose truth can be resolved solely from a binary endpoint difference.
- For long questions or options, keep every clause tied to the same local target and nearby reference, not to a broad neighborhood story.
- **Important Temporal Assumption**: You must assume the last visible image (im{n_phases-1}) is the current time point ('now'), and the task is to predict the state in the hidden future image (img{n_phases}) (im{n_phases}) {last_interval} after now. Do not use image labels like 'im{n_phases-1}' or 'im{n_phases}' in the generated question text; use natural phrasing such as "predict ... {last_interval} after now".
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}

- Adhere strictly to '{name}' — the task is about predicting the **future category state** of a given object {last_interval} after now, using multi-temporal hidden evidence to uncover trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden masks (mask1 to mask{n_phases}) across all phases, leverage temporal mask differences for precise trend extrapolations (e.g., category transition rates over {past_intervals}, density change thresholds, implicit evolution graphs across phases) that visible images from {past_intervals} ago to now cannot resolve alone, using these hidden mask details only to identify hard ambiguous regions, construct close alternatives, and verify that the final item remains solvable from visible RGB evidence.
- To build carefully constrained prediction chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal fragmentation index from mask1 to mask2 over {intervals[0]}, projected density shifts for {last_interval}), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{prior_intervals_str}, the state was X; predict Y {last_interval} after now"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trend**: Trajectory where visible images to now suggest multiple similar states, but hidden masks pivot the extrapolation {last_interval} after now.
  2. **Layer multi-step prediction chains**: 3-5 steps with sub-layers and temporal twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {intervals[0]}; Step 5: First twist via transition rates; Step 6: Second via density shifts; Step 7: Third via evolution graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future state for {last_interval}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the **future category state** of a given object {last_interval} after now (the current visible image).  
- Solver sees images from {past_intervals} ago to now (past to current visible images).  
- Hidden: the future image (img{n_phases}: {last_interval} after now) and all masks (mask1 to mask{n_phases}), used internally to ground transitions and validate predictions.  

- **Constraints for Question and Options Generation**:
  - Generate a long, information-dense question when helpful, but keep every clause anchored to the same localized target, evidence frame, and temporal span; difficulty should come from a layered local trap, not from broad scene storytelling.
  - Generated options should remain parallel, comparable, and tightly localized, focusing only on the predicted future state {last_interval} after now. If options are long, keep the sentence skeleton nearly identical and change only one evidence-bearing local variable.
  - Do NOT include reasoning steps, historical changes, detailed timelines, or process descriptions in options—reserve all logic and explanations for the solving rationale. Naturally integrate time references (e.g., "{last_interval} after now, the state will be...") without repeating past events.

- The reasoning chain must be **multi-step and context-sensitive**:  
  1. Detect how the object changed across visible images from {past_intervals} ago to now, using hidden masks for precise trends.  
  2. Examine surrounding regions (e.g., proximity to roads, rivers, urban edges).  
  3. Combine trajectory + context → predict category {last_interval} after now.  
  4. Ensure distractor categories are plausible but contradicted by spatial or semantic context revealed by masks.  

- Example twists:  
  - Farmland visible shrinking from {past_intervals} ago to now, surrounded by residential expansion, is unlikely to remain farmland → predicted state becomes residential or industrial {last_interval} after now.  
  - Forest visible thinning from {past_intervals} ago to now, adjacent to new infrastructure, is unlikely to stay forest → predicted state becomes bare land or farmland {last_interval} after now.  
  - Wetland visible shrinking from {past_intervals} ago to now, adjacent to urbanization, is unlikely to recover → predicted state becomes urban {last_interval} after now.  
"""

    # 校对：动态生成examples以避免病句。
    if len(intervals) == 2:
        past_description = f"{intervals[0]} ago to now"
    else:
        past_description = f"the past {interval_total} to now"

    examples = f"""
### Example Questions  

<Q> From {past_description}, the central zone changed from intact farmland to scattered houses.  
Now predict what the category will be {last_interval} after now. </Q>  

<Q> From {past_description}, the upper-left forest thinned.  
Now predict what the category will be {last_interval} after now. </Q>  
"""
    return f"""
You are a professional expert in multi-temporal class state prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden masks' pivotal roles in predicting the future state {last_interval} after now.  

    """.strip()


# === Prompt Construction for Multi-Temporal Object Shape Prediction Reasoning (Multi-Phase) ===
def build_multitemporal_shape_prediction_prompt(
    datasets="",
    name="Multi-Temporal Object Shape Prediction Reasoning",
    intervals=["one year", "one year"],
    interval_total="two years"
):
    n_phases = len(intervals) + 1
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

    last_interval = intervals[-1]
    prior_intervals_str = " ; ".join([f"im{i} to im{i+1} over {intervals[i-1]}" for i in range(1, n_phases)])
    past_intervals = ", ".join(intervals[:-1])
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target question that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer contrasts about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion across all options.
- Avoid questions or options whose truth can be resolved solely from a binary endpoint difference.
- For long questions or options, keep every clause tied to the same local target and nearby reference, not to a broad neighborhood story.
- **Important Temporal Assumption**: You must assume the last visible image (im{n_phases-1}) is the current time point ('now'), and the task is to predict the shape in the hidden future image (img{n_phases}) (im{n_phases}) {last_interval} after now. Do not use image labels like 'im{n_phases-1}' or 'im{n_phases}' in the generated question text; use natural phrasing such as "predict ... {last_interval} after now".
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about predicting the **future spatial shape** of a given object {last_interval} after now, using multi-temporal hidden evidence to project geometric trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden masks (mask1 to mask{n_phases}) across all phases, leverage temporal mask geometries for precise shape extrapolations (e.g., boundary change rates over {past_intervals}, fragmentation thresholds, implicit morphology graphs across phases) that visible images from {past_intervals} ago to now cannot resolve alone.
- To build carefully constrained shape prediction chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal shape index from mask1 to mask2 over {intervals[0]}, projected boundary shifts for {last_interval}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{prior_intervals_str}, the shape was X; predict Y {last_interval} after now"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trajectory**: Geometry where visible images to now suggest multiple similar shapes, but hidden masks pivot the projection {last_interval} after now.
  2. **Layer multi-step prediction chains**: 3-5 steps with sub-layers and geometric twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial shapes from mask1-2 over {intervals[0]}; Step 5: First twist via boundary rates; Step 6: Second via fragmentation; Step 7: Third via morphology; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future shape for {last_interval}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the **future spatial shape** of a given object {last_interval} after now (the current visible image).  
- Solver sees images from {past_intervals} ago to now (past to current visible images).  
- Hidden: the future image (img{n_phases}: {last_interval} after now) and all masks (mask1 to mask{n_phases}), used internally to ground the trajectory and validate predictions.  

- **Constraints for Question and Options Generation**:
  - Generate a long, information-dense question when helpful, but keep every clause anchored to the same localized shape target, evidence frame, and temporal span; difficulty should come from a layered local trap, not from broad scene storytelling.
  - Generated options should remain parallel, comparable, and tightly localized, focusing solely on the predicted future shape {last_interval} after now. If options are long, keep the sentence skeleton nearly identical and change only one evidence-bearing geometric variable.
  - Do NOT include reasoning steps, historical changes, detailed timelines, or process descriptions in options—reserve all logic and explanations for the solving rationale. Naturally integrate time references (e.g., "{last_interval} after now, the shape will evolve to...") without repeating past events.

- Reasoning chain:  
  1. Track object geometry across visible images from {past_intervals} ago to now (e.g., forest patch splitting, river widening), using hidden masks for precision.  
  2. Assess external pressures (urban sprawl, agricultural expansion, hydrological changes).  
  3. Predict how the object’s spatial shape evolves {last_interval} after now (expansion, shrinkage, fragmentation, merging).  
  4. Distractors: plausible but geometrically or contextually inconsistent per masks.  

- Example twists:  
  - A forest patch visible fragmenting from {past_intervals} ago to now, near urban expansion → predicted shape likely further fragmented into isolated patches {last_interval} after now.  
  - A water body visible shrinking from {past_intervals} ago to now due to farmland drainage → predicted shape likely further reduced, not suddenly expanding {last_interval} after now.  
  - A residential cluster visible growing denser from {past_intervals} ago to now, near a highway → predicted shape likely expands further, not shrinks {last_interval} after now.  
"""

    # 校对：动态生成examples以避免病句。
    if len(intervals) == 2:
        past_description = f"{intervals[0]} ago to now"
    else:
        past_description = f"the past {interval_total} to now"

    examples = f"""
### Example Questions  

<Q> From {past_description}, a continuous forest patch became fragmented.  
Now predict what the spatial shape will be {last_interval} after now. </Q>  

<Q> From {past_description}, the river narrowed and farmland expanded.  
Now predict how the river and its banks will evolve {last_interval} after now. </Q>  
"""
    return f"""
You are a professional expert in multi-temporal object shape prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden masks' pivotal roles in predicting the future shape {last_interval} after now.  

    """.strip()


# === Prompt Construction for Scenario Uncertainty Prediction Reasoning (Multi-Phase) ===
def build_multitemporal_uncertainty_prediction_prompt(
    datasets="",
    name="Scenario Uncertainty Prediction Reasoning",
    intervals=["one year", "one year"],
    interval_total="two years"
):
    n_phases = len(intervals) + 1
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all uncertainty predictions must focus exclusively on **building-related scenario outcomes**  
    (e.g., new construction patterns, demolition vs persistence, densification trends) over the future {intervals[-1]}.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Uncertainty predictions must therefore consider **multi-class scenario outcomes** (e.g., farmland urbanization vs ecological protection, forest regrowth vs clearance, water expansion vs stability) over the future {intervals[-1]}.  
"""

    last_interval = intervals[-1]
    prior_intervals_str = " ; ".join([f"im{i} to im{i+1} over {intervals[i-1]}" for i in range(1, n_phases)])
    past_intervals = ", ".join(intervals[:-1])
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target question that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer contrasts about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion across all options.
- Avoid questions or options whose truth can be resolved solely from a binary endpoint difference.
- For long questions or options, keep every clause tied to the same local target and nearby reference, not to a broad neighborhood story.
- **Important Temporal Assumption**: You must assume the last visible image (im{n_phases-1}) is the current time point ('now'), and the task is to predict the scenario in the hidden future image (img{n_phases}) (im{n_phases}) {last_interval} after now. Do not use image labels like 'im{n_phases-1}' or 'im{n_phases}' in the generated question text; use natural phrasing such as "predict ... {last_interval} after now".
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about selecting the **future scenario best supported by the visible trajectory** among multiple fixed alternatives {last_interval} after now, using multi-temporal hidden evidence to resolve uncertainties in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden masks (mask1 to mask{n_phases}) and hidden future image (img{n_phases}: {last_interval} after now), leverage temporal mask uncertainties for precise scenario branching (e.g., probabilistic transition matrices over {past_intervals}, density uncertainty thresholds, implicit trend graphs across phases) that visible images from {past_intervals} ago to now cannot resolve.
- To build carefully constrained uncertainty chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal uncertainty index from mask1 to mask2 over {intervals[0]}, projected branching probabilities for {last_interval}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{prior_intervals_str}, the trend was X; predict Y scenario {last_interval} after now"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trajectory**: Trends where visible images to now suggest multiple similar scenarios, but hidden masks and future image (img{n_phases}) pivot the probabilities.
  2. **Layer multi-step uncertainty chains**: 3-5 steps with sub-layers and probabilistic twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {intervals[0]}; Step 5: First twist via transition probabilities; Step 6: Second via density uncertainties; Step 7: Third via trend graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive scenario for {last_interval}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: predict the most plausible **future scenario** among multiple alternatives {last_interval} after now in the hidden future image (img{n_phases}).  
- Solver sees images from {past_intervals} ago to now (past to current visible images).  
- Hidden: the future image (img{n_phases}) ("{last_interval} after now") and all masks (mask1 to mask{n_phases}), used for validation.  

- **Constraints for Question and Options Generation**:
  - Generate a long, information-dense question when helpful, but keep every clause anchored to the same localized target, evidence frame, and temporal span; difficulty should come from a layered local trap, not from broad scene storytelling.
  - Generated options should remain parallel, comparable, and tightly localized, focusing solely on the predicted future scenario {last_interval} after now. If options are long, keep the sentence skeleton nearly identical and change only one evidence-bearing local variable.
  - Do NOT include reasoning steps, historical changes, detailed timelines, or process descriptions in options—reserve all logic and explanations for the solving rationale. Naturally integrate time references (e.g., "{last_interval} after now, the most plausible scenario is...") without repeating past events.


- How to use masks:  
  - Use hidden mask1 to mask{n_phases-1} to identify multiple evolving trends (e.g., farmland shrinking, forest thinning, water expansion) from visible images.  
  - Use hidden mask{n_phases} and future image (img{n_phases}) to validate which scenario materializes.  
  - Distractors must be scenarios that seem plausible but contradict hidden evidence.  

- Reasoning chain:  
  1. Analyze visible images from {past_intervals} ago to now to detect diverging trends, refined by hidden masks.  
  2. Generate multiple plausible scenarios for {last_interval} after now.  
  3. Select the one consistent with hidden semantics + evolution logic.  
"""

    # 校对：动态生成examples以避免病句。
    if len(intervals) == 2:
        past_description = f"{intervals[0]} ago to now"
    else:
        past_description = f"the past {interval_total} to now"

    examples = f"""
### Example Questions  

<Q> From {past_description}, the central farmland was fragmented by scattered houses, northern forest thinned, southern river widened.  
Now predict which scenario is best supported {last_interval} after now: full urban expansion, ecological restoration, or mixed agro-urban land use? </Q>  

<Q> From {past_description}, the eastern industrial zone expanded toward central farmland with highway construction.  
Now predict which scenario is best supported {last_interval} after now: stabilized industry, farmland persistence, or full industrial takeover? </Q>  
"""
    return f"""
You are a professional expert in scenario uncertainty prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden masks' pivotal roles in predicting the future scenario {last_interval} after now.  

    """.strip()


# === Prompt Construction for Spatio-Temporal Sequence Prediction Reasoning (Multi-Phase) ===
def build_multitemporal_sequence_prediction_prompt(
    datasets="",
    name="Spatio-Temporal Sequence Prediction Reasoning",
    intervals=["one year", "one year"],
    interval_total="two years"
):
    n_phases = len(intervals) + 1
    if datasets == "spaceNet7":
        change_scope = f"""
- For the dataset 'spaceNet7':  
  - Semantic changes are restricted to **building category only**.  
  - Therefore, all sequence predictions must focus exclusively on **building-related temporal trends**  
    (e.g., continuous expansion of building clusters, ongoing densification, demolition-reconstruction cycles) projected {intervals[-1]} after now.  
"""
    else:
        change_scope = f"""
- For this dataset:  
  - Semantic changes may involve multiple categories (farmland, forest, water, residential, industrial, etc.).  
  - Sequence predictions must therefore consider **multi-class temporal trends** (e.g., farmland → scattered houses → dense residential, forest → bare land → agriculture, water body shrinkage → further shrinkage) projected {intervals[-1]} after now.  
"""

    last_interval = intervals[-1]
    prior_intervals_str = " ; ".join([f"im{i} to im{i+1} over {intervals[i-1]}" for i in range(1, n_phases)])
    past_intervals = ", ".join(intervals[:-2])
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Use a single target question that requires temporal evidence plus local structural verification; do not let one obvious change verb determine the answer.
- Prefer contrasts about comparative infill, attachment, boundary crossing, gap closure, continuity, or asymmetry rather than plain "new building / no new building" detection.
- In temporal building scenes, require one evolving target and one stable landmark or comparison subregion across all options.
- Avoid questions or options whose truth can be resolved solely from a binary endpoint difference.
- For long questions or options, keep every clause tied to the same local target and nearby reference, not to a broad neighborhood story.
- **Important Temporal Assumption**: You must assume the last visible image (im{n_phases-1}) is the current time point ('now'), and the task is to predict the state in the hidden future image (img{n_phases}) (im{n_phases}) {last_interval} after now. Do not use image labels like 'im{n_phases-1}' or 'im{n_phases}' in the generated question text; use natural phrasing such as "predict ... {last_interval} after now".
{SPACENET7_REASONING_HARDENING if datasets == "spaceNet7" else ""}
- Adhere strictly to '{name}' — the task is about extrapolating deterministic **temporal sequences** to predict the state {last_interval} after now, using multi-temporal hidden evidence to project sustained trends in remote sensing contexts.
- To elevate difficulty via maximal utilization of hidden masks (mask1 to mask{n_phases}) and hidden future image (img{n_phases}: {last_interval} after now), leverage temporal mask sequences for precise trend continuations (e.g., consistent change vectors over {past_intervals}, density evolution thresholds, implicit sequence graphs across phases) that visible images from {past_intervals} ago to now cannot resolve.
- To build carefully constrained sequence chains, build a concise 3-5 step evidence chain, where each step builds on multi-mask derived metrics (e.g., temporal trend index from mask1 to mask2 over {intervals[0]}, projected vector extensions for {last_interval}), creating interdependencies that collapse distractors.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- **Force time interval integration in options**: Every generated option must explicitly reference the time intervals (e.g., "{prior_intervals_str}, the sequence was X; predict Y {last_interval} after now"), ensuring solvers reason with temporal context.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify ultra-ambiguous, multi-mask dependent trend**: Sequence where visible images to now suggest multiple similar continuations, but hidden masks and future image (img{n_phases}) pivot the projection.
  2. **Layer multi-step sequence chains**: 3-5 steps with sub-layers and trend twists (e.g., Step 1: Visible ambiguity; Step 2-4: Initial trends from mask1-2 over {intervals[0]}; Step 5: First twist via change vectors; Step 6: Second via density evolutions; Step 7: Third via sequence graphs; Step 8: Fourth paradoxical inversion; Step 9: Multi-mask exclusive future state for {last_interval}).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
{change_scope}
- Focus: extrapolate deterministic **temporal sequences** to predict the state {last_interval} after now in the hidden future image (img{n_phases}).  
- Solver sees images from {past_intervals} ago to now (past to current visible images).  
- Hidden: the future image (img{n_phases}) ("{last_interval} after now") and all masks (mask1 to mask{n_phases}), used for validation.  

- **Constraints for Question and Options Generation**:
  - Generate a long, information-dense question when helpful, but keep every clause anchored to the same localized target, evidence frame, and temporal span; difficulty should come from a layered local trap, not from broad scene storytelling.
  - Generated options should remain parallel, comparable, and tightly localized, focusing solely on the predicted future state in the sequence {last_interval} after now. If options are long, keep the sentence skeleton nearly identical and change only one evidence-bearing local variable.
  - Do NOT include reasoning steps, historical changes, detailed timelines, or process descriptions in options—reserve all logic and explanations for the solving rationale. Naturally integrate time references (e.g., "{last_interval} after now, the deterministic state will be...") without repeating past events.

- How to use masks:  
  - Use hidden mask1 to mask{n_phases-1} to detect sustained trends (steady shrinkage, continuous expansion) from visible images.  
  - Use hidden mask{n_phases} and future image (img{n_phases}) to ensure deterministic continuation.  
  - Distractors = trend reversals without cause.  

- Reasoning chain:  
  1. Detect consistent trends across visible images from {past_intervals} ago to now, refined by hidden masks.  
  2. Extrapolate deterministically {last_interval} after now.  
  3. Eliminate trend reversals as implausible.  
"""

    # 校对：动态生成examples以避免病句。
    if len(intervals) == 2:
        past_description = f"{intervals[0]} ago to now"
    else:
        past_description = f"the past {interval_total} to now"

    examples = f"""
### Example Questions  

<Q> From {past_description}, the central farmland shrank and was replaced by new residential blocks.  
Now predict what deterministic state it will reach {last_interval} after now: restored farmland, stabilized mixed use, or fully dense residential? </Q>  

<Q> From {past_description}, the upper-left forest thinned continuously.  
Now predict what deterministic state it will reach {last_interval} after now: dense forest, bare land, or cultivated farmland? </Q>  
"""
    return f"""
You are a professional expert in spatio-temporal sequence prediction reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-multi-mask models, resolved by hidden temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, multi-mask critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting hidden masks' pivotal roles in predicting the future state {last_interval} after now.  

    """.strip()


JOBS = ["reason", "counterfactual", "influ", "sem", "plan", "estimate", "st_reason", "st_counterfactual",
        "st_evolution", "st_consistency", "state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]


def get_prompt_for_multi(job, mask_palette, prior, datasets, intervals=["one year", "one year"], interval_total="two years"):
    """
    Build prompts for multi-phase jobs.  Time wording constraints (inserted into `prior`) are strict and written
    in fluent English so the LLM will consistently describe temporal references relative to the middle image.
    - intervals is a list of human-readable offsets (e.g., ["one month", "nine months", "one year"] for four phases).
    - n_phases = len(intervals) + 1
    - Use *now* to refer to the penultimate image (im{n_phases-1}) for prediction tasks.
    - Describe prior images dynamically based on cumulative intervals.
    - For tasks that require only the earliest and latest images (functional evolution / consistency), require a single explicit interval `interval_total` that represents the elapsed time between the earlier image and the later image; the question text must state that elapsed interval in natural language (e.g., "the interval between the earlier and later images is two years") and must avoid any reference to intermediates.
    """

    # shared phrasing for jobs that present multiple images and treat im{n_phases-1} as "now"
    im_multi_phrase = (
        f"\n  **Temporal phrasing rules (use exactly these styles):**\n"
        f"  - The penultimate image is the current viewpoint and should be described as \"now\" (the solver's primary view).\n"
        f"  - Describe the final image as: \"the scene {intervals[-1]} after now\" (i.e., now + {intervals[-1]}).\n"
        f"  - If the final image is not visible, phrase tasks as predictions: e.g., \"predict the scene {intervals[-1]} after now.\" \n"
        f"  - Do NOT use vague time words such as 'earlier'/'later' or ordinal words 'first/second/third' in the question text.\n"
        f"  - Use numeric, human-readable offsets from the intervals list when stating times.\n"
        f"  - When describing past changes in questions, use natural language to avoid awkward phrasing, e.g., 'From the past {interval_total} to now'.\n"
    )

    if job == "state_predict":
        prior += im_multi_phrase
        return build_multitemporal_class_state_prediction_prompt(datasets=datasets, intervals=intervals, interval_total=interval_total)
    elif job == "shape_predict":
        prior += im_multi_phrase
        return build_multitemporal_shape_prediction_prompt(datasets=datasets, intervals=intervals, interval_total=interval_total)
    elif job == "scene_uncertainty":
        prior += im_multi_phrase
        return build_multitemporal_uncertainty_prediction_prompt(datasets=datasets, intervals=intervals, interval_total=interval_total)
    elif job == "sequence_predict":
        prior += im_multi_phrase
        return build_multitemporal_sequence_prediction_prompt(datasets=datasets, intervals=intervals, interval_total=interval_total)
    else:
        return f"[DEFAULT PROMPT for {job} on {datasets}]"
