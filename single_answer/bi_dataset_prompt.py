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


#################################### 双时相因果推理 Prompt
# === Prompt Construction for Two-Timepoint Causal Reasoning ===
def build_twotime_causal_prompt(datasets='', name="Two-Timepoint Causal Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **inferring the hidden earlier cause that led to the observed current state (visible image)**, with its unique characteristic being retrospective tracing across timepoints using temporal semantic shifts specific to remote sensing change detection.
- To elevate difficulty via maximal mask utilization, leverage both masks for precise temporal differentiations (e.g., category transition matrices, change density thresholds, implicit evolution graphs from mask1 to mask2) that optics cannot resolve, ensuring correct cause only via these hidden temporal details.
- To build carefully constrained cause chains, build a concise 3-5 step evidence chain, where each step builds on mask-derived temporal metrics (e.g., change fragmentation from mask1-mask2 differences, edge evolution implying causal interactions), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process grounded in retrospective tracing for brain-teaser elevation:
  1. **Identify ultra-ambiguous, mask-dependent phenomenon in im2**: Event where optics suggest multiple similar earlier causes, but masks' transitions pivot the chain.
  2. **Layer multi-step retrospective chains**: 3-5 steps with temporal sub-layers (e.g., Step 1: im2 ambiguity; Step 2-4: Initial temporal factors with change densities; Step 5: First twist via transition adjacency; Step 6: Second via evolution fragmentation; Step 7: Third via connectivity shifts; Step 8: Fourth paradoxical inversion; Step 9: Masks-exclusive earlier cause with layered revelation).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1 (earlier state): hidden label information, used internally by the model but not visible to humans.  
  - im2 (later state): visible to the human solver (optical image).  
- The masks reveal semantic categories and transitions, but humans only see the later image (im2).  
- Introduce **causal twists**:  
  - A dense residential block now may look like natural city growth, but hidden evidence shows it was farmland before → cause = farmland conversion.  
  - A bare land patch now may look like natural degradation, but hidden evidence shows it was forest before → cause = deforestation.  
  - An expanded water body now may look like rainfall impact, but hidden evidence shows construction blocked drainage → cause = anthropogenic intervention.  
- Questions must be phrased only with respect to the **current visible state** (im2).  
- Questions must avoid vague terms like 'some'; instead, reference **concrete spatial cues** (e.g., "the farmland in the lower-right", "the forest in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Two-Timepoint Causal Reasoning Examples with High-Similarity Distractors  
<Q> The central zone in the current image mimics multiple similar conversion causes from optics, but temporal metrics reveal the true earlier trigger; what is the precise root after layered twists? </Q>  
<Q> Upper-left bare land echoes several degradation hybrids visibly, yet mask transitions differentiate; what is the nested earlier cause? </Q>  
<Q> Lower-right expanded water body parallels rainfall and intervention causes optically, but density shifts pivot; what hyper-similar earlier causes resolve to? </Q>  
"""
    return f"""
You are a professional expert in causal reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-mask models, resolved only by masks' precise temporal details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Two-Timepoint Counterfactual Reasoning ===
# === Prompt Construction for Two-Timepoint Counterfactual Reasoning ===
def build_twotime_counterfactual_prompt(datasets='', name="Two-Timepoint Counterfactual Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **posing hypothetical assumptions about earlier states (im1) and reasoning whether the current state (im2) would remain the same or differ**, with its unique characteristic being hypothetical temporal projections using change evidence from remote sensing.
- To elevate difficulty via maximal mask utilization, use both masks for hypothetical temporal reconstructions (e.g., simulated transition matrices, projected density evolutions from mask1 to alternative mask2) that optics can't infer, making outcomes hinge on these invisible computations.
- To build carefully constrained counterfactual chains, use a concise 3-5 step evidence chain, interlinking via masks metrics (e.g., hypothetical change fragmentation), creating chains where small temporal differences amplify to distinct outcomes.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define ultra-conditional, masks-pivotal assumption**: Hypothesis where optics suggest similar alternatives, but masks' transitions enable precise projections.
  2. **Project multi-step, interdependent chains**: 3-5 steps with metrics (e.g., Step 1: Assumption; Step 2-4: Initial projections; Step 5: First twist via temporal extrapolation; Step 6: Second via simulation; Step 7: Third; Step 8: Fourth inversion; Step 9: Masks-exclusive outcome).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1 (earlier state): hidden label information, used internally by the model but not visible to humans.  
  - im2 (later state): visible to the human solver (optical image).  
- The masks reveal semantic categories and transitions, but humans only see the current image (im2).  
- Introduce **counterfactual twists**:  
  - If a forest patch in im1 turned into bare land in im2, ask: "If the forest had not been cleared, would it still be forest, or would urban expansion still replace it?"  
  - If farmland in im1 became residential in im2, ask: "If the farmland had been preserved, would it remain farmland, or still be converted due to road expansion?"  
  - If a river widened from im1 to im2, ask: "If upstream construction had not occurred, would the river still expand?"  
- ⚠️ In the question text:  
  - Never mention 'mask', 'segmentation', 'im1', or 'earlier image'.  
  - Phrase questions as a closed-set counterfactual check, e.g. **"If the earlier change in this target area had not occurred, which current-state description would best match the visible evidence now?"**  
- Options must be diverse: correct counterfactual category/state, plausible but wrong alternatives, and misleading distractors.  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Example Questions  

<Q> The upper-left area in the current image is bare land. If the earlier clearing had not occurred, which current-state description would best match the visible evidence for that area now? </Q>  

<Q> The central zone in the current image is residential. If the earlier farmland preservation assumption held, which current-state description would best match the visible evidence now? </Q>  

<Q> The river in the current image has widened. If the earlier upstream intervention had not occurred, which current-state description would best match the visible evidence now? </Q>  
"""
    return f"""
You are a professional expert in counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame scenarios that confuse non-mask models, pivoted by masks' projective temporal details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()



# === Prompt Construction for Two-Timepoint Spatio-Temporal Causal Chain Reasoning ===
def build_twotime_causal_chain_prompt(datasets='', name="Two-Timepoint Spatio-Temporal Causal Chain Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **inferring and explaining the causal chain of changes observed between two timepoints (im1 → im2)**, with its unique characteristic being temporal chain reconstruction using spatio-temporal interactions in remote sensing.
- To elevate difficulty via maximal mask utilization, exploit both masks for dynamic chain inferences (e.g., transition flow directions, feedback strengths from category evolutions between mask1 and mask2) invisible in optics.
- To build carefully constrained causal chains, use a concise 3-5 step evidence chain, chained via masks metrics (e.g., evolving interaction graphs across timepoints), where links are fragile without precise temporal details.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify misleading, masks-dependent change**: Shift where optics imply similar chains, but masks' evolutions define uniqueness.
  2. **Unfold multi-step, interdependent chains**: 3-5 steps with metrics (e.g., Step 1: Observed change; Step 2-4: Temporal interactions; Step 5: First loop; Step 6: Second twist; Step 7: Third; Step 8: Fourth amplification; Step 9: Masks-exclusive chain conclusion).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1 (earlier state): visible to the human solver (optical image).  
  - im2 (later state): visible to the human solver (optical image).  
- The masks reveal semantic categories and transitions, but humans only see im1 and im2.  
- Introduce **causal twists**:  
  - A forest disappearance may look natural, but masks show adjacency to new residential blocks → causal chain = urban expansion caused forest loss.  
  - A water body expansion may look like rainfall, but masks show farmland conversion → causal chain = irrigation flooding farmland.  
  - A farmland loss may look like degradation, but masks show road extension → causal chain = infrastructure expansion.  
- ⚠️ In the question text:  
  - Phrase questions as "Based on the changes visible between the earlier and later images..."  
- Questions must avoid vague terms like 'some'; instead, reference **specific spatial cues** (e.g., "the central farmland", "the forest in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Example Questions  

<Q> Between the earlier and later images, the central farmland turned into residential. What is the causal chain behind this change? </Q>  

<Q> Between the earlier and later images, the upper-left forest turned into bare land. What is the causal chain behind this change? </Q>  

<Q> Between the earlier and later images, the lower-right river widened. What is the causal chain behind this change? </Q>  
"""
    return f"""
You are a professional expert in two-timepoint spatio-temporal causal chain reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame chains that baffle non-mask models, distinguished by masks' temporal inferences.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Two-Timepoint Spatio-Temporal Counterfactual Reasoning ===
def build_twotime_Spatio_Temporal_counterfactual_prompt(datasets='', name="Two-Timepoint Spatio-Temporal Counterfactual Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **posing counterfactual assumptions based on changes observed between im1 and im2, and reasoning what the current state would be if those changes had not occurred**, with its unique characteristic being spatio-temporal hypothetical projections in remote sensing.
- To elevate difficulty via maximal mask utilization, use both masks for spatio-temporal hypotheticals (e.g., alternative evolution simulations, projected connectivity changes from mask1 to hypothetical mask2) that optics can't compute.
- To build carefully constrained counterfactual chains, use a concise 3-5 step evidence chain, interlinking via masks metrics (e.g., hypothetical temporal fragmentation), where small differences amplify.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define biased, masks-pivotal assumption**: Assumption where optics imply similar alternatives, but masks' evolutions enable precision.
  2. **Project multi-step chains**: 3-5 steps (e.g., Step 1: Assumption; Step 2-4: Projections; Step 5: First twist; Step 6: Second; Step 7: Third; Step 8: Fourth inversion; Step 9: Masks-exclusive state).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1 (earlier state): visible optical image for human solver, with mask only for model reference.  
  - im2 (later state): visible optical image for human solver, with mask only for model reference.  
- Humans see both im1 and im2, but never the masks.  
- Introduce **counterfactual twists**:  
  - If a forest patch in im1 turned into bare land in im2, ask: "If the forest had not been cleared, would it still be forest, or would urban expansion still replace it?"  
  - If farmland in im1 became residential in im2, ask: "If the farmland had been preserved, would it remain farmland, or still be converted due to road expansion?"  
  - If a river widened from im1 to im2, ask: "If upstream construction had not occurred, would the river still expand?"  
- ⚠️ In the question text:  
  - Phrase in terms of **visible changes between the two images**.  
- Questions must avoid vague terms like 'some'; instead, reference **specific spatial or positional cues** (e.g., "the farmland in the central zone", "the forest patch in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Example Questions  

<Q> Between the earlier and later images, the upper-left forest patch turned into bare land. If the forest had not been cleared, what would the current state be? </Q>  

<Q> Between the earlier and later images, the central farmland became residential. If the farmland had been preserved, what would the current state be? </Q>  

<Q> Between the earlier and later images, the river widened. If upstream construction had not occurred, what would the current state be? </Q>  
"""
    return f"""
You are a professional expert in two-timepoint spatio-temporal counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame alternatives that confuse non-mask models, pivoted by masks' projective spatio-temporal details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  
    """.strip()


# === Prompt Construction for Two-Timepoint Functional Zone Evolution Reasoning ===
def build_twotime_functional_evolution_prompt(datasets='', name="Two-Timepoint Functional Zone Evolution Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **integrating semantic evidence from im1 (earlier state) and im2 (later state) to infer the functional zone evolution of specific areas**, with its unique characteristic being temporal zone synthesis in remote sensing land use monitoring.
- To elevate difficulty via maximal mask utilization, use both masks for evolution quantifications (e.g., zone transition scores, density change maps from mask1 to mask2) that optics approximate but can't precisely fuse.
- To build carefully constrained evolution chains, use a concise 3-5 step evidence chain, linked via masks metrics (e.g., evolving heterogeneity indices), building to differentiated evolutions.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Catalog ambiguous, masks-pivotal changes**: Changes where optics suggest similar evolutions, but masks' transitions define uniqueness.
  2. **Aggregate into hyper-nested evolutions**: 3-5 steps (e.g., Step 1: Change pairing; Step 2-4: Temporal clustering; Step 5: First conflict; Step 6: Second via change heterogeneity; Step 7: Third; Step 8: Fourth inversion; Step 9: Masks-exclusive evolution type).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1 (earlier state): visible optical image for human solver, mask used internally by model.  
  - im2 (later state): visible optical image for human solver, mask used internally by model.  
- Humans see both im1 and im2, but never the masks.  
- Introduce **functional evolution twists**:  
  - A farmland in im1 becomes scattered houses in im2 → functional evolution = agricultural → residential transition.  
  - A forest patch in im1 becomes fragmented and partly farmland in im2 → functional evolution = deforestation and agricultural expansion.  
  - A low-density residential zone in im1 becomes high-density blocks in im2 → functional evolution = urban densification.  
- ⚠️ In the question text:  
  - Phrase in terms of **visible changes between the two images**.  
- Questions must avoid vague terms like 'some'; instead, reference **specific spatial or positional cues** (e.g., "the farmland in the central zone", "the forest patch in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Example Questions  

<Q> Between the earlier and later images, the central farmland turned into scattered houses. How did the functional zone evolve? </Q>  

<Q> Between the earlier and later images, the upper-left forest patch became fragmented and partly farmland. How did the functional zone evolve? </Q>  

<Q> Between the earlier and later images, the lower-right low-density residential became high-density blocks. How did the functional zone evolve? </Q>  
"""
    return f"""
You are a professional expert in two-timepoint functional zone evolution reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame evolutions that confound non-mask models, refined by masks' temporal fusion details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Two-Timepoint Temporal Evolution Consistency Reasoning ===
def build_twotime_consistency_prompt(datasets='', name="Two-Timepoint Temporal Evolution Consistency Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **judging whether the observed evolution from im1 (earlier state) to im2 (later state) is logically consistent with natural or social evolution patterns**, with its unique characteristic being consistency checks using temporal patterns in remote sensing monitoring.
- To elevate difficulty via maximal mask utilization, harness both masks for consistency quantifications (e.g., evolution plausibility scores from category transitions, invisible paradox detections via density comparisons) that optics can't verify.
- To build carefully constrained consistency chains, use a concise 3-5 step evidence chain, chained via masks metrics (e.g., temporal compatibility indices), where assessments hinge on subtle evolution differences.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Present misleading, masks-dependent evolution**: Evolution where optics imply similar consistencies, but masks quantify uniqueness.
  2. **Conduct multi-step consistency assessments**: 3-5 steps (e.g., Step 1: Observed evolution; Step 2-4: Pattern checks; Step 5: First nested paradox; Step 6: Second; Step 7: Third; Step 8: Fourth inversion; Step 9: Masks-exclusive judgment).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
- Each image (im1, im2) has its own mask.  
  - im1: visible optical image for human solver, mask used internally by model.  
  - im2: visible optical image for human solver, mask used internally by model.  
- Humans see both im1 and im2, but never the masks.  
- Introduce **consistency twists**:  
  - A forest changing into bare land may be consistent (deforestation), but if simultaneously nearby farmland turns back into dense forest, this is inconsistent.  
  - A residential area densifying into urban blocks is consistent, but if adjacent industrial areas disappear into open land, that is inconsistent.  
  - A river widening while nearby floodplains turn into farmland is inconsistent.  
- ⚠️ In the question text:  
  - Phrase in terms of **visible changes between the two images**.  
- Questions must avoid vague terms like 'some'; instead, reference **specific spatial or positional cues** (e.g., "the farmland in the central zone", "the forest patch in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Example Questions  

<Q> Between the earlier and later images, the central zone shows significant changes. Is this evolution process consistent? </Q>  

<Q> Between the earlier and later images, the upper-left area has notable transformations. Is this evolution process consistent? </Q>  

<Q> Between the earlier and later images, the lower-right region exhibits major shifts. Is this evolution process consistent? </Q>  
"""
    return f"""
You are a professional expert in two-timepoint temporal evolution consistency reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame judgments that confuse non-mask models, clarified by masks' evolution quantifications.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}

Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()



JOBS = ["reason", "counterfactual", "influ", "sem", "plan", "estimate", "st_reason", "st_counterfactual",
        "st_evolution", "st_consistency", "state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]


def get_prompt_for_bi(job, mask_palette, prior, datasets):
    if job == "st_evolution":
        return build_twotime_functional_evolution_prompt(datasets=datasets)
    elif job == "st_consistency":
        return build_twotime_consistency_prompt(datasets=datasets)
    elif job == "st_counterfactual":
        return build_twotime_Spatio_Temporal_counterfactual_prompt(datasets=datasets)
    elif job == "counterfactual":
        return build_twotime_counterfactual_prompt(datasets=datasets)
    elif job == "st_reason":
        return build_twotime_causal_chain_prompt(datasets=datasets)
    elif job == "reason":
        return build_twotime_causal_prompt(datasets=datasets)
    else:
        # 若未定义的 job，返回一个通用占位
        return f"[DEFAULT PROMPT for {job} on {datasets}]"
