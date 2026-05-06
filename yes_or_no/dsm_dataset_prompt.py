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


# === Prompt Construction for Causal Reasoning with DSM ===
# === Prompt Construction for Causal Reasoning with DSM ===
def build_reason_prompt_dsm(datasets='', name="Causal Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **inferring the causes of observed phenomena** in the image, with its unique characteristic being reverse causal tracing that uncovers hidden root causes through semantic conflicts, spatial interactions, and elevation dynamics specific to remote sensing data.
- To elevate difficulty via maximal mask and DSM utilization, leverage mask's pixel-level semantics combined with DSM's elevation gradients for precise differentiations (e.g., exact slope thresholds, height-density interactions, implicit topographic flow graphs) that optics alone may not localize cleanly during construction, while hidden information is used only to discover ambiguous regions, build close alternatives, and verify RGB-only fairness.
- To build carefully constrained cause chains, build a concise 3-5 step evidence chain, where each step builds on mask-DSM derived metrics (e.g., elevation-based fragmentation index, topographic edge effects implying causal flows), creating logical interdependencies that collapse distractors stepwise.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process grounded in reverse-tracing for brain-teaser elevation:
  1. **Identify ultra-ambiguous, mask-DSM dependent phenomenon**: Event where optics suggest multiple similar causes, but mask-DSM synergies (e.g., micro-slopes) pivot the chain.
  2. **Layer multi-step reverse chains**: 3-5 steps with sub-layers and topographic twists (e.g., Step 1: Optical ambiguity; Step 2-4: Initial factors with height densities; Step 5: First twist via slope adjacency; Step 6: Second via elevation fragmentation; Step 7: Third via connectivity gradients; Step 8: Fourth paradoxical inversion; Step 9: Mask-DSM exclusive root with layered revelation).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the inferred cause against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Use semantic categories from the mask and elevation gradients from DSM only as internal tools to mine difficult regions and cross-check fairness; the final question must still be judgeable from visible RGB evidence.
- Introduce **causal twists via semantic-elevation conflicts**: e.g., flooding may look like natural rainfall, but mask shows dense roads/buildings on low DSM elevations blocking channels; landslide may look natural, but mask shows deforested areas on steep DSM slopes.
- Each distractor option must form a superficially plausible cause, but fail under closer inspection of mask-DSM synergies (e.g., attributing erosion to farmland when mask shows no farmland and DSM shows incompatible slopes).
- The correct option must emerge through **multi-step exclusion**, using the semantic map and elevation data as hidden evidence for the model but not visible to the human solver.
- Ensure questions are **image-grounded**: all categories and elevations mentioned must align with the current image/mask/DSM. 
- Avoid forward-looking predictions or planning narratives. Only focus on **explaining why an event occurred given the current scene**.
- **Do not mention 'mask', 'DSM', or 'elevation' in the question**. Humans only see the optical image. These are for the model's internal guidance only.
- Questions must not include vague terms like 'some'; they must reference concrete spatial or semantic cues, e.g., "the river in the lower-right", "the dense buildings in the upper-left".
"""
    examples = """
### Example Questions  

<Q> The lower-right area in the image shows flooded farmland, with visible water accumulation patterns. Which localized causal description is best supported for this flooding pattern? </Q>  

<Q> The central area in the image displays sparse vegetation with soil exposure, with sparse vegetation and exposed soil. Which localized causal description is best supported for this condition? </Q>  

<Q> The upper-left area in the image features bare patches, with visible bare patches and disturbed boundaries. Which localized causal description is best supported for these patches? </Q>  
"""
    return f"""
You are a professional expert in causal reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where same-frame options mislead non-mask/DSM models, resolved only by mask-DSM precise, chained details in remote sensing style.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()


# === Prompt Construction for Counterfactual Reasoning with DSM ===
def build_counterfactual_prompt_dsm(datasets='', name="Counterfactual Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about simulating **alternative historical scenarios**: "what if something in the past had been different?", with its unique characteristic being the exploration of hypothetical past changes and their projected outcomes, leveraging hidden semantic and elevation evidence in remote sensing contexts.
- To elevate difficulty via maximal mask and DSM utilization, use mask for category hypotheticals combined with DSM for elevation simulations (e.g., projected slope changes, height-density extrapolations) that optics can't infer, making outcomes hinge on these invisible computations.
- To build carefully constrained counterfactual chains, use a concise 3-5 step evidence chain, interlinking via mask-DSM metrics (e.g., hypothetical topographic fragmentation), creating chains where small elevation-semantic differences amplify to distinct outcomes.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define ultra-conditional, mask-DSM pivotal assumption**: Hypothesis where optics suggest similar alternatives, but mask-DSM synergies enable precise projections.
  2. **Project multi-step, interdependent chains**: 3-5 steps with metrics (e.g., Step 1: Assumption; Step 2-4: Initial projections; Step 5: First twist via elevation extrapolation; Step 6: Second via semantic simulation; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-DSM exclusive outcome).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the hypothetical outcome against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Use hidden semantic information from the mask, DSM, and any pre-event state only to formulate a hard but fair local counterfactual whose answer can still be judged from visible RGB evidence.  
- Questions must start from a **clear counterfactual assumption** (e.g., "if this industrial zone had not been built", "if the forest had remained intact", "if farmland had not been converted into buildings").  
- Introduce **counterfactual twists**:  
  - An assumption seems positive but hidden mask-DSM information reveals unintended consequences (e.g., if farmland had remained, food supply would increase, but DSM shows it was flood-prone, so flooding would worsen).  
  - A forest preservation assumption seems beneficial, but hidden adjacency to expanding roads on steep DSM means urban sprawl still occurs.  
- Distractor options must appear coherent under the assumption but collapse when checked against hidden evidence.  
- The correct option emerges only after careful elimination, respecting both the counterfactual assumption and the hidden context.  
- Ensure **clear distinction from causal reasoning**:  
  - Causal reasoning explains why something *actually* happened.  
  - Counterfactual reasoning explores what *would have happened if the past were different*.  
- Avoid vague terms like 'some'; always reference concrete spatial or positional cues (e.g., "the farmland in the lower-right", "the industrial block in the upper-left").  
"""
    examples = """
### Example Questions  

<Q> The central area in the image shows flooded farmland. If the nearby buildings had not been constructed, which current-state description would best match the visible evidence now? </Q>  

<Q> The upper-left area in the image displays bare land. If the vegetation had been preserved, which current-state description would best match the visible evidence now? </Q>  

<Q> The lower-right area in the image features an expanded river. If the upstream areas had remained unchanged, which current-state description would best match the visible evidence now? </Q>  
"""
    return f"""
You are a professional expert in counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame scenarios that confuse non-mask/DSM models, pivoted by mask-DSM projective details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()







# === Prompt Construction for Impact Mechanism Reasoning with DSM ===
# === Prompt Construction for Impact Mechanism Reasoning with DSM ===
def build_influ_prompt_dsm(datasets='', name="Impact Mechanism Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **describing forward multi-factor interactive mechanisms**, with its unique characteristic being the unfolding of dynamic feedback loops and non-linear interactions between environmental factors, distinct from backward tracing or hypothetical scenarios in remote sensing analysis.
- To elevate difficulty via maximal mask and DSM utilization, exploit mask for category dynamics combined with DSM for elevation-driven inferences (e.g., slope-induced flow directions, height-based feedback strengths) invisible in optics.
- To build carefully constrained impact chains, use a concise 3-5 step evidence chain, chained via mask-DSM metrics (e.g., topographic interaction graphs), where links are fragile without precise details.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify misleading, mask-DSM dependent change**: Shift where optics imply similar loops, but mask-DSM synergies define uniqueness.
  2. **Unfold multi-step interaction chains**: 3-5 steps with metrics (e.g., Step 1: Trigger; Step 2-4: Interactions with slopes; Step 5: First loop; Step 6: Second twist; Step 7: Third; Step 8: Fourth amplification; Step 9: Mask-DSM exclusive composite).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the impact mechanism against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Impact mechanism reasoning must rely on **hidden semantic from the mask and elevation from DSM**: these reveal distributions and dynamics for inferring interactions. Humans never see them.  
- Introduce **interactive twists**:  
  - A change may appear beneficial but hidden mask-DSM shows secondary effects (e.g., road on steep slope causes erosion).  
  - A forest reduction seems minor but triggers chain erosion on slopes to flooding in lowlands.  
  - A water expansion seems stable but adjacency to elevated industrial zones reveals pollution runoff.  
- Distractor options must describe **linear or oversimplified mechanisms**, while the correct option reveals the **non-linear interactive feedback chain**.  
- Ensure realism: questions should resemble actual environmental monitoring or disaster process analysis.  
- Do not mention 'mask' or 'DSM'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete spatial cues (e.g., "the river in the lower-right", "the dense buildings in the upper-left").  
"""
    examples = """
### Example Questions  

<Q> The farmland near the river in the image is waterlogged. If vegetation coverage decreases in this area, which localized impact mechanism is best supported? </Q>  

<Q> The slope in the image has bare patches. If deforestation continues on this hillside, what dynamic feedback process would likely unfold? </Q>  

<Q> The ridge area in the image shows erosion. If a new structure is added on this terrain, what non-linear mechanisms would primarily emerge? </Q>  
"""
    return f"""
You are a professional expert in impact mechanism reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame mechanisms that baffle non-mask/DSM models, distinguished by mask-DSM dynamic inferences.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()


# === Prompt Construction for Semantic Integration Reasoning with DSM ===
def build_sem_prompt_dsm(datasets='', name="Semantic Integration Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about making **comprehensive high-level judgments** by integrating multiple semantic elements together, with its unique characteristic being the holistic synthesis of diverse land features into functional zones or scene classifications, drawing on remote sensing's emphasis on spatial aggregation.
- To elevate difficulty via maximal mask and DSM utilization, use mask for category fusions combined with DSM for elevation-contextualized integrations (e.g., slope-based zone typing, height-density blends) that optics approximate but can't precisely resolve.
- To build carefully constrained integration chains, use a concise 3-5 step evidence chain, linked via mask-DSM metrics (e.g., topographic heterogeneity indices), building to differentiated judgments.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Catalog ambiguous, mask-DSM pivotal elements**: Features where optics suggest similar groupings, but mask-DSM synergies define uniqueness.
  2. **Aggregate into hyper-nested zones**: 3-5 steps (e.g., Step 1: Pairing; Step 2-4: Clustering with elevations; Step 5: First conflict; Step 6: Second via topographic heterogeneity; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-DSM exclusive judgment).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the integrated judgment against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Semantic integration reasoning must leverage **hidden spatial information from the mask and DSM**: these show distributions and elevations for inferring functional zones. Humans never see them.  
- Introduce **integration twists**:  
  - A dense building cluster near a river may look like industrial, but integration with surrounding farmland on low DSM suggests agricultural-industrial mixed-use conflict.  
  - A forest patch near roads may look like natural reserve, but with steep DSM suggests tourism-functional zone.  
  - A large open space may look barren, but combined with nearby roads on flat DSM reveals planned expansion area.  
- Distractor options must be **superficially convincing but based on partial integration**.  
- The correct option emerges only through **multi-element fusion** validated by hidden evidence.  
- Ensure distinction from other reasoning types.  
- Do not mention 'mask' or 'DSM'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete cues.  
"""
    examples = """
### Example Questions  

<Q> The central valley in the image features a river and surrounding farmland. Synthesizing these elements, what is the most likely functional zone classification? </Q>  

<Q> The elevated ridge in the image includes forest patches and rocky outcrops. Aggregating these features, what overall scene type best fits? </Q>  

<Q> The sloped area in the image combines vegetation and structures. Based on this combination, what inferred zone with potential conflicts emerges? </Q>  
"""
    return f"""
You are a professional expert in semantic integration reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame judgments that confound non-mask/DSM models, refined by mask-DSM nuanced fusions.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()



# === Prompt Construction for Planning Reasoning with DSM ===
# === Prompt Construction for Planning Reasoning with DSM ===
def build_plan_prompt_dsm(datasets='', name="Planning Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — this task is about **multi-step goal-oriented reasoning**, with its unique characteristic being multi-objective optimization in designing or evaluating plans under constraints, emphasizing trade-offs in remote sensing-based spatial planning.
- To elevate difficulty via maximal mask and DSM utilization, employ mask for category constraints combined with DSM for topographic optimizations (e.g., slope accessibility graphs, elevation risk maps) that optics underestimate.
- To build carefully constrained planning chains, use a concise 3-5 step evidence chain, interdependent via mask-DSM metrics (e.g., trade-off indices from elevations and distributions), leading to differentiated plans.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define biased, mask-DSM pivotal multi-goal**: Objective where optics suggest similar optima, but mask-DSM synergies define uniqueness.
  2. **Generate and optimize multi-steply**: 3-5 steps (e.g., Step 1: Biased candidates; Step 2-4: Trade-offs with elevations; Step 5: First nested; Step 6: Second; Step 7: Third; Step 8: Fourth reversal; Step 9: Mask-DSM exclusive plan).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the optimized plan against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Planning logic must be **multi-objective optimization**: e.g., minimize distance, maximize safety, ensure resource proximity, balance environmental sustainability.  
- Introduce **semantic twists** (mask-driven contradictions):  
  - A road seems direct but ends in a blocked building cluster.  
  - A farmland site seems open but far from roads, reducing logistics feasibility.  
  - A riverside site seems resource-rich but is flood-prone (semantic water adjacency).  
  - A forest route seems short but is impassable for vehicles.  
- Distractor options must **look attractive initially** but fail after semantic analysis. The correct option should only emerge after careful elimination.  
- Ensure realism: questions should resemble real-world tasks in disaster response, urban planning, or logistics (e.g., where to place a hospital tent, which route is best for evacuation trucks).  
- Do not mention 'mask' or 'DSM'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete semantic or positional cues (e.g., "bridge in the upper-right", "farmland in the lower-left").  
"""
    examples = """
### Example Questions  

<Q> The central zone in the image includes a river and nearby farmland. To establish an emergency shelter balancing accessibility and flood safety, what is the optimal location? </Q>  

<Q> The upper-left area in the image features forested slopes and roads. For a logistics route from north to south prioritizing speed and terrain stability, which path is best? </Q>  

<Q> The lower-right region in the image shows urban structures and open fields. In planning a new facility considering environmental impact and proximity to resources, what site optimizes the trade-offs? </Q>  
"""
    return f"""
You are a professional expert in planning reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame plans that mislead non-mask/DSM models, optimized by mask-DSM constraint details.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()


# === Prompt Construction for Evaluation Reasoning with DSM ===
def build_estimate_prompt_dsm(datasets='', name="Evaluation Reasoning"):
    specific_constraints = f"""
{TASK_INSTANTIATION_OVERRIDE}
- Adhere strictly to '{name}' — the task is about **evaluating the feasibility of given schemes** (e.g., whether a planned facility, route, or intervention is safe, sustainable, and practical), with its unique characteristic being systematic risk assessment and compatibility checks using spatial constraints in remote sensing evaluation.
- To elevate difficulty via maximal mask and DSM utilization, harness mask for category risks combined with DSM for topographic quantifications (e.g., slope vulnerability scores, elevation conflict zones) that optics can't quantify.
- To build carefully constrained evaluation chains, use a concise 3-5 step evidence chain, chained via mask-DSM metrics (e.g., compatibility indices from elevations and distributions), where assessments hinge on subtle differences.
- Design for same-frame comparability: all options should stay in the same task frame and same specificity level, differing only by one localized evidence-bearing variable.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Present misleading, mask-DSM dependent scheme**: Proposal where optics imply similar feasibilities, but mask-DSM quantifies uniqueness.
  2. **Conduct multi-step assessments**: 3-5 steps (e.g., Step 1: Illusory check; Step 2-4: Risks with elevations; Step 5: First nested; Step 6: Second; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-DSM exclusive judgment).
  3. **Keep the contrast local and controlled**: vary only one region/object relation, direction, extent, boundary, connectivity, temporal order, or identity at a time.
  4. **Design localized distractors**: every wrong option should stay plausible only because it is close to the truth, but fail on one concrete visible mismatch.
  5. **Validate image dependence**: ensure a solver must inspect visible evidence to separate the options instead of relying on commonsense plausibility.
  6. **Validate RGB-only solvability**: use hidden mask/DSM information only to discover ambiguous regions, generate close alternatives, and verify fairness; reject any item whose final answer still depends on hidden-only evidence.
  7. **Cross-verify with hidden mask and DSM**: After chain construction, cross-check the feasibility evaluation against hidden mask and DSM to confirm logical correctness and alignment with actual semantic-elevation evidence.
- Evaluation reasoning must leverage **hidden semantic from the mask and elevation from DSM**: these reveal distributions and terrains for assessing suitability. Humans never see them.  
- Introduce **evaluation twists**:  
  - A site looks spacious but hidden mask-DSM shows it lies in a floodplain with low elevation.  
  - A route looks direct but crosses a dense forest on steep DSM, making it impractical.  
  - A warehouse site looks near roads but is in farmland on sloped DSM, conflicting with use.  
- Distractor options must present **superficially reasonable evaluations** but collapse when hidden constraints are considered.  
- The correct option emerges only after multi-step elimination, considering all constraints.  
- Ensure distinction from other reasoning types.  
- Do not mention 'mask' or 'DSM'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete cues.  
"""
    examples = """
### Example Questions  

<Q> The central zone in the image includes a proposed site for a housing block near the river. Is this location feasible considering safety and sustainability? </Q>  

<Q> The farmland near the river in the image is suggested for expansion. How feasible is this plan given potential environmental risks? </Q>  

<Q> The slope site in the image is planned for a new road. What is the feasibility of this intervention accounting for terrain challenges? </Q>  
"""
    return f"""
You are a professional expert in evaluation reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with same-frame evaluations that confuse non-mask/DSM models, clarified by mask-DSM risk quantifications.

{auto_start_prompt(name=name, datasets=datasets)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-DSM critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask and DSM's pivotal roles.  

    """.strip()




def get_prompt_for_dsm(job, mask_palette, prior, datasets):
    if job == "estimate":
        return build_estimate_prompt_dsm(datasets=datasets)
    elif job == "sem":
        return build_sem_prompt_dsm(datasets=datasets)
    elif job == "influ":
        return build_influ_prompt_dsm(datasets=datasets)
    elif job == "counterfactual":
        return build_counterfactual_prompt_dsm(datasets=datasets)
    elif job == "plan":
        return build_plan_prompt_dsm(datasets=datasets)
    elif job == "reason":
        return build_reason_prompt_dsm(datasets=datasets)
    else:
        # 若未定义的 job，返回一个通用占位
        return f"[DEFAULT PROMPT for {job} on {datasets}]"