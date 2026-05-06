# === Prompt Construction ===
from common_part import auto_start_prompt

# === Prompt Construction for Causal Reasoning ===

# === Prompt Construction for Causal Reasoning ===
def build_reason_prompt(datasets='', name="Causal Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — the task is about **inferring the causes of observed phenomena** in the image, with its unique characteristic being reverse causal tracing that uncovers hidden root causes through semantic conflicts and spatial interactions specific to remote sensing data.
- To elevate difficulty via maximal mask utilization, leverage mask's pixel-level semantics for precise differentiations (e.g., exact density thresholds, adjacency counts, implicit connectivity graphs) that optics cannot resolve, ensuring correct chain only via these hidden details.
- To build hyper-complex cause chains, construct at least 9 steps with quintuple-nested twists, where each step builds on mask-derived metrics (e.g., fragmentation index from category patches, edge density implying interactions), creating logical interdependencies that collapse distractors stepwise.
- Design for high similarity in answers: All options must share 80-90% structural similarity (e.g., all attribute to 'human activity' but differ in subtle mask-validated mechanisms like specific building-road overlaps), causing non-mask models to misjudge due to inability to discern fine distinctions.
- Follow this advanced multi-step process grounded in reverse-tracing for brain-teaser elevation:
  1. **Identify ultra-ambiguous, mask-dependent phenomenon**: Event where optics suggest multiple similar causes, but mask's fine details (e.g., micro-adjacencies) pivot the chain.
  2. **Layer hyper-deep, interdependent reverse chains**: 9+ steps with mask metrics (e.g., Step 1: Optical ambiguity; Step 2-4: Density-based factors; Step 5: First twist via adjacency; Step 6: Second via fragmentation; Step 7: Third via connectivity; Step 8: Fourth paradoxical inversion; Step 9: Mask-exclusive root with quintuple revelation).
  3. **Incorporate quintuple-nested twists**: Multi-level deceptions with high similarity (e.g., twists progressively refine similar hypotheses, only mask's precision breaks the tie).
  4. **Design hyper-similar deceptive distractors**: Options mirror correct chain in wording/structure but falter on mask specifics (e.g., 'blocked by buildings' vs. 'blocked by roads'—mask alone confirms).
  5. **Validate extreme difficulty and mask centrality**: Ensure without mask, similarity leads to random/misjudged choices; with mask, chain logically coheres via remote sensing authenticity like causal landscape analysis.
- Causal reasoning must rely exclusively on **semantic categories and fine-grained distributions from the mask** for resolution, while optics mislead.
- Introduce **causal twists via semantic conflicts**: e.g., flooding may look like natural rainfall, but mask shows dense roads/buildings blocking water channels; forest fire may look natural, but presence of nearby human activity categories (roads, vehicles, industrial sites) suggests human ignition. 
- Each distractor option must form a superficially plausible cause, but fail under closer inspection of mask semantics (e.g., attributing flooding to farmland irrigation when mask shows no farmland nearby).
- The correct option must emerge through **multi-step exclusion**, using the semantic map as hidden evidence for the model but not visible to the human solver.
- Ensure questions are **image-grounded**: all categories mentioned must exist in the current image/mask. 
- Avoid forward-looking predictions or planning narratives (that belongs to planning reasoning). Only focus on **explaining why an event occurred given the current scene**.
- **Do not mention 'mask' or 'segmentation' in the question**. Humans only see the optical image, not the mask. The mask is for the model's internal guidance only.
- Questions must not include vague terms like 'some'; they must reference concrete spatial or semantic cues, e.g., "the river in the lower-right", "the dense buildings in the upper-left".
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Causal Reasoning Examples with High-Similarity Distractors  
<Q> Waterlogging in lower-right zone mimics multiple similar drainage issues from optics, but subtle spatial metrics reveal the true chain; what is the precise root after quintuple twists? </Q>  
<Q> Fire spread in central patch echoes several human-natural hybrids visibly, yet mask adjacencies differentiate; what is the nested ignition mechanism? </Q>  
<Q> Congestion at upper-left junction parallels volume and design causes optically, but density gradients pivot; what hyper-similar causes resolve to? </Q>  
"""
    return f"""
You are a professional expert in causal reasoning for remote sensing images, specializing in hyper-challenging brain-teasers where high-similarity options mislead non-mask models, resolved only by mask's precise, chained details in remote sensing style.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Counterfactual Reasoning ===
def build_counterfactual_prompt(datasets='', name="Counterfactual Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — the task is about simulating **alternative historical scenarios**: "what if something in the past had been different?", with its unique characteristic being the exploration of hypothetical past changes and their projected outcomes, leveraging hidden spatial evidence to reveal unintended consequences in remote sensing contexts.
- To elevate difficulty via maximal mask utilization, use mask for hypothetical reconstructions (e.g., pre-event connectivity simulations, density extrapolations) that optics can't infer, making outcomes hinge on these invisible computations.
- To build hyper-complex counterfactual chains, ensure at least 9 steps with quintuple-nested twists, interlinking via mask metrics (e.g., projected fragmentation changes), creating chains where small mask differences amplify to distinct outcomes.
- Design for high similarity in answers: Options must be nearly identical in scenario description (e.g., all predict 'increased flooding' but vary in degree/mechanism per mask details), leading non-mask models to conflate them.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define ultra-conditional, mask-pivotal assumption**: Hypothesis where optics suggest similar alternatives, but mask's pre-state enables precise projections.
  2. **Project hyper-deep, interdependent chains**: 9+ steps with metrics (e.g., Step 1: Assumption; Step 2-4: Initial projections; Step 5: First twist via extrapolation; Step 6: Second via simulation; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-exclusive outcome).
  3. **Incorporate quintuple-nested twists**: Layers where similar paths diverge subtly via mask, creating brain-teaser differentiations.
  4. **Design hyper-similar deceptive distractors**: Options echo correct in phrasing but err on mask-specific projections (e.g., 'moderate biodiversity loss' vs. 'severe' per hidden connectivity).
  5. **Validate extreme difficulty and mask centrality**: Without mask, similarity causes misjudgment; with mask, resolves via remote sensing tasks like hypothetical modeling.
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
You are a professional expert in counterfactual reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with high-similarity scenarios that confuse non-mask models, pivoted by mask's projective details.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Impact Mechanism Reasoning ===
def build_influ_prompt(datasets='', name="Impact Mechanism Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — the task is about **describing forward multi-factor interactive mechanisms**, with its unique characteristic being the unfolding of dynamic feedback loops and non-linear interactions between environmental factors, distinct from backward tracing or hypothetical scenarios in remote sensing analysis.
- To elevate difficulty via maximal mask utilization, exploit mask for dynamic inferences (e.g., implied flow directions from category gradients, feedback loop strengths from density clusters) invisible in optics.
- To build hyper-complex impact chains, ensure at least 9 steps with quintuple-nested twists, chained via mask metrics (e.g., evolving interaction graphs), where links are fragile without precise details.
- Design for high similarity in answers: Options describe near-identical mechanisms (e.g., all 'erosion leading to pollution' but differ in loop intensity per mask), inducing misjudgment in non-mask models.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Identify misleading, mask-dependent change**: Shift where optics imply similar loops, but mask's gradients define uniqueness.
  2. **Unfold hyper-deep, interdependent chains**: 9+ steps with metrics (e.g., Step 1: Trigger; Step 2-4: Interactions; Step 5: First loop; Step 6: Second twist; Step 7: Third; Step 8: Fourth amplification; Step 9: Mask-exclusive composite).
  3. **Incorporate quintuple-nested twists**: Layers amplifying similarity before mask divergence.
  4. **Design hyper-similar deceptive distractors**: Options parallel in description, only mask metrics separate (e.g., 'moderate runoff' vs. 'severe' per hidden clusters).
  5. **Validate extreme difficulty and mask centrality**: Without mask, similarity blurs choices; with mask, unfolds via remote sensing like process simulation.
- Impact mechanism reasoning must rely on **hidden spatial information from the mask**: the mask reveals the distribution of natural and human-made features (roads, rivers, forests, farmland, buildings, etc.), which the model must use internally to infer interactions. Humans never see the mask.  
- Introduce **interactive twists**:  
  - A change may appear beneficial but hidden mask shows secondary negative effects (e.g., a road improves access but cuts through farmland, reducing food production).  
  - A forest reduction seems minor but triggers chain erosion-flooding-urban damage.  
  - A water expansion seems stable but adjacency to industrial zones reveals pollution feedback.  
- Distractor options must describe **linear or oversimplified mechanisms**, while the correct option reveals the **non-linear interactive feedback chain**.  
- Ensure realism: questions should resemble actual environmental monitoring or disaster process analysis (e.g., “if forest cover decreases, what dynamic feedback will occur?”).  
- Do not mention 'mask' or 'segmentation'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete spatial cues (e.g., "the river in the lower-right", "the dense buildings in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Impact Mechanism Reasoning Examples with High-Similarity Distractors  
<Q> Vegetation decrease in lower-left echoes similar erosion paths visibly, but mask gradients differentiate intensities; what is the precise looped mechanism? </Q>  
<Q> Industrial expansion in upper-right mimics contained feedbacks optically, yet hidden clusters vary; what nuanced impacts unfold? </Q>  
<Q> New road across wetland parallels straightforward dynamics, but mask details pivot degrees; what similar processes emerge distinctly? </Q>  
"""
    return f"""
You are a professional expert in impact mechanism reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with high-similarity mechanisms that baffle non-mask models, distinguished by mask's dynamic inferences.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Semantic Integration Reasoning ===
def build_sem_prompt(datasets='', name="Semantic Integration Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — the task is about making **comprehensive high-level judgments** by integrating multiple semantic elements together, with its unique characteristic being the holistic synthesis of diverse land features into functional zones or scene classifications, drawing on remote sensing's emphasis on spatial aggregation.
- To elevate difficulty via maximal mask utilization, use mask for nuanced fusions (e.g., sub-pixel level category blends, density-based zone typing) that optics approximate but can't precisely resolve.
- To build hyper-complex integration chains, ensure at least 9 steps with quintuple-nested twists, linked via mask metrics (e.g., heterogeneity indices from distributions), building to differentiated judgments.
- Design for high similarity in answers: Options present near-identical zone descriptions (e.g., all 'mixed-use' but vary in subtype per mask blends), causing confusion without mask precision.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Catalog ambiguous, mask-pivotal elements**: Features where optics suggest similar groupings, but mask blends define uniqueness.
  2. **Aggregate into hyper-nested zones**: 9+ steps (e.g., Step 1: Ambiguous pairing; Step 2-4: Clustering; Step 5: First conflict; Step 6: Second via heterogeneity; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-exclusive judgment).
  3. **Incorporate quintuple-nested twists**: Layers heightening similarity before mask's fine distinctions.
  4. **Design hyper-similar deceptive distractors**: Options mirror in classification but err on mask specifics (e.g., 'urban-mixed' vs. 'agri-urban' per hidden densities).
  5. **Validate extreme difficulty and mask centrality**: Without mask, similarity ambiguates; with mask, fuses to remote sensing like advanced classification.
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
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Semantic Integration Reasoning Examples with High-Similarity Distractors  
<Q> Integrating upper-left elements, optics implying similar mixed zones, but mask blends differentiate subtypes; what is the precise classification? </Q>  
<Q> Aggregating lower-right features, visible as parallel agriculturals, yet hidden densities vary; what nuanced zone fits? </Q>  
<Q> Combining upper-right components, echoing conflict types, but mask details pivot; what similar zones resolve to? </Q>  
"""
    return f"""
You are a professional expert in semantic integration reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with high-similarity judgments that confound non-mask models, refined by mask's nuanced fusions.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Planning Reasoning ===
def build_plan_prompt(datasets='', name="Planning Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — this task is about **multi-step goal-oriented reasoning**, with its unique characteristic being multi-objective optimization in designing or evaluating plans (e.g., routes, locations) under constraints, emphasizing trade-offs in remote sensing-based spatial planning.
- To elevate difficulty via maximal mask utilization, employ mask for optimization constraints (e.g., precise accessibility graphs, risk density maps) that optics underestimate.
- To build hyper-complex planning chains, ensure at least 9 steps with quintuple-nested twists, interdependent via mask metrics (e.g., trade-off indices from distributions), leading to differentiated plans.
- Design for high similarity in answers: Options outline near-identical plans (e.g., all 'route via roads' but vary in viability per mask constraints), misleading non-mask models to equivalent choices.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Define biased, mask-pivotal multi-goal**: Objective where optics suggest similar optima, but mask constraints define uniqueness.
  2. **Generate and optimize hyper-deeply**: 9+ steps (e.g., Step 1: Biased candidates; Step 2-4: Trade-offs; Step 5: First nested; Step 6: Second; Step 7: Third; Step 8: Fourth reversal; Step 9: Mask-exclusive plan).
  3. **Incorporate quintuple-nested twists**: Layers amplifying similarity before mask's distinctions.
  4. **Design hyper-similar deceptive distractors**: Options parallel in plan structure, only mask metrics separate (e.g., 'safe short route' vs. 'risky short' per hidden risks).
  5. **Validate extreme difficulty and mask centrality**: Without mask, similarity randomizes; with mask, optimizes via remote sensing like GIS planning.
- Planning logic must be **multi-objective optimization**: e.g., minimize distance, maximize safety, ensure resource proximity, balance environmental sustainability.  
- Introduce **semantic twists** (mask-driven contradictions):  
  - A road seems direct but ends in a blocked building cluster.  
  - A farmland site seems open but far from roads, reducing logistics feasibility.  
  - A riverside site seems resource-rich but is flood-prone (semantic water adjacency).  
  - A forest route seems short but is impassable for vehicles.  
- Distractor options must **look attractive initially** but fail after semantic analysis. The correct option should only emerge after careful elimination.  
- Ensure realism: questions should resemble real-world tasks in disaster response, urban planning, or logistics (e.g., where to place a hospital tent, which route is best for evacuation trucks).  
- Do not mention 'mask' or 'segmentation'. Humans only see the optical image, not the mask.  
- Avoid vague terms like 'some'; always reference concrete semantic or positional cues (e.g., "bridge in the upper-right", "farmland in the lower-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Planning Reasoning Examples with High-Similarity Distractors  
<Q> For hazard response, optics implying similar facility sites, but mask constraints differentiate viabilities; what is the precise optimized plan? </Q>  
<Q> Convoy routing echoes parallel feasible paths visibly, yet hidden metrics vary risks; which nuanced route balances? </Q>  
"""
    return f"""
You are a professional expert in planning reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with high-similarity plans that mislead non-mask models, optimized by mask's constraint details.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
Based on the above requirements, generate one extremely complex, mask-critical question that aligns with '{name}' for remote sensing image challenges, and provide a complete step-by-step solving rationale highlighting mask's pivotal role.  

    """.strip()


# === Prompt Construction for Evaluation Reasoning ===
def build_estimate_prompt(datasets='', name="Evaluation Reasoning"):
    specific_constraints = f"""
- Adhere strictly to '{name}' — the task is about **evaluating the feasibility of given schemes** (e.g., whether a planned facility, route, or intervention is safe, sustainable, and practical), with its unique characteristic being systematic risk assessment and compatibility checks using spatial constraints in remote sensing evaluation.
- To elevate difficulty via maximal mask utilization, harness mask for risk quantifications (e.g., vulnerability scores from category interactions, invisible conflict zones) that optics can't quantify.
- To build hyper-complex evaluation chains, ensure at least 9 steps with quintuple-nested twists, chained via mask metrics (e.g., compatibility indices), where assessments hinge on subtle differences.
- Design for high similarity in answers: Options give near-identical feasibility ratings (e.g., all 'moderately feasible' but vary in rationale per mask risks), causing non-mask models to overlook distinctions.
- Follow this advanced multi-step process for brain-teaser elevation:
  1. **Present misleading, mask-dependent scheme**: Proposal where optics imply similar feasibilities, but mask quantifies uniqueness.
  2. **Conduct hyper-deep assessments**: 9+ steps (e.g., Step 1: Illusory check; Step 2-4: Risks; Step 5: First nested; Step 6: Second; Step 7: Third; Step 8: Fourth inversion; Step 9: Mask-exclusive judgment).
  3. **Incorporate quintuple-nested twists**: Layers building similarity before mask's precise divergences.
  4. **Design hyper-similar deceptive distractors**: Options parallel in evaluation but fail on mask specifics (e.g., 'feasible with minor risks' vs. 'infeasible due to major' per hidden scores).
  5. **Validate extreme difficulty and mask centrality**: Without mask, similarity misleads; with mask, assesses via remote sensing like feasibility mapping.
- Evaluation reasoning must leverage **hidden spatial information from the mask**: the mask reveals distributions of roads, water, vegetation, farmland, and buildings, which the model must use internally to assess suitability. Humans never see the mask.  
- Introduce **evaluation twists**:  
  - A site looks spacious but hidden mask shows it lies in a floodplain.  
  - A route looks direct but crosses a dense forest, making it impractical.  
  - A warehouse site looks near roads but is in farmland, conflicting with agricultural use.  
- Distractor options must present **superficially reasonable evaluations** but collapse when hidden constraints are considered.  
- The correct option emerges only after multi-step elimination, considering all constraints.  
- Ensure distinction from other reasoning types:  
  - Not tracing actual causes (causal).  
  - Not assuming past changes (counterfactual).  
  - Not optimizing new plans (planning).  
  - Not describing dynamic processes (impact mechanism).  
  - Not synthesizing zones (semantic integration).  
  - Evaluation is about **feasibility judgment of a specific scheme**.  
- Do not mention 'mask' or 'segmentation'. Humans only see the optical image.  
- Avoid vague terms like 'some'; always reference concrete cues (e.g., "the farmland in the lower-right", "the cluster of buildings in the upper-left").  
"""
    examples = """  
**The examples are only used to indicate the content of the questions to be set, and the number of words in each question should be at least 40-60**
### Hyper-Complex Evaluation Reasoning Examples with High-Similarity Distractors  
<Q> Scheme for extension in western zones echoes moderate feasibilities visibly, but mask risks differentiate; what is the precise assessment? </Q>  
<Q> Park plan in central appears similarly sustainable, yet hidden quantifications vary; does it hold? </Q>  
<Q> Warehouse site parallels practical ratings, but mask details pivot; what nuanced feasibility emerges? </Q>  
"""
    return f"""
You are a professional expert in evaluation reasoning for remote sensing images, specializing in hyper-challenging brain-teasers with high-similarity evaluations that confuse non-mask models, clarified by mask's risk quantifications.

{auto_start_prompt(datasets, name)}
{specific_constraints}
{examples}
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