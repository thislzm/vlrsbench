

CLAIM_GROUNDING_OVERRIDE = """
### HARD EVIDENCE-GROUNDING OVERRIDE
- Preserve the requested task category, but instantiate it as one localized claim, not a broad scene-level plausibility judgment.
- Anchor the claim to one specific target region/object/change relation that a solver can check from visible evidence.
- Avoid wording that leaks labels through generic priors such as predominantly, primarily, generally, typically, mostly, appears, suggests, or indicates unless the wording is tied to explicit measurable visual evidence.
- Make the truth value depend on concrete anchors such as counts, adjacency, direction, boundary contact, relative position, connectivity, extent, or temporal order.
- For temporal tasks, hinge the claim on one explicit before/after or sequence relation instead of a generic narrative of development.
- Do not rely on overall plausibility, desirability, or hidden functional stories to decide True/False.
- Privileged information should be used only to identify hard regions, build close alternatives, and validate fairness; the final claim must remain judgeable from RGB alone.
"""

CLAIM_DEPTH_OVERRIDE = """
### HARD CLAIM-DEPTH OVERRIDE
- Treat the claim as a verification trap, not a caption: it must require checking one target, one stable reference landmark, and one measurable relation/comparison or limiting condition.
- A claim is too shallow if a solver can answer it from one salient word such as `new`, `disappeared`, `unchanged`, `demolished`, or `expanded` without checking additional local evidence.
- For temporal items, do not let a bare appearance/disappearance/persistence statement decide the label by itself. The same sentence must also require at least one extra RGB-verifiable constraint such as attachment vs isolation, boundary contact, gap closure, continuity break, count threshold, directional extension, or asymmetry between two nearby subregions.
- Prefer paired local checks over broad summaries: compare two candidate regions, or verify one evolving target against one nearby stable reference.
- If the claim still reads like direct change detection after drafting, reject it and rewrite it with tighter local constraints.
"""

# FAST — bolded category and color
FAST_descriptions = {
    "Airplane": "**Airplane** — **Bright red (rgb:(215, 25, 21))** is the category name associated with that mask color.",
    "Baseball field": "**Baseball field** — **Dark blue (rgb:(29, 23, 215))** is the category name associated with that mask color.",
    "Basketball court": "**Basketball court** — **Cyan (rgb:(34, 173, 211))** is the category name associated with that mask color.",
    "Tennis court": "**Tennis court** — **Navy (rgb:(23, 48, 129))** is the category name associated with that mask color.",
    "Football field": "**Football field** — **Maroon (rgb:(129, 23, 32))** is the category name associated with that mask color.",
    "Bridge": "**Bridge** — **Dark magenta (rgb:(128, 23, 99))** is the category name associated with that mask color.",
    "Roundabout": "**Roundabout** — **Pink (rgb:(202, 25, 152))** is the category name associated with that mask color.",
    "Intersection": "**Intersection** — **Brown (rgb:(133, 76, 20))** is the category name associated with that mask color.",
    "Large land vehicle": "**Large land vehicle** — **Golden (rgb:(216, 171, 31))** is the category name associated with that mask color.",
    "Small land vehicle": "**Small land vehicle** — **Lime green (rgb:(118, 190, 30))** is the category name associated with that mask color.",
    "Engineering vehicle": "**Engineering vehicle** — **Magenta (rgb:(216, 33, 215))** is the category name associated with that mask color.",
    "Ship": "**Ship** — **Green (rgb:(21, 201, 92))** is the category name associated with that mask color."
}

# SIOR — bolded category and color
SIOR_descriptions = {
    "airplane": "**airplane** — **Bright red (rgb:(215, 25, 21))** is the category name associated with that mask color.",
    "airport": "**airport** — **Bright green (rgb:(22, 215, 29))** is the category name associated with that mask color.",
    "baseballfield": "**baseballfield** — **Dark blue (rgb:(29, 23, 215))** is the category name associated with that mask color.",
    "basketballcourt": "**basketballcourt** — **Cyan (rgb:(34, 173, 211))** is the category name associated with that mask color.",
    "bridge": "**bridge** — **Dark magenta (rgb:(128, 23, 99))** is the category name associated with that mask color.",
    "chimney": "**chimney** — **Olive (rgb:(135, 123, 22))** is the category name associated with that mask color.",
    "dam": "**dam** — **Blue (rgb:(20, 100, 207))** is the category name associated with that mask color.",
    "expressway-service-area": "**expressway-service-area** — **Yellow green (rgb:(201, 215, 22))** is the category name associated with that mask color.",
    "expressway-toll-station": "**expressway-toll-station** — **Forest green (rgb:(25, 129, 20))** is the category name associated with that mask color.",
    "golffield": "**golffield** — **Orange (rgb:(214, 120, 25))** is the category name associated with that mask color.",
    "groundtrackfield": "**groundtrackfield** — **Purple (rgb:(82, 20, 146))** is the category name associated with that mask color.",
    "harbor": "**harbor** — **Teal blue (rgb:(23, 85, 132))** is the category name associated with that mask color.",
    "overpass": "**overpass** — **Aquamarine (rgb:(36, 211, 208))** is the category name associated with that mask color.",
    "ship": "**ship** — **Green (rgb:(21, 201, 92))** is the category name associated with that mask color.",
    "stadium": "**stadium** — **Blue (rgb:(34, 73, 216))** is the category name associated with that mask color.",
    "storagetank": "**storagetank** — **Sky blue (rgb:(32, 140, 215))** is the category name associated with that mask color.",
    "tenniscourt": "**tenniscourt** — **Navy (rgb:(23, 48, 129))** is the category name associated with that mask color.",
    "trainstation": "**trainstation** — **Sea green (rgb:(23, 135, 78))** is the category name associated with that mask color.",
    "vehicle": "**vehicle** — **Magenta (rgb:(216, 33, 215))** is the category name associated with that mask color."
}

# SOTA — bolded category and color (concise, generalized inference cue)
SOTA_descriptions = {
    "airport": "**airport** — **Bright green (rgb:(22, 215, 29))** is the category name associated with that mask color.",
    "baseball-diamond": "**baseball-diamond** — **Dark blue (rgb:(29, 23, 215))** is the category name associated with that mask color.",
    "basketball-court": "**basketball-court** — **Cyan (rgb:(34, 173, 211))** is the category name associated with that mask color.",
    "bridge": "**bridge** — **Dark magenta (rgb:(128, 23, 99))** is the category name associated with that mask color.",
    "container-crane": "**container-crane** — **Aquamarine (rgb:(37, 211, 156))** is the category name associated with that mask color.",
    "ground-track-field": "**ground-track-field** — **Purple (rgb:(82, 20, 146))** is the category name associated with that mask color.",
    "harbor": "**harbor** — **Teal blue (rgb:(23, 85, 132))** is the category name associated with that mask color.",
    "helicopter": "**helicopter** — **Magenta (rgb:(214, 25, 104))** is the category name associated with that mask color.",
    "helipad": "**helipad** — **Teal (rgb:(20, 129, 113))** is the category name associated with that mask color.",
    "large-vehicle": "**large-vehicle** — **Golden (rgb:(216, 171, 31))** is the category name associated with that mask color.",
    "plane": "**plane** — **Bright red (rgb:(215, 25, 21))** is the category name associated with that mask color.",
    "roundabout": "**roundabout** — **Pink (rgb:(202, 25, 152))** is the category name associated with that mask color.",
    "ship": "**ship** — **Green (rgb:(21, 201, 92))** is the category name associated with that mask color.",
    "small-vehicle": "**small-vehicle** — **Lime green (rgb:(118, 190, 30))** is the category name associated with that mask color.",
    "soccer-ball-field": "**soccer-ball-field** — **Maroon (rgb:(129, 23, 32))** is the category name associated with that mask color.",
    "storage-tank": "**storage-tank** — **Sky blue (rgb:(32, 140, 215))** is the category name associated with that mask color.",
    "swimming-pool": "**swimming-pool** — **Violet (rgb:(141, 35, 209))** is the category name associated with that mask color.",
    "tennis-court": "**tennis-court** — **Navy (rgb:(23, 48, 129))** is the category name associated with that mask color."
}

# loveDA — bolded category and color (concise, generalized inference cue)
loveDA_descriptions = {
    "Background": "**Background** — **Red (rgb:(255, 0, 0))** is the category name associated with that mask color.",
    "Buildings": "**Buildings** — **Blue (rgb:(0, 0, 255))** is the category name associated with that mask color.",
    "Roads": "**Roads** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Water bodies": "**Water bodies** — **Blue-green (rgb:(0, 125, 255))** is the category name associated with that mask color.",
    "Barren land": "**Barren land** — **Yellow (rgb:(255, 255, 0))** is the category name associated with that mask color.",
    "Forests": "**Forests** — **Green (rgb:(0, 255, 0))** is the category name associated with that mask color.",
    "Agricultural land": "**Agricultural land** — **Cyan (rgb:(0, 255, 255))** is the category name associated with that mask color.",
    "Insignificant": "**Insignificant** — **Black (rgb:(0, 0, 0))** is the category name associated with that mask color."
}

# GID — bolded category and color (concise, generalized inference cue)
GID_descriptions = {
    "Industrial land": "**Industrial land** — **Red (rgb:(200, 0, 0))** is the category name associated with that mask color.",
    "Urban residential": "**Urban residential** — **Pink (rgb:(250, 0, 150))** is the category name associated with that mask color.",
    "Rural residential": "**Rural residential** — **Light red (rgb:(200, 150, 150))** is the category name associated with that mask color.",
    "Traffic land": "**Traffic land** — **Pale red (rgb:(250, 150, 150))** is the category name associated with that mask color.",
    "Paddy field": "**Paddy field** — **Green (rgb:(0, 200, 0))** is the category name associated with that mask color.",
    "Irrigated land": "**Irrigated land** — **Yellow-green (rgb:(150, 250, 0))** is the category name associated with that mask color.",
    "Dry cropland": "**Dry cropland** — **Light green (rgb:(150, 200, 150))** is the category name associated with that mask color.",
    "Garden plot": "**Garden plot** — **Purple (rgb:(200, 0, 200))** is the category name associated with that mask color.",
    "Arbor woodland": "**Arbor woodland** — **Deep purple (rgb:(150, 0, 250))** is the category name associated with that mask color.",
    "Shrub land": "**Shrub land** — **Light purple (rgb:(150, 150, 250))** is the category name associated with that mask color.",
    "Natural grassland": "**Natural grassland** — **Golden yellow (rgb:(250, 200, 0))** is the category name associated with that mask color.",
    "Artificial grassland": "**Artificial grassland** — **Yellow (rgb:(200, 200, 0))** is the category name associated with that mask color.",
    "River": "**River** — **Blue (rgb:(0, 0, 200))** is the category name associated with that mask color.",
    "Lake": "**Lake** — **Cyan-blue (rgb:(0, 150, 200))** is the category name associated with that mask color.",
    "Pond": "**Pond** — **Light cyan (rgb:(0, 200, 250))** is the category name associated with that mask color."
}

# Potsdam — bolded category and color (concise, generalized inference cue)
Potsdam_descriptions = {
    "Impervious surfaces": "**Impervious surfaces** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Buildings": "**Buildings** — **Blue (rgb:(0, 0, 255))** is the category name associated with that mask color.",
    "Low vegetation": "**Low vegetation** — **Cyan (rgb:(0, 255, 255))** is the category name associated with that mask color.",
    "Trees": "**Trees** — **Green (rgb:(0, 255, 0))** is the category name associated with that mask color.",
    "Cars": "**Cars** — **Yellow (rgb:(255, 255, 0))** is the category name associated with that mask color.",
    "Background": "**Background** — **Red (rgb:(255, 0, 0))** is the category name associated with that mask color."
}

# Vaihingen — bolded category and color (concise, generalized inference cue)
Vaihingen_descriptions = {
    "Impervious surfaces": "**Impervious surfaces** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Building roofs": "**Building roofs** — **Blue (rgb:(0, 0, 255))** is the category name associated with that mask color.",
    "Low vegetation": "**Low vegetation** — **Cyan (rgb:(0, 255, 255))** is the category name associated with that mask color.",
    "Trees": "**Trees** — **Green (rgb:(0, 255, 0))** is the category name associated with that mask color.",
    "Cars": "**Cars** — **Yellow (rgb:(255, 255, 0))** is the category name associated with that mask color.",
    "Power lines": "**Power lines** — **Light gold (rgb:(255, 125, 0))** is the category name associated with that mask color.",
    "Building facades": "**Building facades** — **Blue-green (rgb:(0, 125, 255))** is the category name associated with that mask color.",
    "Shrubs": "**Shrubs** — **Yellow-green (rgb:(125, 255, 0))** is the category name associated with that mask color.",
    "Fences/hedges": "**Fences/hedges** — **Cyan-green (rgb:(0, 255, 125))** is the category name associated with that mask color."
}

# xView2 — bolded category and color (concise, generalized inference cue)
xView2_descriptions = {
    "No damage": "**No damage** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Minor damage": "**Minor damage** — **Light yellow (rgb:(255, 255, 150))** is the category name associated with that mask color.",
    "Major damage": "**Major damage** — **Light blue (rgb:(200, 255, 255))** is the category name associated with that mask color.",
    "Destroyed": "**Destroyed** — **Light green (rgb:(200, 255, 200))** is the category name associated with that mask color."
}

# SECOND — bolded category and color (concise, generalized inference cue)
SECOND_descriptions = {
    "Non-change": "**Non-change** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Low vegetation": "**Low vegetation** — **Dark green (rgb:(0, 128, 0))** is the category name associated with that mask color.",
    "N.v.g. surface": "**N.v.g. surface** — **Gray (rgb:(128, 128, 128))** is the category name associated with that mask color.",
    "Tree": "**Tree** — **Bright green (rgb:(0, 255, 0))** is the category name associated with that mask color.",
    "Water": "**Water** — **Blue (rgb:(0, 0, 255))** is the category name associated with that mask color.",
    "Building": "**Building** — **Dark red (rgb:(128, 0, 0))** is the category name associated with that mask color.",
    "Playground": "**Playground** — **Orange-red (rgb:(255, 0, 0))** is the category name associated with that mask color."
}


def get_palette(dataset_path, image_name):
    """
    根据数据集路径和图像名获取调色板描述

    Args:
        dataset_path: 数据集路径，如 "bi-change/SECOND"
        image_name: 图像文件名，如 "image_001.png"
    """

    # 基础prompt
    mask_palette = 'Masks Image Category—Color Format: Category — Color (rgb:R,G,B) | Color in mask = semantic label (map exact RGB to category) | Use the description only as a neutral label glossary, not as a planning, causality, or plausibility prior | Treat Background/No-change as context, not a primary target.'

    # 根据数据集选择对应的描述字典
    descriptions = None
    dataset_name = dataset_path.name  # 获取最后一个目录名
    dataset_str = str(dataset_path)  # 转换为字符串用于判断

    if dataset_name == "SECOND":
        descriptions = SECOND_descriptions
    elif dataset_name == "xView2":
        descriptions = xView2_descriptions
    elif dataset_name == "Potsdam":
        descriptions = Potsdam_descriptions
    elif dataset_name == "Vaihingen":
        descriptions = Vaihingen_descriptions
    elif dataset_name == "GID":
        descriptions = GID_descriptions
    elif dataset_name == "loveDA":
        descriptions = loveDA_descriptions
    elif dataset_name == "FAST":
        descriptions = FAST_descriptions
    elif dataset_name == "SIOR":
        descriptions = SIOR_descriptions
    elif dataset_name == "SOTA":
        descriptions = SOTA_descriptions

    if not descriptions:
        return mask_palette

    # 准备txt文件路径列表
    # 转换为Path对象处理

    # 获取描述字典
    descriptions = globals().get(f"{dataset_name}_descriptions", {})

    if not descriptions:
        return mask_palette

    # 准备txt文件路径列表
    txt_paths = []
    txt_filename = image_name.replace('.png', '.txt').replace('.jpg', '.txt')

    # 根据数据集类型确定txt文件路径
    if dataset_name == "miniucd":
        # miniucd特殊格式：读取2017_txt, 2018_txt, 2019_txt
        years = ["2017", "2018", "2019"]
        for year in years:
            txt_paths.append(dataset_path / "label" / f"{year}_txt" / txt_filename)

    elif "bi-change" in dataset_str:
        # bi-change类型：读取label1_txt和label2_txt
        txt_paths.append(dataset_path / "label1_txt" / txt_filename)
        txt_paths.append(dataset_path / "label2_txt" / txt_filename)

    else:
        # 其他类型：只读取masks_txt
        txt_paths.append(dataset_path / "masks_txt" / txt_filename)

    # 收集所有类别（使用set自动去重）
    all_categories = set()

    for txt_path in txt_paths:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 解析格式：R, G, B, "类别", "描述"
                parts = line.split(',')
                if len(parts) >= 4:
                    # 提取类别名称
                    category = parts[3].strip().strip('"').strip("'")

                    # 跳过Unknown类别
                    if category and category != "Unknown":
                        all_categories.add(category)

        except FileNotFoundError:
            continue  # 文件不存在时继续
        except Exception:
            continue

    # 根据类别匹配描述并添加到prompt
    for category in sorted(all_categories):
        if category in descriptions:
            mask_palette += '\n' + descriptions[category]

    return mask_palette


def get_mask_palette_and_prior(datasets, dataset_path, im1_name):
    mask_palette = "Colors are only identifiers of class membership (color → class); they do not encode object identity, instance ID, temporal order, or any other information.\n"
    prior = ''

    ###################  images  masks
    if datasets == 'FAST':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains images with a spatial resolution between 0.3m and 0.8m, covering more than 100 civilian airports, ports, and cities across Asia, the Americas, Europe, Africa, and Oceania."

    elif datasets == 'SIOR':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset has a spatial resolution ranging from 0.5m to 30m. It mainly captures suburban urban areas, with categories covering transportation, sports fields, infrastructure, and man-made structures."

    elif datasets == 'SOTA':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains dense and sparse instance regions, such as ships in harbors and vehicles in parking lots. Objects exhibit arbitrary orientations, diverse scales, and extreme aspect ratios."

    elif datasets == 'loveDA':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides 30 cm resolution urban and rural scenes from Nanjing, Changzhou, and Wuhan, covering diverse land covers such as roads, buildings, water, forests, and farmland."

    elif datasets == 'GID':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides 1 m resolution land-cover annotations across cities in China, Vietnam, Japan, and France, covering diverse urban, rural, agricultural, and water landscapes."

    ###################  images  masks  dsms
    elif datasets == 'Potsdam':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "Potsdam offers 12.5 cm resolution imagery of a historic city with dense settlements, large building complexes, and narrow streets."

    elif datasets == 'Vaihingen':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "Vaihingen provides 15 cm resolution imagery of a small village with detached houses and low-rise buildings, captured with near-infrared, red, and green bands."

    ###################  im1 im2 label1 label2
    elif datasets == 'xView2':  # 单独设置
        mask_palette = "Colors are only identifiers of object ID (color → object ID); they do not encode object identity, instance ID, temporal order, or any other information.\n"
        mask_palette += "In mask1 and mask2, the bounding-box stroke color is used solely as a unique building identifier (color = object ID). It carries no semantic meaning beyond identifying the same building across time." \
                        "Separately, the fill color inside each box (not the box stroke) encodes the building’s damage severity between mask1 → mask2. The fill color is independent of the bounding-box color. Use the temporal change mask1 -> mask2 and the fill color to reason about damage progression and to formulate chain-of-thought style questions.\n"
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides <80 cm resolution satellite imagery from 19 natural disaster events for building damage assessment."

    elif datasets == 'SECOND':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains urban imagery from Chinese cities such as Hangzhou, Chengdu, and Shanghai, focusing on land-cover change detection including vegetation, buildings, and impervious surfaces."

    ###################  im1 im2 im3 label1 label2 label3
    elif datasets == 'miniucd':
        mask_palette += 'In label1 and label2, every color consistently maps to the same semantic class across both masks.\n'
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides 10 cm resolution annotations over parts of Tallinn, Estonia, covering diverse urban land-cover types."

    elif datasets == 'spaceNet7':
        mask_palette = "Colors are only identifiers of object ID (color → object ID); they do not encode object identity, instance ID, temporal order, or any other information.\n"
        mask_palette += 'In label1, label2, and label3, every color corresponds to a single, unique building ID — the color is an object identifier only and does not encode any other semantic information.' \
                        'Use the temporal sequence mask1 → mask2 → mask3 to reason about changes to a specific building (e.g., construction, demolition, expansion, merging, splitting, or no change) and design step-by-step reasoning questions based on that sequence.'
        prior = "This dataset offers 4 m resolution imagery for building change detection across rapidly urbanizing regions."

    return mask_palette, prior


import random

import random

def get_answer_bias_instruction():
    """Prefer False labels without encouraging linguistic deception."""
    bias_threshold = 0.6  # 60% chance for False; adjust to 0.7 for 70%, etc.
    if random.random() < bias_threshold:
        return "Additional Instruction: Prefer generating a False claim to counter the model's tendency to default to True, but make the False label come from one localized RGB-verifiable contradiction rather than from deceptive wording, hidden-only evidence, or surface plausibility tricks."
    else:
        return "Additional Instruction: When generating a True claim, keep the same difficulty level as False claims by using dense, localized evidence checks rather than stylistic cues or answer-prior tricks."



def start_common_prompt_dsm( name=""):
    return f"""
{CLAIM_GROUNDING_OVERRIDE}
{CLAIM_DEPTH_OVERRIDE}

You are an expert in crafting intricate visual reasoning challenges
The model will internally inspect a MASK segmentation map and a DSM elevation map (for verification only). Human solvers will only see the RGB image.  

Objective: generate one high-quality True/False challenge in the '{name}' style.
{get_answer_bias_instruction()}

Requirements for the generated output:
- Produce a **single declarative sentence** (one statement) that can be judged as **True** or **False** using only the visible RGB image.
- Before the statement, provide a concise coarse-to-fine summary of the scene and confirm key visual cues against the RGB image.
- The statement must be verifiable from the RGB image and must **not** contain dataset-internal terms such as 'mask', 'DSM', 'segmentation', or file identifiers.
- Prefer statements that require multi-step visual reasoning rather than trivial pixel facts. Useful patterns include:
  - Quantifiers: "No", "At least one", "Exactly one", "More than two".
  - Spatial anchors: "in the lower-right quadrant", "adjacent to the main pier", "north of the large building".
  - Comparative or relational claims: "larger than", "closer to", "denser than".
  - Concrete visible relations: "contains stacked containers beside the pier", "has several vehicles aligned in marked bays", "shows a bright cleared patch with sharp soil boundaries".
- Avoid unverifiable assumptions about non-visible properties (e.g., internal cargo, unseen signage).
- The output must include: a short design rationale, the single-sentence claim enclosed in `<Q>...</Q>`, the binary `<Answer>` (`True` or `False`), and a `<think>` block that justifies the assigned answer step-by-step using only observable RGB evidence.
- If this prompt is wrapped by a JSON schema, do NOT emit literal `<Q>`, `<Answer>`, or `<think>` tags; place their content directly into the corresponding JSON fields instead.

- **Mandatory Multi-Step Reasoning**: The single-sentence claim **must** incorporate at least 2-3 combined elements (e.g., quantifiers + descriptive spatial anchors + relational claims) to require verifiable multi-step visual reasoning. Internally, build a chain of 4-6 steps to finalize the claim (e.g., Step 1: Identify key regions with descriptive anchors like 'the empty area in the lower-left corner of the image'; Step 2: Apply quantifiers to visible features within those regions; Step 3: Establish relations or comparisons (e.g., 'adjacent to'); Step 4: compare concrete visible attributes and boundaries; Step 5: Cross-verify cues for ambiguity; Step 6: Ensure the claim is testable and grounded in visible details).
- **Prohibit Trivial Outputs**: Strictly avoid simple pixel-level or descriptive statements without reasoning layers; instead, design claims that appear straightforward but demand step-by-step analysis in  to judge True/False.
- **Structural Verification Requirement**: Every claim must contain a target anchor plus at least one stable reference landmark, and it must require at least one measurable relation or comparison between them.
- **Descriptive Spatial Anchor Requirement**: **Must include precise, descriptive spatial anchors** that combine location with visible features (e.g., 'the empty area in the lower-left corner of the image', 'the cleared patch adjacent to the main structure', 'the dense cluster north of the large building') in the claim to anchor the reasoning to specific, observable areas. Avoid abstract terms like 'lower-left quadrant' without descriptive integration.
- **Claim Length and Richness Requirement**: To ensure the claim is informationally complete and abundant in details, **the single-sentence claim must be at least 20-40 words long**, achieved by logically incorporating nested descriptions, quantifiers, relations, and inferences (e.g., using clauses to elaborate visible cues). Avoid short, incomplete sentences; prioritize richness that supports multi-step verification without redundancy.
- **Deterministic Language Requirement**: **The claim must use definitive, certain language** to form a clear declarative statement that can be judged strictly as True or False. Avoid any words or phrases implying uncertainty, such as 'seems', 'appears', 'suggests', 'possibly', 'might', or equivalents in other languages. Instead, use direct, assertive terms like 'consists of', 'contains', 'demonstrates', 'is used as', 'shows', or 'indicates' only if it conveys certainty based on visible evidence.
- **Label Balance Requirement**: keep the dataset balanced across `True` and `False`, but do not make any individual claim misleading by design. A claim should be judged from localized visible evidence, not from a surface-writing trick or answer-bias rule.
- **Difficulty Assurance**: Validate that the claim is challenging (e.g., surface-level ambiguity resolvable only through detailed RGB evidence), aligning with high-quality reasoning standards without relying on dataset-specific internals.
- **Force Logical Height Integration**: For datasets with DSM elevation data (used internally for verification), **every claim must logically incorporate height-related information** grounded in visible RGB cues (e.g., 'elevated ridge', 'sloped terrain higher than adjacent areas', 'raised platforms indicating height variations'). Ensure this is naturally integrated into the reasoning (e.g., combined with spatial anchors or functional inferences) to enhance verifiability, without mentioning DSM or other internals.
Final required elements: a brief design rationale, <Q> (single-sentence claim), <Answer> (`True` or `False`), and  (step-by-step justification using only RGB evidence).

Use the following color→category mapping for internal verification (do not reveal these mappings in the question text):

"""


def start_common_prompt(name=""):
    return f"""
{CLAIM_GROUNDING_OVERRIDE}
{CLAIM_DEPTH_OVERRIDE}

You are an expert in crafting intricate visual reasoning challenges
The model will internally inspect one MASK segmentation map (for verification only) and the optical RGB image (which humans will see).

Objective: generate one high-quality True/False challenge in the '{name}' style.
{get_answer_bias_instruction()}

Instructions:
- First provide a concise coarse→fine summary of the RGB scene and confirm visual cues you will rely on.
- Then write a **single-sentence declarative claim** whose truth can be determined from the RGB image alone.
- Do NOT mention 'mask', 'segmentation', 'DSM', or other internal dataset terms in the claim.
- If this prompt is wrapped by a JSON schema, do NOT emit literal `<Q>`, `<Answer>`, or `<think>` tags; place their content directly into the corresponding JSON fields instead.
- The claim should avoid trivial pixel-level facts and instead require verifiable reasoning. Favor constructions like:
  - "There are at least three small vehicles parked to the north of the large blue-roof building."
  - "No vessel in the visible harbor is currently moored alongside the main pier."
- "At least one cleared patch adjacent to the construction area has sharp soil boundaries and differs clearly from the surrounding vegetation."
- Use precise quantifiers and spatial qualifiers to increase clarity and test reasoning.
- **Mandatory Multi-Step Reasoning**: The single-sentence claim **must** incorporate at least 2-3 combined elements (e.g., quantifiers + descriptive spatial anchors + relational claims) to require verifiable multi-step visual reasoning. Internally, build a chain of 4-6 steps to finalize the claim (e.g., Step 1: Identify key regions with descriptive anchors like 'the empty area in the lower-left corner of the image'; Step 2: Apply quantifiers to visible features within those regions; Step 3: Establish relations or comparisons (e.g., 'adjacent to'); Step 4: compare concrete visible attributes and boundaries; Step 5: Cross-verify cues for ambiguity; Step 6: Ensure the claim is testable and grounded in visible details).
- **Prohibit Trivial Outputs**: Strictly avoid simple pixel-level or descriptive statements without reasoning layers; instead, design claims that appear straightforward but demand step-by-step analysis in  to judge True/False.
- **Structural Verification Requirement**: Every claim must contain a target anchor plus at least one stable reference landmark, and it must require at least one measurable relation or comparison between them.
- **Descriptive Spatial Anchor Requirement**: **Must include precise, descriptive spatial anchors** that combine location with visible features (e.g., 'the empty area in the lower-left corner of the image', 'the cleared patch adjacent to the main structure', 'the dense cluster north of the large building') in the claim to anchor the reasoning to specific, observable areas. Avoid abstract terms like 'lower-left quadrant' without descriptive integration.
- **Claim Length and Richness Requirement**: To ensure the claim is informationally complete and abundant in details, **the single-sentence claim must be at least 20-40 words long**, achieved by logically incorporating nested descriptions, quantifiers, relations, and inferences (e.g., using clauses to elaborate visible cues). Avoid short, incomplete sentences; prioritize richness that supports multi-step verification without redundancy.
- **Deterministic Language Requirement**: **The claim must use definitive, certain language** to form a clear declarative statement that can be judged strictly as True or False. Avoid any words or phrases implying uncertainty, such as 'seems', 'appears', 'suggests', 'possibly', 'might', or equivalents in other languages. Instead, use direct, assertive terms like 'consists of', 'contains', 'demonstrates', 'is used as', 'shows', or 'indicates' only if it conveys certainty based on visible evidence.
- **Label Balance Requirement**: keep the dataset balanced across `True` and `False`, but do not make any individual claim misleading by design. A claim should be judged from localized visible evidence, not from a surface-writing trick or answer-bias rule.
- **Difficulty Assurance**: Validate that the claim is challenging (e.g., surface-level ambiguity resolvable only through detailed RGB evidence), aligning with high-quality reasoning standards without relying on dataset-specific internals.
Final required elements: a brief design rationale, <Q> (single-sentence claim), <Answer> (`True` or `False`), and  (step-by-step justification using only RGB evidence).

Internal verification mapping (do not reveal inside the question): 

"""



def start_common_prompt_twotime( name=""):
    return f"""
{CLAIM_GROUNDING_OVERRIDE}
{CLAIM_DEPTH_OVERRIDE}

You are an expert in crafting intricate visual reasoning challenges
The model will internally inspect two MASK maps and two RGB images corresponding to two timepoints (for verification only). The human solver may see one or both RGB images depending on the task setup.

Objective: generate one high-quality temporal True/False claim in the '{name}' style.
{get_answer_bias_instruction()}

Instructions:
- Provide a brief coarse→fine summary of scene differences between the two timepoints and confirm which RGB frames the solver will see.
- Then write a single-sentence temporal claim that can be judged True/False from the visible RGB frames alone. Example patterns:
  - "Between the earlier and later frame, the number of vehicles in the parking area decreased by at least half."
  - "A new paved surface appeared where dense vegetation was previously present."
- Do NOT mention 'im1', 'im2', 'mask', 'DSM' or other dataset internals in the claim text.
- Prefer precise, testable language using quantifiers and spatial anchors, and avoid speculation about intent or unseen data.
- A temporal claim is only acceptable if it combines a change statement with at least one additional local check tied to a stable landmark or comparison region.
- Forbidden shallow templates include bare statements such as "a new building appeared", "the area remained unchanged", "no new structures appeared", or "the building was demolished" unless the same sentence also requires a second concrete relation, count, boundary, or continuity check.
- Final required elements: design rationale, `<Q>` (one-sentence claim), `<Answer>` (`True` or `False`), and `<think>` (evidence-driven step-by-step justification based only on visible frames).
- If this prompt is wrapped by a JSON schema, do NOT emit literal `<Q>`, `<Answer>`, or `<think>` tags; place their content directly into the corresponding JSON fields instead.

- **Mandatory Multi-Step Reasoning**: The single-sentence claim **must** incorporate at least 2-3 combined elements (e.g., quantifiers + descriptive spatial anchors + relational claims) to require verifiable multi-step visual reasoning. Internally, build a chain of 4-6 steps to finalize the claim (e.g., Step 1: Identify key regions with descriptive anchors like 'the empty area in the lower-left corner of the image'; Step 2: Apply quantifiers to visible features within those regions; Step 3: Establish relations or comparisons (e.g., 'adjacent to'); Step 4: Infer functional aspects based on patterns; Step 5: Cross-verify cues for ambiguity; Step 6: Ensure the claim is testable and grounded in visible details).
- **Prohibit Trivial Outputs**: Strictly avoid simple pixel-level or descriptive statements without reasoning layers; instead, design claims that appear straightforward but demand step-by-step analysis in  to judge True/False.
- **Structural Verification Requirement**: Every claim must contain a target anchor plus at least one stable reference landmark, and it must require at least one measurable relation or comparison between them.
- **Descriptive Spatial Anchor Requirement**: **Must include precise, descriptive spatial anchors** that combine location with visible features (e.g., 'the empty area in the lower-left corner of the image', 'the cleared patch adjacent to the main structure', 'the dense cluster north of the large building') in the claim to anchor the reasoning to specific, observable areas. Avoid abstract terms like 'lower-left quadrant' without descriptive integration.
- **Claim Length and Richness Requirement**: To ensure the claim is informationally complete and abundant in details, **the single-sentence claim must be at least 20-40 words long**, achieved by logically incorporating nested descriptions, quantifiers, relations, and inferences (e.g., using clauses to elaborate visible cues). Avoid short, incomplete sentences; prioritize richness that supports multi-step verification without redundancy.
- **Deterministic Language Requirement**: **The claim must use definitive, certain language** to form a clear declarative statement that can be judged strictly as True or False. Avoid any words or phrases implying uncertainty, such as 'seems', 'appears', 'suggests', 'possibly', 'might', or equivalents in other languages. Instead, use direct, assertive terms like 'consists of', 'contains', 'demonstrates', 'is used as', 'shows', or 'indicates' only if it conveys certainty based on visible evidence.
- **Label Balance Requirement**: keep the dataset balanced across `True` and `False`, but do not make any individual claim misleading by design. A claim should be judged from localized visible evidence, not from a surface-writing trick or answer-bias rule.
- **Difficulty Assurance**: Validate that the claim is challenging (e.g., surface-level ambiguity resolvable only through detailed RGB evidence), aligning with high-quality reasoning standards without relying on dataset-specific internals.
- **Force Temporal Integration**: If the task involves multiple timepoints, **every claim must explicitly incorporate temporal information** (e.g., 'between the earlier and later frame', 'over the one-year interval', 'from the first to the last visible frame') to describe changes, comparisons, or evolutions. Ensure this is woven into the single-sentence claim for verifiable temporal reasoning, without mentioning hidden internals.

Internal verification mapping (for your internal checks only): 

"""


def start_common_prompt_threetime( name=""):
    return f"""
{CLAIM_GROUNDING_OVERRIDE}
{CLAIM_DEPTH_OVERRIDE}

You are an expert in crafting intricate visual reasoning challenges
The model will internally inspect three MASK maps and three RGB images for cross-time verification (for internal use). Human solvers will only see the RGB image(s) specified by the task.

Objective: generate one concise, testable True/False claim about scene evolution in the '{name}' style.
{get_answer_bias_instruction()}

Instructions:
- Start with a short coarse→fine summary of observed evolution across the three frames, confirming which frames are visible to the solver.
- Write one single-sentence claim regarding change or relative comparison across time that can be judged from the visible RGB frames. Example:
- "From the first to the last visible frame, the area previously covered by vegetation has been converted into an impervious surface."
- Do NOT use dataset-internal labels such as 'mask', 'im1', 'im2', or 'DSM' in the claim.
- Use precise quantifiers, spatial anchors, or comparative phrasing to ensure the claim is verifiable.
- For three-time or longer temporal items, the claim must depend on the evolution pattern plus one extra local structural constraint; endpoint difference alone is not enough.
- Forbidden shallow templates include bare statements such as "a new building appeared", "the area stayed unchanged", "no new buildings were added", or "a structure disappeared" unless the sentence also checks attachment, gap filling, boundary crossing, continuity, count, or a nearby comparison region.
- Final output must include: design rationale, `<Q>` (single-sentence claim), `<Answer>` (`True` or `False`), and `<think>` (stepwise justification using only RGB evidence).
- If this prompt is wrapped by a JSON schema, do NOT emit literal `<Q>`, `<Answer>`, or `<think>` tags; place their content directly into the corresponding JSON fields instead.
- The claim should avoid trivial pixel-level facts and instead require verifiable reasoning. Favor constructions like:
- "The empty area in the lower-left corner of the image contains at least three temporary structures adjacent to the organized trailer parking in the upper-right."
- "No cleared patch in the upper-right of the image has denser texture than the adjacent forested area."

- **Mandatory Multi-Step Reasoning**: The single-sentence claim **must** incorporate at least 2-3 combined elements (e.g., quantifiers + descriptive spatial anchors + relational claims) to require verifiable multi-step visual reasoning. Internally, build a chain of 4-6 steps to finalize the claim (e.g., Step 1: Identify key regions with descriptive anchors like 'the empty area in the lower-left corner of the image'; Step 2: Apply quantifiers to visible features within those regions; Step 3: Establish relations or comparisons (e.g., 'adjacent to'); Step 4: Infer functional aspects based on patterns; Step 5: Cross-verify cues for ambiguity; Step 6: Ensure the claim is testable and grounded in visible details).
- **Prohibit Trivial Outputs**: Strictly avoid simple pixel-level or descriptive statements without reasoning layers; instead, design claims that appear straightforward but demand step-by-step analysis in  to judge True/False.
- **Structural Verification Requirement**: Every claim must contain a target anchor plus at least one stable reference landmark, and it must require at least one measurable relation or comparison between them.
- **Descriptive Spatial Anchor Requirement**: **Must include precise, descriptive spatial anchors** that combine location with visible features (e.g., 'the empty area in the lower-left corner of the image', 'the cleared patch adjacent to the main structure', 'the dense cluster north of the large building') in the claim to anchor the reasoning to specific, observable areas. Avoid abstract terms like 'lower-left quadrant' without descriptive integration.
- **Claim Length and Richness Requirement**: To ensure the claim is informationally complete and abundant in details, **the single-sentence claim must be at least 20-40 words long**, achieved by logically incorporating nested descriptions, quantifiers, relations, and inferences (e.g., using clauses to elaborate visible cues). Avoid short, incomplete sentences; prioritize richness that supports multi-step verification without redundancy.
- **Deterministic Language Requirement**: **The claim must use definitive, certain language** to form a clear declarative statement that can be judged strictly as True or False. Avoid any words or phrases implying uncertainty, such as 'seems', 'appears', 'suggests', 'possibly', 'might', or equivalents in other languages. Instead, use direct, assertive terms like 'consists of', 'contains', 'demonstrates', 'is used as', 'shows', or 'indicates' only if it conveys certainty based on visible evidence.
- **Label Balance Requirement**: keep the dataset balanced across `True` and `False`, but do not make any individual claim misleading by design. A claim should be judged from localized visible evidence, not from a surface-writing trick or answer-bias rule.
- **Difficulty Assurance**: Validate that the claim is challenging (e.g., surface-level ambiguity resolvable only through detailed RGB evidence), aligning with high-quality reasoning standards without relying on dataset-specific internals.
- **Force Temporal Integration**: If the task involves multiple timepoints, **every claim must explicitly incorporate temporal information** (e.g., 'between the earlier and later frame', 'over the one-year interval', 'from the first to the last visible frame') to describe changes, comparisons, or evolutions. Ensure this is woven into the single-sentence claim for verifiable temporal reasoning, without mentioning hidden internals.
Reference mapping for internal verification: 

"""


def auto_start_prompt(name, datasets):
    # preserve original dataset branching logic; only the prompt content was adjusted to True/False style
    if datasets in ["xView2", "SECOND"]:
        return start_common_prompt_twotime(name=name)
    elif datasets in ["miniucd", "spaceNet7"]:
        return start_common_prompt_threetime(name=name)
    elif datasets in ["Potsdam", "Vaihingen"]:
        return start_common_prompt_dsm(name=name)
    else:
        return start_common_prompt(name=name)


