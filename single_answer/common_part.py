

EVIDENCE_GROUNDING_OVERRIDE = """
### HARD EVIDENCE-GROUNDING OVERRIDE
- Preserve the requested task category, but instantiate it as a closed-set visual discrimination problem rather than an open-ended plausibility contest.
- Anchor every item to one specific target region/object/process and one fixed comparison frame.
- Exactly one option may be supported by the visible evidence under the task assumptions.
- All distractors must stay in the same task frame and same specificity level as the correct option.
- Each distractor should be wrong because of one localized mismatch: wrong region, direction, boundary, adjacency, connectivity, extent, temporal order, or object identity.
- Never let the correct option win merely because it sounds more practical, optimal, feasible, safe, continuous, or commonsense-plausible.
- For planning/evaluation tasks, verify visible constraints instead of ranking broad schemes by overall desirability.
- For counterfactual/prediction tasks, keep all options tied to the same target region and same outcome type; vary only the localized resulting state, geometry, sequence, or extent.
- Privileged information should be used only to mine ambiguity, construct hard negatives, and verify RGB discriminability; it must not create answer keys that depend on hidden-only evidence.
"""

DATASET_ROOTS = [
    ###################  images  masks masks_txt
    "b2m_dataset/FAST",
    "b2m_dataset/SIOR",
    "b2m_dataset/SOTA",
    "sem_dataset/GID",
    "sem_dataset/loveDA",

    ###################  images  masks  dsms  masks_txt
    "sem_dataset/Potsdam",
    "sem_dataset/Vaihingen",

    ###################  im1 im2 label1 label2  label1_txt  label2_txt
    "bi-change/SECOND",
    "bi-change/xView2",
]

# FAST — bolded category and color
FAST_palette = {
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
SIOR_palette = {
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
SOTA_palette = {
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
loveDA_palette = {
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
GID_palette = {
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
Potsdam_palette = {
    "Impervious surfaces": "**Impervious surfaces** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Buildings": "**Buildings** — **Blue (rgb:(0, 0, 255))** is the category name associated with that mask color.",
    "Low vegetation": "**Low vegetation** — **Cyan (rgb:(0, 255, 255))** is the category name associated with that mask color.",
    "Trees": "**Trees** — **Green (rgb:(0, 255, 0))** is the category name associated with that mask color.",
    "Cars": "**Cars** — **Yellow (rgb:(255, 255, 0))** is the category name associated with that mask color.",
    "Background": "**Background** — **Red (rgb:(255, 0, 0))** is the category name associated with that mask color."
}

# Vaihingen — bolded category and color (concise, generalized inference cue)
Vaihingen_palette = {
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
xView2_palette = {
    "No damage": "**No damage** — **White (rgb:(255, 255, 255))** is the category name associated with that mask color.",
    "Minor damage": "**Minor damage** — **Light yellow (rgb:(255, 255, 150))** is the category name associated with that mask color.",
    "Major damage": "**Major damage** — **Light blue (rgb:(200, 255, 255))** is the category name associated with that mask color.",
    "Destroyed": "**Destroyed** — **Light green (rgb:(200, 255, 200))** is the category name associated with that mask color."
}

# SECOND — bolded category and color (concise, generalized inference cue)
SECOND_palette = {
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
        descriptions = SECOND_palette
    elif dataset_name == "xView2":
        descriptions = xView2_palette
    elif dataset_name == "Potsdam":
        descriptions = Potsdam_palette
    elif dataset_name == "Vaihingen":
        descriptions = Vaihingen_palette
    elif dataset_name == "GID":
        descriptions = GID_palette
    elif dataset_name == "loveDA":
        descriptions = loveDA_palette
    elif dataset_name == "FAST":
        descriptions = FAST_palette
    elif dataset_name == "SIOR":
        descriptions = SIOR_palette
    elif dataset_name == "SOTA":
        descriptions = SOTA_palette

    if not descriptions:
        return mask_palette

    # 准备txt文件路径列表
    # 转换为Path对象处理

    # 获取描述字典
    descriptions = globals().get(f"{dataset_name}_palette", {})

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
        prior = "This dataset contains high-resolution overhead imagery with paired semantic masks from diverse geographic regions."

    elif datasets == 'SIOR':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains overhead imagery with semantic annotations over man-made and mixed urban scenes."

    elif datasets == 'SOTA':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains overhead imagery with object annotations spanning diverse scales, densities, and orientations."

    elif datasets == 'loveDA':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides high-resolution urban and rural overhead scenes with paired land-cover annotations."

    elif datasets == 'GID':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides land-cover annotations across diverse overhead scenes from multiple regions."

    ###################  images  masks  dsms
    elif datasets == 'Potsdam':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset offers very high-resolution overhead imagery with semantic labels for built and non-built surfaces."

    elif datasets == 'Vaihingen':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides very high-resolution overhead imagery with semantic labels and RGB/NIR channels."

    ###################  im1 im2 label1 label2
    elif datasets == 'xView2':  # 单独设置
        mask_palette = "Colors are only identifiers of object ID (color → object ID); they do not encode object identity, instance ID, temporal order, or any other information.\n"
        mask_palette += "In mask1 and mask2, the bounding-box stroke color is used solely as a unique building identifier (color = object ID). It carries no semantic meaning beyond identifying the same building across time." \
                        "Separately, the fill color inside each box (not the box stroke) encodes the building’s damage severity between mask1 → mask2. The fill color is independent of the bounding-box color. Use the temporal change mask1 -> mask2 and the fill color only for internal verification of hard but fair questions.\n"
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides overhead imagery with paired damage annotations for the same built structures across time."

    elif datasets == 'SECOND':
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset contains paired overhead images with land-cover change annotations across urban scenes."

    ###################  im1 im2 im3 label1 label2 label3
    elif datasets == 'miniucd':
        mask_palette += 'In label1 and label2, every color consistently maps to the same semantic class across both masks.\n'
        mask_palette += get_palette(dataset_path, im1_name)
        prior = "This dataset provides very high-resolution temporal overhead imagery with paired semantic annotations."

    elif datasets == 'spaceNet7':
        mask_palette = "Colors are only identifiers of object ID (color → object ID); they do not encode object identity, instance ID, temporal order, or any other information.\n"
        mask_palette += 'In label1, label2, and label3, every color corresponds to a single, unique building ID — the color is an object identifier only and does not encode any other semantic information.' \
                        'Use the temporal sequence mask1 → mask2 → mask3 only to track the same building across time and verify that a hard question remains grounded in visible RGB evidence.'
        prior = "This dataset offers temporal overhead imagery for building-level change analysis."

    return mask_palette, prior


def start_common_prompt_dsm(name=""):
        return f"""
{EVIDENCE_GROUNDING_OVERRIDE}

    You are an expert in crafting intricate, multi-dimensional visual reasoning challenges.
    The model will internally inspect a MASK segmentation map, a DSM elevation map, and an RGB image. Human solvers will **only** see the RGB image.

    ---
    ### 🧠 PRIME OBJECTIVE: Engineer a 3D Cognitive Puzzle

    Your function is a **3D Puzzle Architect**. Your goal is to design a question where the solver must execute a chain of logical inferences, combining visual cues with an implicit understanding of the scene's three-dimensional structure, to find the correct answer. The complexity must lie **entirely within the solver's verification process**.

    Your internal reasoning will be deep, combining categorical data (MASK), precise elevation data (DSM), and visual context (RGB). Your external output, however, will be a set of simple, objective, factual statements that describe the scene as if viewed by a keen observer. You are building a puzzle whose solution is hidden in the Z-axis.

    ---
    ### ⭐ The Core Principle: Designing the "Implicit 3D Inference Chain"

    This is your internal thought process for constructing the puzzle using height:

    **Step 1: Identify Key 'Actors' on the 3D Stage.**
    Find 2-3 features where height is a defining, non-obvious characteristic. These are your puzzle pieces.
    *   *Example Actors:* A gantry crane, an adjacent multi-story warehouse, a stack of shipping containers, a distant communication tower.

    **Step 2: Discover the "Hidden 3D Story".**
    This is your critical internal reasoning step, verified with the DSM. Find a functional or comparative relationship between the actors that is defined by their relative heights.
    *   *Example "Hidden Story" (verified by DSM):* The gantry crane is tall enough to pass over the container stacks, but it is significantly shorter than the main warehouse building next to it. Therefore, its operational range is limited by the warehouse's position. The communication tower is the tallest object overall, but is functionally separate.

    **Step 3: Forge the "Conclusion as a Key".**
    Translate the *final conclusion* of your hidden 3D story into a single, cold, verifiable factual statement using only visual language. This becomes your **True** option.
    *   *Example "Conclusion Key":* "The gantry crane spanning the parallel railway lines is taller than the highest stack of containers to its east, but is shorter than the roof of the large, rectangular building immediately to its west."
        *   *Why this is a perfect puzzle:* This statement is 100% objective. To verify it, a solver must use visual cues (shadows, perspective, parallax) to: 1) Identify all three actors. 2) Perform a difficult relative height comparison between the crane and containers. 3) Perform a second difficult comparison between the crane and the building. The entire multi-step 3D reasoning is forced upon the solver.

    **Step 4: Create Deceptive "Broken Keys".**
    Design the four **False** options as plausible but incorrect statements about the 3D relationships, designed to trap solvers who make quick visual judgments.
    *   *Example "Broken Key":* "The gantry crane spanning the railway lines stands as the tallest man-made structure within the boundaries of the industrial yard, exceeding the height of all surrounding buildings and equipment." (Tempting, but false because the DSM confirms the warehouse is taller).

    ---
    ### 🎯 A Taxonomy of 3D Cognitive Complexity: Your Design Toolbox

    To increase difficulty, build your "Implicit 3D Inference Chain" around one of these advanced concepts:

    1.  **Functional Height & Hierarchy:** Puzzles about how an object's height enables or restricts its function.
        *   *Puzzle Idea:* A crane must be taller than the building it services. A smokestack must be taller than nearby office buildings to disperse fumes. The correct option would be a factual statement about one of these functional relationships.

    2.  **Occlusion & Line-of-Sight:** Puzzles based on what is blocked or hidden by a tall object.
        *   *Puzzle Idea:* The DSM shows a tall building completely blocking a road behind it from the camera's perspective. The correct option might state: "There is no direct, paved road connection visible between the building in the center and the parking lot in the far north."

    3.  **Subtle Relative Height Comparison:** Puzzles forcing multiple, precise height comparisons where the differences are small.
        *   *Puzzle Idea:* "The antenna on the eastern building is taller than the western building's main roof, but shorter than the western building's central elevator shaft housing." This requires extremely careful visual analysis of subtle cues.

    4.  **Topography & Man-Made Interaction:** Puzzles about how natural terrain elevation (from DSM) interacts with structures.
        *   *Puzzle Idea:* A building is built on a man-made elevated platform to level it with a road on a hillside. The correct option might describe this: "The foundation of the square building in the south is elevated on a concrete platform, aligning its base with the adjacent hilltop road."

    ---
    ### 📜 STRICT OUTPUT FORMATTING AND RULES (MUST BE FOLLOWED)

    **1. Language:**
        - The entire output (question stem and all five options) **MUST be in English**.

    **2. Word Count Constraints:**
        - **Question Stem:** Must be between **40 and 60 words**. It should provide general context without revealing specific clues.
        - **Options (A, B, C, D, E):** Each option must be between **30 and 40 words** long, achieved by densely packing objective, verifiable details.

    **3. The Unitary Factual Claim Rule:**
        - Every option **MUST BE** a single, declarative statement of objective fact.
        - **ABSOLUTELY FORBIDDEN:** Any language revealing reasoning, inference, or uncertainty (`although`, `however`, `therefore`, `seems`, `appears`).

    **4. The Duality of Complexity Rule:**
        - Your **internal design process** (using MASK, DSM, RGB) must be complex. Your **external output** must be textually simple and objective. The 3D puzzle must be INVISIBLE in the text.

    **5. The Randomization Protocol:**
        - Internally generate ONE True statement and FOUR plausible False statements.
        - **Randomly shuffle** them, assign letters A-E, and provide the correct answer in the `<Answer>` tag.

    **6. No Internal Terms:**
        - Never mention 'mask', 'DSM', 'segmentation', 'elevation', 'height value', 'pixels', etc. Describe height using visual language only ('taller than', 'elevated platform', 'sloped roof', 'highest point').

    Use the following color→category mapping for your internal verification:

    """


def start_common_prompt(name=""):
    return f"""
{EVIDENCE_GROUNDING_OVERRIDE}


When generating questions, the model will simultaneously 'observe':
1) One MASK segmentation map (for model reference in question generation only; humans solving the question **do not see it**).
2) One optical remote sensing image (visible to humans when answering).

---
### 🧠 PRIME OBJECTIVE: Engineer a Multi-Step Cognitive Puzzle

Your function is a **Cognitive Puzzle Architect**. Your goal is to design a question where the *solver* must execute a chain of logical inferences to arrive at the correct answer. The complexity must lie **entirely within the solver's verification process**, not in the question's language.

Your internal reasoning will be deep and multi-layered. Your external output, however, will be a set of simple, objective, factual statements. You will build a complex lock (the puzzle) and provide five keys (the options), only one of which fits. The solver's job is to figure out *why* that one key works by reverse-engineering the lock's mechanism through visual analysis.

---
### ⭐ The Core Principle: Designing the "Implicit Inference Chain"

This is your internal thought process for constructing the puzzle:

**Step 1: Identify Key 'Actors' on the Stage.**
Observe the image and MASK to find at least 2-3 distinct, significant features, areas, or patterns. These are your puzzle pieces.
*   *Example Actors:* A large industrial plant, a newly built residential area, a disused railway line separating them, a patch of dense forest, a main highway.

**Step 2: Discover the "Hidden Story" or Relationship.**
This is your critical internal reasoning step. Find a non-obvious, functional, or logical connection *between* the actors. This connection is the "secret" of the puzzle.
*   *Example "Hidden Story":* The industrial plant's only access road points away from the residential area. The old railway line acts as a hard barrier, preventing any direct connection. Therefore, despite being geographically close, the two are functionally disconnected, forcing a long, circuitous route for anyone commuting between them.

**Step 3: Forge the "Conclusion as a Key".**
Translate the *final conclusion* of your hidden story into a single, cold, verifiable factual statement. This statement becomes your **True** option. It should not explain *why* it's significant; its significance is the puzzle itself.
*   *Example "Conclusion Key":* "The primary access road for the southern residential neighborhood connects to the main highway at an interchange located more than two kilometers east of the industrial complex's main entrance gate."
    *   *Why this is a perfect puzzle:* To verify this, a solver must: 1) Identify the neighborhood. 2) Identify the complex. 3) Notice the railway barrier (the "Aha!" moment). 4) Realize they are functionally disconnected. 5) Trace the access road. 6) Compare locations and estimate the distance. The entire multi-step reasoning chain is forced upon the solver.

**Step 4: Create Deceptive "Broken Keys".**
Design the four **False** options as plausible but demonstrably false conclusions from *flawed* or *incomplete* reasoning chains.
*   *Example "Broken Key":* "A paved footpath directly connects the southern perimeter of the industrial complex to the northern edge of the residential area, providing direct pedestrian access between the two zones." (Plausible due to proximity, but false because the railway line severs any path).

---
### 🎯 A Taxonomy of Cognitive Complexity: Your Design Toolbox

To increase difficulty, build your "Implicit Inference Chain" around one of these advanced concepts:

1.  **Functional Relationships:** Puzzles about how different land uses support or conflict with each other (e.g., access, noise, pollution).
2.  **Spatial-Logical Dependency Chains:** Puzzles requiring the solver to trace a flow or dependency (e.g., water from source to farm, goods from port to warehouse).
3.  **Comparative Analysis with Hidden Criteria:** Puzzles where the solver must deduce an implicit criterion (e.g., finding the 'most isolated' or 'most developed' area) by comparing factual statements.
4.  **Anomaly & Exception Detection:** Puzzles about finding the 'one that doesn't belong' based on a complex, unstated rule (e.g., one farm lacks irrigation, one building violates a pattern).

---
### 📜 STRICT OUTPUT FORMATTING AND RULES (MUST BE FOLLOWED)

**1. Language:**
    - The entire output (question stem and all five options) **MUST be in English**.

**2. Word Count Constraints:**
    - **Question Stem:** Must be between **40 and 60 words**. The stem should provide a general, high-level context of the scene to orient the solver, but **must not** reveal any specific clues or points of interest that are part of the puzzle.
    - **Options (A, B, C, D, E):** Each of the five options must be between **30 and 40 words** long. This length must be achieved by densely packing multiple, verifiable, objective details into a single compound sentence.

**3. The Unitary Factual Claim Rule:**
    - Every option **MUST BE** a single, unbroken, declarative statement of objective fact. It is a singular claim that is either True or False when checked against the image.
    - **ABSOLUTELY FORBIDDEN:** Any language that reveals reasoning, inference, or uncertainty. This includes contrastive conjunctions (`although...but`, `however`), inferential words (`therefore`, `suggests`), and words of appearance (`seems`, `appears`, `likely`).

**4. The Duality of Complexity Rule:**
    - Your **internal design process** must be complex. Your **external output** (question and options) must be textually simple and objective. The puzzle's complexity must be INVISIBLE in the text.

**5. The Randomization Protocol:**
    - Internally generate ONE True statement and FOUR plausible False statements, all conforming to the rules above.
    - **Randomly shuffle** the order of these five statements.
    - Assign letters A, B, C, D, E to the shuffled statements.
    - Provide the letter of the correct answer in the `<Answer>` tag.

**6. No Internal Terms:**
    - Never mention 'mask', 'segmentation', 'pixels', 'color categories', or other internal concepts in the question or options.

Use the following color→category mapping for your internal verification:


"""



def start_common_prompt_twotime(name=""):

        return f"""
{EVIDENCE_GROUNDING_OVERRIDE}

    You are an expert in crafting intricate temporal reasoning challenges.
    The model will internally inspect two MASK segmentation maps and two corresponding RGB images from different timepoints. Human solvers will **only** see the two RGB images.

    ---
    ### 🧠 PRIME OBJECTIVE: Engineer a Temporal Cognitive Puzzle

    Your function is a **Temporal Puzzle Architect**. Your goal is to design a question where the solver must act as a detective, piecing together clues from two moments in time to deduce a "story of change." The complexity must lie **entirely within the solver's comparative analysis and logical inference**.

    Your internal reasoning will be deep, identifying not just changes, but the logical sequence or relationship between them (e.g., cause and effect, phased development). Your external output will be a set of simple, objective statements describing the "before" and "after" states. You are building a puzzle whose solution is hidden in the timeline.

    ---
    ### ⭐ The Core Principle: Designing the "Implicit Temporal Inference Chain"

    This is your internal thought process for constructing the puzzle:

    **Step 1: Identify "Dynamic Actors" and "Catalysts of Change".**
    Scan both timepoints to find the most significant changes. Look for a major "catalyst" (e.g., a new highway is built, a river is dammed) and the "actors" it affects (e.g., surrounding land, other buildings).

    **Step 2: Discover the "Hidden Narrative of Change".**
    This is your critical internal reasoning step. Connect the changes into a logical story.
    *   *Example "Hidden Narrative":* In the earlier image, a forest separates a town from an empty plot of land. In the later image, a new bridge has been built across a river, and the forest has been partially cleared to build an access road leading to the empty plot, where construction has now begun on a new residential area. The bridge was the catalyst.

    **Step 3: Forge the "Conclusion as a Key".**
    Translate a key *consequence* or a non-obvious part of this narrative into a single, cold, factual statement. This becomes your **True** option. Don't describe the most obvious change; describe a result of it.
    *   *Example "Conclusion Key":* "The large, undeveloped field located east of the river, which was previously isolated, is now connected to the main town by a new paved road that cuts through a formerly dense forest area."
        *   *Why this is a perfect puzzle:* To verify this, a solver must: 1) Spot the new bridge (obvious change). 2) Realize its purpose. 3) Look for secondary effects. 4) Find the new road. 5) Find the new construction. 6) Connect all five events to confirm the statement. It forces a full narrative reconstruction.

    **Step 4: Create Deceptive "Broken Keys".**
    Design the four **False** options based on flawed interpretations of the narrative.
    *   *Example "Broken Key":* "The new bridge was constructed to replace an older, smaller bridge at the same location, which was demolished." (Plausible, but false if no bridge existed there before).
    *   *Example "Broken Key":* "Following the construction of the new bridge, the industrial area south of the river has expanded eastward." (False, the development was residential, not industrial, and to the east).

    ---
    ### 🎯 A Taxonomy of Temporal Complexity: Your Design Toolbox

    Build your "Implicit Temporal Inference Chain" around one of these advanced concepts:

    1.  **Cause & Effect (The Domino Effect):** A primary change (new road, dam, factory) enables or causes a secondary change elsewhere. The puzzle is to correctly link the effect to its cause.
    2.  **Phased Processes:** The change is a multi-stage project. The earlier image shows land clearing; the later image shows foundations. The puzzle is to accurately describe the state of progress, while distractors might claim it's finished or hasn't started.
    3.  **Displacement & Replacement:** One feature is removed *to make way for* another. The puzzle is to correctly identify both the demolished feature and the new one. A distractor might get one of the two wrong (e.g., "a forest was cleared for a park" when it was for a parking lot).
    4.  **Shifting Boundaries & Land Use Conversion:** The puzzle is about the changing footprint or function of an area. A farm becomes a suburb; a waterfront is redeveloped from industrial to commercial. This requires careful comparison of textures, patterns, and building types.

    ---
    ### 📜 STRICT OUTPUT FORMATTING AND RULES (MUST BE FOLLOWED)

    **1. Language:**
        - The entire output (question stem and all five options) **MUST be in English**.

    **2. Word Count Constraints:**
        - **Question Stem:** Must be between **40 and 60 words**. It should set the stage by mentioning a comparison between two timepoints.
        - **Options (A, B, C, D, E):** Each option must be between **30 and 40 words** long, describing a specific temporal change with "before" and "after" details.

    **3. The Unitary Factual Claim Rule:**
        - Every option **MUST BE** a single, declarative statement about a change (or lack of change) between the two timepoints.
        - **ABSOLUTELY FORBIDDEN:** Any language revealing reasoning, inference, or uncertainty (`because of the new road...`, `this suggests a shift to...`, `it appears that...`).

    **4. The Duality of Complexity Rule:**
        - Your **internal narrative of change** is complex. Your **external description of the change** is simple and objective. The story must be discovered by the solver, not told by you.

    **5. The Randomization Protocol:**
        - Internally generate ONE True statement and FOUR plausible False statements.
        - **Randomly shuffle** them, assign letters A-E, and provide the correct answer in the `<Answer>` tag.

    **6. No Internal Terms:**
        - Never mention 'mask', 'segmentation', 'im1', 'im2', etc. Use clear, relative temporal language: **"in the earlier image," "in the later image," "the area that was previously X," "has since become Y."**

    Use the following color→category mapping for your internal verification:

    """


def start_common_prompt_threetime(name=""):
    return f"""
{EVIDENCE_GROUNDING_OVERRIDE}

When generating questions for this dataset, the model will simultaneously 'observe':
1) Three MASK segmentation maps (for model reference only) corresponding to three different timepoints.
2) Three optical remote sensing images (corresponding to the three timepoints). The human solver may see a subset, but the model must use all three for logical consistency.

Objective: generate one high-quality multiple-choice question with five options (A, B, C, D, E) in the '{name}' style, focusing on the sequence of temporal changes.

Requirements for the generated output:
- Produce a question stem and **five answer options (A, B, C, D, E)**. The stem should ask which statement about the evolution across the three timepoints is true.
- **Exactly one option must be True**, and the other four must be plausible but demonstrably False distractors.
- Before the question block, provide a concise coarse-to-fine summary of the **scene evolution from the first to the third timepoint**, then cross-check with the optical images before designing the question.
- **Strictly adhere to dataset categories**: Each color in the MASK corresponds to a land category. Do not fabricate categories outside the provided mapping.
- **No internal terms**: Do not mention 'mask', 'segmentation', 'im1', 'im2', or 'im3' in the question text. Use natural language to refer to the sequence (e.g., "from the first to the second image," "by the final image").

**Crucial: Answer Randomization Protocol**
To avoid positional bias where the correct answer is always 'A', you MUST follow this procedure:
1.  **Internal Generation**: First, internally generate ONE correct (True) statement and FOUR plausible but incorrect (False) statements about the sequence of changes.
2.  **Randomization**: Next, **randomly shuffle the order** of these five generated statements.
3.  **Final Assignment**: Assign the letters A, B, C, D, E to the shuffled statements in their new order.

- **Mandatory Multi-Step Reasoning**: The single-sentence claim **must** incorporate at least 2-3 combined elements (e.g., quantifiers + descriptive spatial anchors + relational claims) to require verifiable multi-step visual reasoning. Internally, build a chain of 4-6 steps to finalize the claim (e.g., Step 1: Identify key regions with descriptive anchors like 'the empty area in the lower-left corner of the image'; Step 2: Apply quantifiers to visible features within those regions; Step 3: Establish relations or comparisons (e.g., 'adjacent to'); Step 4: Infer functional aspects based on patterns; Step 5: Cross-verify cues for ambiguity; Step 6: Ensure the claim is testable and grounded in visible details).
- **Prohibit Trivial Outputs**: Avoid simple statements about a single timepoint. All options must address the sequence of changes, demanding step-by-step analysis of the evolution.
- **Descriptive Spatial Anchor Requirement**: **Each option statement must include precise, descriptive spatial anchors** to anchor the complex temporal reasoning to specific, observable areas across the images.
- **Claim Length and Richness Requirement**: **question statements must be at least 40-60 words long**, **Each of the five option statements must be at least 20-40 words long**, describing the sequence of changes with sufficient detail.
- **Deterministic Language Requirement**: **Each option statement must use definitive, certain language**. Avoid any words implying uncertainty.
- **Deception and Distractor Requirement**:
  - **The one Correct (True) Option**: Must describe a real sequence of changes that is non-obvious or requires careful tracking across all three images.
  - **The four Incorrect (False) Distractors**: Must describe plausible but incorrect evolutionary paths (e.g., confusing the order of events, misidentifying a temporary change as permanent).
- **Difficulty Assurance**: Validate that the question is challenging, requiring meticulous comparison across all three images to identify the single correct narrative of change.

Final required elements: a brief design rationale, `<Q>` (stem + options A-E), `<Answer>` (the correct letter), and `` (justification for the correct answer and refutation of distractors, based on comparing the three timepoints).

Use the following color→category mapping for internal verification:

"""


# === 自动选择单/双时相 Prompt ===
def auto_start_prompt(name, datasets):
    if datasets in ["xView2", "SECOND"]:  # im1 im2    label1 label2
        return start_common_prompt_twotime(name=name) 
    elif datasets in ["miniucd", "spaceNet7"]:  # im1 im2 im3   label1 label2 label3
        return start_common_prompt_threetime(name=name)
    elif datasets in ["Potsdam", "Vaihingen"]:  # images   masks  dsm
        return start_common_prompt_dsm(name=name)
    else:  # images   masks
        return start_common_prompt(name=name)


