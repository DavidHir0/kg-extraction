# Universal Neural Architecture Graph Schema (v5.8)

**Metadata Requirement:** Every graph must contain `architecture_name` (String) and `extraction_version` ("5.8").

---

## 1. Global Node Keys (Required on all nodes)

| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique UUID for the node. |
| `class` | String | The primary functional category (e.g., "LAYER"). |
| `sub_type` | String | The specific algorithm or type (e.g., "convolution"). |
| `raw_text` | String | Exact visual text extracted, preserving original LaTeX. **Use an empty string `""` (never `null`) if no text is visually present.** |
| `properties` | Dictionary | Nested object containing specific attributes. **Strictly optional: Keep as `{}` unless the visual text provides detailed algorithms, dimensions, or mechanisms.** |

---

## 2. The `properties` Dictionary Taxonomy

**Global Properties (Available to EVERY class/subtype):**
* `name` (String): Explicit human-readable label (e.g., "Stem Conv", "Projection").
* `symbol` (String): Mathematical variable or notation (e.g., "W_g", "alpha").
* `note` (String): Catch-all for visual caveats, asterisks, or diagram legends.
* **`sequence` (Array of Objects):** CRITICAL for fused nodes. A list of sub-nodes (each containing `class`, `sub_type`, and their own nested `properties`) defining an ordered chain of operations combined in a single visual block.

### Class 1: DATA (Topological state, dimensions, or data modality)
*Note: All DATA subtypes share the same allowed properties.*

| Subtypes | Allowed Properties | Data Types |
| :--- | :--- | :--- |
| `input`, `output`, `intermediate`, `constant` | `dimensions`, `dtype`, `modality` | Array, String, String |

### Class 2: LAYER (Trainable core feature transformations)

| Subtype | Allowed Properties | Data Types |
| :--- | :--- | :--- |
| `convolution` | `kernel`, `channels`, `stride`, `dilation`, `groups` | Array, Int/String, Int/Array, Int/Array, Int |
| `transposed_convolution` | `kernel`, `channels`, `stride` | Array, Int/String, Int/Array |
| `linear` | `units`, `features` | Int/String, Int/String |
| `attention` | `heads`, `dim` | Int, Int/String |
| `recurrent` | `hidden_size`, `direction` | Int/String, String |
| `embedding` | `vocab_size`, `dim` | Int/String, Int/String |
| `custom` | `algorithm` | String |

### Class 3: MODIFIER (Stateless, sequential 1-to-1 operations)

| Subtype | Allowed Properties | Data Types |
| :--- | :--- | :--- |
| `activation` | `algorithm` | String (e.g., "ReLU", "GELU") |
| `normalization` | `algorithm` | String (e.g., "BatchNorm", "LayerNorm") |
| `downsample` | `algorithm`, `kernel`, `stride` | String, Array, Int/Array |
| `upsample` | `algorithm` | String (e.g., "Bilinear") |
| `reshape` | `algorithm` | String (e.g., "Flatten", "Transpose") |
| `dropout` | `rate` | Float (e.g., 0.5) |
| `mask` | `algorithm` | String (e.g., "Causal") |
| `element_wise` | `algorithm` | String (e.g., "Exp", "Log") |
| `positional_encoding` | `algorithm` | String (e.g., "Sinusoidal", "RoPE") |
| `loss` | `algorithm` | String (e.g., "CrossEntropy", "MSE") |
| `custom` | `algorithm` | String |

### Class 4: JUNCTION (Explicit topological N-to-1 routing logic)

| Subtype | Allowed Properties | Data Types |
| :--- | :--- | :--- |
| `add` | *None* | N/A |
| `average` | *None* | N/A |
| `multiply` | `mechanism` | String (e.g., "MatMul", "Hadamard") |
| `concat` | `dim` | String (e.g., "channel", "spatial") |
| `gating` | `mechanism` | String (e.g., "GLU", "Highway") |

### Class 5: MACRO (Named abstractions / sequential black boxes)

| Subtype | Allowed Properties | Data Types |
| :--- | :--- | :--- |
| `block` | `repetition`, `derived_from_definition` | Int/String (e.g., 6, "N"), String (target group ID) |
| `network` | *None* | N/A |

<!-- ### Class 6: SEQUENCE(Named abstractions / sequential black boxes) -->

<!-- | Subtype | Allowed Properties | Data Types | -->
<!-- | :--- | :--- | :--- | -->
<!-- | `block` | `repetition`, `derived_from_definition` | Int/String (e.g., 6, "N"), String (target group ID) | -->
<!-- | `network` | *None* | N/A | -->

---

## 3. Edge Taxonomy
Edges strictly define routing topology and do not transform data. Every edge object **MUST** contain four root-level keys to pass schema guardrails.

| Key | Type | Description |
| :--- | :--- | :--- |
| `source` | String | The originating structural `node_id`. |
| `target` | String | The destination structural `node_id`. |
| `role` | String | **CRITICAL Validation Requirement.** Must be exactly one of: `forward`, `skip`, `condition`, or `backward`. |
| `properties` | Dictionary | Nested object containing extra edge attributes. **MUST exist even if empty `{}`.** |

**`backward` role:** for arrows drawn AGAINST the main data flow: recurrent state loops (e.g. an SSM/RNN hidden state fed back into its own block) and gradient/feedback arrows (e.g. Grad-CAM backpropagation). The graph MUST be acyclic over `forward`/`skip`/`condition` edges; any drawn cycle must pass through a `backward` edge.

### Edge `properties` Dictionary Keys

| Key | Type | Description / Constraints |
| :--- | :--- | :--- |
| `labels` | Array | Optional list of strings representing characters/text intercepted along the edge path (e.g., `["Q", "K", "V"]`). |

*Example Valid Structure:*
```json
{
  "source": "node_uuid_1",
  "target": "node_uuid_2",
  "role": "skip",
  "properties": {
    "labels": ["Identity Connection"]
  }
}
```

---

## 4. Group Taxonomy
Used to map visual boundaries and subgraph definitions without nesting the mathematical nodes.

| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique group ID. |
| `type` | String | Must be: `container` (main flow) or `subgraph_definition` (zoomed-in dictionary). |
| `label` | String | Human-readable name (e.g., "Encoder Section", "N \times"). |
| `boundary_type` | String | Must be: `explicit_box` (drawn lines) or `implicit_header` (floating text over cluster). |
| `member_node_ids` | Array | List of string `node_ids` that belong inside this group. |

---

## 5. The Golden Rules (Extraction Mandates)

1. **The Prime Directive:** Extract visual truth. Do not hallucinate implied math or missing layers.
2. **The Merging Mandate:** Merge dimensions or symbols floating next to an arrow/box into a single `DATA` node. No standalone floating math.
3. **The Composite Sequence Rule:** If a visual node contains multiple operations (e.g., "Conv + BN + ReLU", "Add & Norm", "Conv 64x64"), do NOT break it into multiple root nodes. Output a single root node and define the sub-steps in the `properties.sequence` array.
4. **The Visual Bypass Rule:** Purge empty boxes (no text) with exactly 1 incoming and 1 outgoing arrow. Route the edge directly through.
5. **The Invisible Box Rule:** Text operations floating on arrows (e.g., "Softmax") must be extracted as nodes. Break the edge to route through them.
6. **Literal Dimensions:** Do not calculate missing dimensions. Extract exactly what is written.
7. **LaTeX Stripping Rule:** Preserve LaTeX syntax in `raw_text`, but critically strip it out when populating arrays/strings in the `properties` dictionary.
8. **The Implicit Merge Rule:** When two or more arrows MERGE into one data stream without a drawn merge symbol (a residual arrow joining the main line, skip + upsampled features entering a decoder conv, position embeddings joining tokens), materialize the merge as a JUNCTION node with `raw_text: ""` and `properties.note: "implicit"`, typed by its semantics (`add` for residuals/embedding sums, `concat` for U-Net-style channel merges, ...). Do NOT create junctions for multi-input OPERATORS whose inputs play distinct roles (Q/K/V into attention, gating signal + features into an attention gate, modulation scalars into a block, weight-shared blocks applied to parallel streams): those keep their multiple incoming edges directly. LIKEWISE keep direct edges for dense fan-in: when several arrows each terminate SEPARATELY at the target box (DenseNet-style all-to-all connectivity), that is multi-input wiring, not a merge -- materialize an implicit JUNCTION only where the lines visibly JOIN INTO ONE ARROW before reaching the target. A figure full of plain fan-ins must produce ZERO implicit junctions.
9. **The Backward Edge Rule:** Arrows drawn against the main flow (recurrence loops, gradient/feedback paths) take `role: "backward"`. Ignoring `backward` edges, the graph must be a DAG.
10. **The Boundary Rule (I/O sub_types):** A connected `DATA` node where the depicted flow begins (no incoming edges) is `sub_type: "input"` — or `"constant"` if it is a learned/fixed parameter (positional encodings, class tokens, external memory matrices, fixed masks, scalar factors). A connected `DATA` node where the flow ends (no outgoing edges, ignoring `backward`) is `sub_type: "output"`. This applies even when the figure does not literally say "input"/"output", and applies per figure (module diagrams included). Exceptions stay `intermediate`: recurrence/feedback states that are sinks only because `backward` edges are ignored, and disconnected annotation labels or legend entries. If the flow starts or ends with a bare arrow and NO drawn data box, materialize the missing endpoint as a `DATA` node with `raw_text: ""` and `properties.note: "implicit"` (like Rule 8). One drawn data element = one node: a name/symbol and its nearby dimension label are the SAME node — put the sizes in `properties.dimensions`, never a second `DATA` node.
11. **The Feature-Map Label Rule:** A box or label whose text is ONLY channel counts and/or spatial sizes (e.g. "64" over "570 x 570", "224x224", "$284^2$", "512") is a `DATA` feature map — `sub_type: "intermediate"` unless Rule 10 makes it input/output — with `properties.dimensions` filled as `[channels, height, width]` from the visible numbers (`$284^2$` means 284 x 284; a lone "512" over a box is `[512]`). It is NEVER an operation (`LAYER`/`MODIFIER`) and never a `properties.sequence` composite: in such figures the operations are the arrows/blocks BETWEEN the feature maps (Rule 5). This is the encoder-decoder convention (U-Net-style): bars/boxes = data tensors, arrows = ops. The arrow-operations themselves DO become nodes (the raw extraction materializes each legend-defined arrow instance, e.g. "conv 3x3, ReLU" riding a blue arrow): type them as the LAYER/MODIFIER they name — they are operations, never DATA, and must not be dropped.
12. **Named-Block and Merge-Symbol Disambiguation:** A named processing block drawn inline in the flow ("Attention Gate", "Memory Unit", "RRCU") is a `LAYER` (`sub_type: "custom"` when no standard sub_type fits) — reserve `MACRO/block` for references to a structure defined elsewhere (a `subgraph_definition` group, a legend entry, or an "xN" repetition). For circled merge symbols, type by semantics, not glyph: a symbol merging channels (U-Net skip joining a decoder path) is `JUNCTION/concat` even when drawn as $\oplus$; $\oplus$ summing residual/parallel streams is `JUNCTION/add`; $\otimes$ combining TWO data streams is `JUNCTION/multiply` — `MODIFIER/element_wise` is only for a math op applied to a SINGLE stream. Consult the figure's legend before assigning.

