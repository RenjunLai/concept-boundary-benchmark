from __future__ import annotations

from .constants import RENDERING_TEMPLATE_VERSION
from .io_utils import stable_hash
from .schemas import RenderedRequest


def _labelled_surface(prefix: str, index: int, obj: dict) -> str:
    return f"{prefix}{index} {obj['surface']}"


def _concept_lines(objects: list[dict], prefix: str, start: int = 1) -> list[str]:
    return [f"- {_labelled_surface(prefix, index, obj)}" for index, obj in enumerate(objects, start)]


def _concept_note_lines(objects: list[dict], prefix: str) -> list[str]:
    lines = []
    for index, obj in enumerate(objects, 1):
        gloss = obj.get("gloss") or ""
        if gloss:
            lines.append(f"- {_labelled_surface(prefix, index, obj)}: {gloss}")
    return lines


def _candidate_lines(objects: list[dict]) -> list[str]:
    return [f"{index}. {obj['surface']}" for index, obj in enumerate(objects, 1)]


def _candidate_note_lines(objects: list[dict], single_candidate: bool) -> list[str]:
    lines = []
    for index, obj in enumerate(objects, 1):
        gloss = obj.get("gloss") or ""
        if not gloss:
            continue
        label = "Candidate" if single_candidate else f"{index}. {obj['surface']}"
        lines.append(f"- {label}: {gloss}" if not single_candidate else f"- {label} {obj['surface']}: {gloss}")
    return lines


def _append_wordnet_condition(lines: list[str], a1_obj: dict, branch_objs: list[dict], b_objs: list[dict]) -> None:
    lines.extend(["Target condition:", f"- A1 {a1_obj['surface']} is the outer required concept."])
    if branch_objs:
        lines.append("- The target object must also be a kind of at least one concept below:")
        lines.extend(_concept_lines(branch_objs, "A", start=2))
    if b_objs:
        lines.append("- Exclude objects that are a kind of any concept below:")
        lines.extend(_concept_lines(b_objs, "B"))
    else:
        lines.append("- There are no excluded concepts.")


def _append_babelnet_condition(lines: list[str], a1_obj: dict, branch_objs: list[dict], b_objs: list[dict]) -> None:
    lines.extend(["目标条件：", f"- A1 {a1_obj['surface']} 是外层必须满足的概念。"])
    if branch_objs:
        lines.append("- 目标对象还必须至少属于下面一个概念：")
        lines.extend(_concept_lines(branch_objs, "A", start=2))
    if b_objs:
        lines.append("- 排除属于下面任一概念的对象：")
        lines.extend(_concept_lines(b_objs, "B"))
    else:
        lines.append("- 没有排除概念。")


def _append_notes(lines: list[str], family: str, information: str, concept_objs: list[tuple[str, dict]], candidate_objs: list[dict], single_candidate: bool) -> None:
    if information in ("I1", "I2"):
        concept_notes = []
        for prefix, obj in concept_objs:
            gloss = obj.get("gloss") or ""
            if gloss:
                concept_notes.append(f"- {prefix} {obj['surface']}: {gloss}")
        if concept_notes:
            lines.append("")
            lines.append("概念解释：" if family == "babelnet" else "Concept notes:")
            lines.extend(concept_notes)
    if information == "I2":
        candidate_notes = _candidate_note_lines(candidate_objs, single_candidate)
        if candidate_notes:
            lines.append("")
            lines.append("候选解释：" if family == "babelnet" else "Candidate notes:")
            lines.extend(candidate_notes)


def _render_wordnet(sample: dict, a1_obj: dict, branch_objs: list[dict], b_objs: list[dict], candidate_objs: list[dict]) -> tuple[str, str, str]:
    task = sample["task"]
    lines = ["You are given a concept-boundary question.", ""]
    _append_wordnet_condition(lines, a1_obj, branch_objs, b_objs)
    lines.append("")
    if task == "T1":
        lines.extend(["Candidate:", f"- {candidate_objs[0]['surface']}"])
    else:
        lines.append("Candidates:")
        lines.extend(_candidate_lines(candidate_objs))
    concept_objs = [("A1", a1_obj)] + [(f"A{index}", obj) for index, obj in enumerate(branch_objs, 2)] + [(f"B{index}", obj) for index, obj in enumerate(b_objs, 1)]
    _append_notes(lines, "wordnet", sample["information"], concept_objs, candidate_objs, single_candidate=task == "T1")
    lines.append("")
    if task == "T1":
        lines.extend(["Question: Does the candidate satisfy the target condition?", "Required answer: begin with True or False."])
        expected = "boolean_first_word"
    elif task == "T2":
        lines.append("Required answer: return only the list of candidate numbers that satisfy the target condition, for example [1, 3] or [].")
        expected = "candidate_number_list"
    else:
        lines.extend(["Question: Does at least one candidate satisfy the target condition?", "Required answer: begin with True or False. True means at least one candidate satisfies the target condition."])
        expected = "boolean_first_word"
    return "\n".join(lines), "Answer exactly in the required format.", expected


def _render_babelnet(sample: dict, a1_obj: dict, branch_objs: list[dict], b_objs: list[dict], candidate_objs: list[dict]) -> tuple[str, str, str]:
    task = sample["task"]
    lines = ["请判断下面的概念边界题。", ""]
    _append_babelnet_condition(lines, a1_obj, branch_objs, b_objs)
    lines.append("")
    if task == "T1":
        lines.extend(["候选项：", f"- {candidate_objs[0]['surface']}"])
    else:
        lines.append("候选项：")
        lines.extend(_candidate_lines(candidate_objs))
    concept_objs = [("A1", a1_obj)] + [(f"A{index}", obj) for index, obj in enumerate(branch_objs, 2)] + [(f"B{index}", obj) for index, obj in enumerate(b_objs, 1)]
    _append_notes(lines, "babelnet", sample["information"], concept_objs, candidate_objs, single_candidate=task == "T1")
    lines.append("")
    if task == "T1":
        lines.extend(["问题：该候选项是否满足目标条件？", "作答要求：第一个词必须是 True 或 False。"])
        expected = "boolean_first_word"
    elif task == "T2":
        lines.append("作答要求：只输出满足目标条件的候选编号列表，例如 [1, 3] 或 []。")
        expected = "candidate_number_list"
    else:
        lines.extend(["问题：候选项中是否至少有一个满足目标条件？", "作答要求：第一个词必须是 True 或 False；True 表示至少存在一个满足条件的候选项。"])
        expected = "boolean_first_word"
    return "\n".join(lines), "请严格按照题目要求的格式作答。", expected


def render_sample(sample: dict, resources: dict[str, dict[str, dict]]) -> RenderedRequest:
    family = sample["dataset_family"]
    objects = resources[family]
    a1_obj = objects[sample["A1_id"]]
    branch_objs = [objects[object_id] for object_id in sample.get("A_branch_ids", [])]
    b_objs = [objects[object_id] for object_id in sample.get("B_ids", [])]
    candidate_objs = [objects[object_id] for object_id in sample["candidate_ids"]]
    if family == "babelnet":
        prompt, system_prompt, expected = _render_babelnet(sample, a1_obj, branch_objs, b_objs, candidate_objs)
    else:
        prompt, system_prompt, expected = _render_wordnet(sample, a1_obj, branch_objs, b_objs, candidate_objs)
    request_hash = stable_hash(
        {
            "sample_id": sample["sample_id"],
            "renderer_version": RENDERING_TEMPLATE_VERSION,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "expected_protocol": expected,
        },
        length=32,
    )
    request_id = f"request_{request_hash[:16]}"
    rendered_sample = dict(sample)
    rendered_sample["request_hash"] = request_hash
    rendered_sample["renderer_version"] = RENDERING_TEMPLATE_VERSION
    return RenderedRequest(
        request_id=request_id,
        sample_id=sample["sample_id"],
        base_sample_id=sample["base_sample_id"],
        task=sample["task"],
        dataset_family=family,
        information=sample["information"],
        renderer_version=RENDERING_TEMPLATE_VERSION,
        prompt=prompt,
        system_prompt=system_prompt,
        expected_protocol=expected,
        request_hash=request_hash,
        provider_payload={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        },
        sample=rendered_sample,
    )


def render_samples(samples: list[dict], resources: dict[str, dict[str, dict]]) -> list[RenderedRequest]:
    return [render_sample(sample, resources) for sample in samples]
