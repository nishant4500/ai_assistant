import json
from typing import List, Dict

def refine_sections(input_list: str, llm) -> List[Dict]:
    prompt = f"""
You are a precise data processor.

I will give you a JSON list of sections from a research paper. Each item may have:
- "section" (string)
- optional "subsection" (string)
- "start" (integer)

Some entries are unnecessary and must be **removed completely**:
1. Figure or Table captions (any "section" starting with "Figure" or "Table").
2. Incomplete, meaningless, or fragment sections (e.g., "making", "length nis smaller...").
3. Any other irrelevant entries that are not proper sections or subsections.

Your task is to **refine this list**:

- Keep only meaningful main sections and their subsections.
- Main sections should be in the format: 
  {{"section": "Section Name", "start": number}}
- Subsections should be in the format: 
  {{"section": "Parent Section", "subsection": "Subsection Name", "start": number}}
- The output must be **strictly a JSON array of dictionaries**.
- Do **not** include any explanations, notes, extra text, or commentary.
- If a section is unnecessary (e.g., figure, table, fragment), **exclude it completely**.

Here is the input JSON:

{input_list}

Always return list of dictionaries only, no preamble.
"""

    # Call the LLM and gracefully fallback if response is not usable.
    try:
        assistant_text = llm.invoke(prompt).content.strip()
        sections = json.loads(assistant_text)
        if not isinstance(sections, list):
            raise TypeError("Refined sections response is not a list")
    except Exception as e:
        print("Warning: Could not refine sections with LLM. Falling back to extracted sections.")
        print("Error:", e)
        try:
            sections = json.loads(input_list)
            if not isinstance(sections, list):
                sections = []
        except Exception:
            sections = []

    return sections


def split_sections_with_content(text: str, detected_sections: List[Dict]) -> List[Dict]:
    """
    Split text into sections/subsections using detected start positions.
    Returns a list of dicts with: section, subsection (if any), start, content.
    """
    if not detected_sections:
        return {"Full_Paper" : text}

    # Sort by start index to ensure correct order
    detected_sections = sorted(detected_sections, key=lambda x: x["start"])
    results = {}

    for i, sec in enumerate(detected_sections):
        start = sec["start"]
        end = detected_sections[i + 1]["start"] if i + 1 < len(detected_sections) else len(text)

        # Section details
        section_name = sec["section"]
        subsection_name = sec.get("subsection", None)
        section_text = text[start:end].strip()

        results[section_name] = section_text

        if subsection_name:
            results[subsection_name] = results.pop(section_name)

    return results