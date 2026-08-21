import re
from typing import Dict, Any, List

CATEGORIES = {
    "Policy": [r"(?i)\bpolicy\b", r"(?i)\bguidelines\b", r"(?i)\bframework\b", r"(?i)\bact\b", r"(?i)\bregulation\b"],
    "Statistics": [r"%", r"\b\d{1,3}(,\d{3})+(\.\d+)?\b", r"(?i)\bdata\b", r"(?i)\brate\b", r"(?i)\bratio\b"],
    "Healthcare Infrastructure": [r"(?i)\bhospital\b", r"(?i)\bclinic\b", r"(?i)\bbed capacity\b", r"(?i)\binfrastructure\b", r"(?i)\bfacilities\b"],
    "Disease Burden": [r"(?i)\bdisease\b", r"(?i)\bburden\b", r"(?i)\bmorbidity\b", r"(?i)\bmortality\b", r"(?i)\binfection\b"],
    "Maternal Health": [r"(?i)\bmaternal\b", r"(?i)\bpregnancy\b", r"(?i)\bantenatal\b", r"(?i)\bneonatal\b", r"(?i)\bchildbirth\b"],
    "Healthcare Workforce": [r"(?i)\bdoctor\b", r"(?i)\bnurse\b", r"(?i)\bworkforce\b", r"(?i)\bstaff\b", r"(?i)\bspecialist\b"],
    "Budget": [r"(?i)\bfund\b", r"(?i)\bbudget\b", r"(?i)\bexpenditure\b", r"(?i)\bcrore\b", r"(?i)\bfinance\b"],
    "Recommendation": [r"(?i)\brecommend\b", r"(?i)\bsuggest\b", r"(?i)\bshould be\b", r"(?i)\bpropose\b"],
    "Table Explanation": [r"(?i)\btable \d+\b", r"(?i)\bfigure \d+\b", r"(?i)\bshown in table\b", r"(?i)\bas per table\b"]
}

def classify_chunk(text: str) -> str:
    """
    Classify a chunk into a category using keyword/regex heuristics.
    Returns the category with the highest number of matches.
    """
    scores = {cat: 0 for cat in CATEGORIES}
    
    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            scores[category] += len(matches)
            
    best_category = max(scores, key=scores.get)
    
    # If no specific matches, fall back to Narrative or General Healthcare
    if scores[best_category] == 0:
        if len(text.split()) > 100:
            return "Narrative"
        return "General Healthcare"
        
    return best_category

def classify_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add category to all chunks.
    """
    classified_chunks = []
    for chunk in chunks:
        category = classify_chunk(chunk.get('text', ''))
        new_chunk = chunk.copy()
        new_chunk['category'] = category
        classified_chunks.append(new_chunk)
    return classified_chunks
