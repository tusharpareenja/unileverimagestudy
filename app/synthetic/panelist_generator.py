import json
import os
import itertools
import random
from typing import List, Dict, Any, Tuple


def load_study_data(file_path: str) -> Dict[str, Any]:
    """Load the study data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_random_age_from_range(age_range: str) -> int:
    """
    Extract age range and return a random age within that range.
    """
    if age_range == "65+":
        return random.randint(65, 85)
    
    try:
        parts = age_range.split(" - ")
        if len(parts) == 2:
            min_age = int(parts[0].strip())
            max_age = int(parts[1].strip())
            return random.randint(min_age, max_age)
        else:
            age = int(age_range.strip().split()[0])
            return age
    except (ValueError, IndexError):
        return random.randint(18, 65)


def calculate_demographic_distribution(
    total_panelists: int,
    gender_distribution: Dict[str, float],
    age_distribution: Dict[str, float]
) -> List[Tuple[str, str]]:
    """
    Calculate demographic distribution (gender + age combinations) for panelists.
    """
    active_genders = {g: p for g, p in gender_distribution.items() if p > 0}
    active_ages = {a: p for a, p in age_distribution.items() if p > 0}
    
    combinations = []
    counts = []
    
    for gender, gender_pct in active_genders.items():
        for age_range, age_pct in active_ages.items():
            expected_count = total_panelists * (gender_pct / 100.0) * (age_pct / 100.0)
            combinations.append((gender, age_range))
            counts.append(expected_count)
    
    distribution = []
    total_assigned = 0
    rounded_counts = []
    remainders = []
    
    for i, count in enumerate(counts):
        rounded = int(count)
        remainder = count - rounded
        rounded_counts.append(rounded)
        remainders.append((i, remainder))
        total_assigned += rounded
    
    remaining = total_panelists - total_assigned
    
    if remaining > 0:
        remainders.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(remaining, len(remainders))):
            idx = remainders[i][0]
            rounded_counts[idx] += 1
    
    for i, (gender, age_range) in enumerate(combinations):
        count = rounded_counts[i]
        distribution.extend([(gender, age_range)] * count)
    
    random.shuffle(distribution)
    return distribution


def generate_all_panelist_combinations(study_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all possible panelist combinations based on classification questions.
    Each panelist will have a unique combination of answers plus age and gender
    assigned based on audience segmentation.
    """
    classification_questions = study_data.get('classification_questions', [])
    
    if not classification_questions:
        return []
    
    questions = []
    for q in classification_questions:
        answer_options = q.get('answer_options', [])
        answer_texts = [opt.get('text', str(opt)) for opt in answer_options]
        questions.append({
            'question_id': q['question_id'],
            'question_text': q['question_text'],
            'answer_options': answer_options,
            'answer_texts': answer_texts,
            'order': q.get('order', 0)
        })
    
    questions.sort(key=lambda x: x['order'])
    answer_option_lists = [q['answer_texts'] for q in questions]
    
    total_combinations = 1
    for options in answer_option_lists:
        total_combinations *= len(options)
    
    audience_seg = study_data.get('audience_segmentation', {})
    gender_distribution = audience_seg.get('gender_distribution', {})
    age_distribution = audience_seg.get('age_distribution', {})
    
    demographic_distribution = calculate_demographic_distribution(
        total_combinations,
        gender_distribution,
        age_distribution
    )
    
    panelists = []
    for idx, combination in enumerate(itertools.product(*answer_option_lists), start=1):
        panelist = {
            'panelist_id': f"panelist_{idx:06d}",
            'panelist_number': idx,
            'answers': {}
        }
        
        for i, question in enumerate(questions):
            answer_text = combination[i]
            answer_index = question['answer_texts'].index(answer_text)
            panelist['answers'][question['question_id']] = {
                'question_text': question['question_text'],
                'answer': answer_text,
                'answer_index': answer_index,
                'order': question['order']
            }
        
        if idx <= len(demographic_distribution):
            gender, age_range = demographic_distribution[idx - 1]
            panelist['gender'] = gender
            panelist['age_range'] = age_range
            panelist['age'] = get_random_age_from_range(age_range)
        else:
            panelist['gender'] = list(gender_distribution.keys())[0] if gender_distribution else 'unknown'
            fallback_age_range = list(age_distribution.keys())[0] if age_distribution else 'unknown'
            panelist['age_range'] = fallback_age_range
            panelist['age'] = get_random_age_from_range(fallback_age_range)
        
        panelists.append(panelist)
    
    return panelists


def create_panelists_json(study_data: Dict[str, Any], panelists: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a comprehensive JSON structure with study info and all panelists."""
    return {
        'study_id': study_data.get('id'),
        'study_title': study_data.get('title'),
        'total_panelists': len(panelists),
        'classification_questions_summary': [
            {
                'question_id': q['question_id'],
                'question_text': q['question_text'],
                'number_of_options': len(q['answer_options']),
                'order': q.get('order', 0)
            }
            for q in study_data.get('classification_questions', [])
        ],
        'panelists': panelists
    }
