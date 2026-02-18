"""
AI Respondent for generating survey responses based on panelist personas.

This module provides functionality to generate survey responses using AI,
where each response rates entire vignettes (combinations of elements) rather
than individual elements.
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIRespondent:
    """
    AI-powered respondent that simulates survey responses based on a panelist persona.
    
    This class handles rating entire vignettes (combinations of elements) based on
    the panelist's classification answers and persona.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the AI Respondent.
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini or OPENAI_MODEL env var)
                   Note: For image support, use vision-capable models like gpt-4o or gpt-4o-mini
        """
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.client = None
        
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("Warning: No OpenAI API key provided. AI features will be disabled.")
                print("Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        else:
            print("Warning: OpenAI library not available. Install with: pip install openai")
            print("AI features will use fallback heuristic ratings.")
    
    def build_persona_prompt(self, panelist: Dict[str, Any], study_data: Dict[str, Any]) -> str:
        """
        Build a detailed persona prompt for the AI based on panelist answers and demographics.
        
        Args:
            panelist: Panelist dictionary with classification answers, gender, and age_range
            study_data: Study data with background, objectives, etc.
        
        Returns:
            Formatted persona prompt string
        """
        answers = panelist.get('answers', {})
        gender = panelist.get('gender', 'unknown')
        age_range = panelist.get('age_range', 'unknown')
        
        # Build factors from preliminary questions (classification answers)
        factors_lines = []
        for q_id in sorted(answers.keys(), key=lambda x: answers[x].get('order', 0)):
            answer_data = answers[q_id]
            factors_lines.append(f"- {answer_data['question_text']}: {answer_data['answer']}")
        
        factors_text = "\n".join(factors_lines) if factors_lines else "- No preliminary factors defined"
        
        # Get study background and objectives
        background = study_data.get('background', '')
        objectives = study_data.get('objectives_text', study_data.get('orientation_text', ''))
        main_question = study_data.get('main_question', '')
        
        # Get rating scale labels
        rating_scale = study_data.get('rating_scale', {})
        min_label = rating_scale.get('min_label', 'Bad (does not appeal to you, doesn\'t align with your preferences)')
        middle_label = rating_scale.get('middle_label', 'Bad (does not appeal to you, doesn\'t align with your preferences)')
        max_label = rating_scale.get('max_label', 'Very good (strongly appeals to you, perfectly aligns with your preferences)')
        
        prompt = f"""You are a role-based evaluator. Your judgment should reflect how things make sense from within the role defined. Do not generalize beyond it. You are evaluating from a specific human context defined on the following role assigned to you.

You are taking on the following role. You are  representing a lived context shaped by the following factors

{factors_text}

These factors influence your sensitivity, priorities, and interpretation, NOT expertise or factual knowledge.

Do NOT evaluate scientific accuracy, technical feasibility, or clinical truth. Evaluate perceived sense-making only specific to your current role defined by the factors above.

You are currently participating in an evaluation where this role evaluates various ideas related to the following background and objectives.

Background: {background}

Objectives: {objectives}

Now you will be presented with a set of stimuli. There are multiple elements together to present ONE stimulus object. Based on your current role evaluate whole set as a combined proposition to elicit a rating response the following question

{main_question}

Rate the entire SET as a WHOLE in response to the question on a scale of 1-5 where:

<<1 = {min_label}
2 = 
3 = {middle_label}
4 = 
5 = {max_label}>>"""
        
        return prompt
    
    def rate_vignette_with_ai(self, task: Dict[str, Any], persona_prompt: str, study_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rate an entire vignette (task with combined elements) based on the persona.
        
        IMPORTANT: This rates the ENTIRE VIGNETTE as a whole, not individual elements.
        A vignette is a combination of multiple elements shown together, and we rate
        how well this combination works together based on the persona.
        
        Args:
            task: Task dictionary with elements_shown_content and elements_shown
            persona_prompt: The persona description
            study_context: Study context (background, main question, etc.)
        
        Returns:
            Dictionary with rating (1-5) and reasoning for the entire vignette
        """
        if not self.client:
            return self._generate_fallback_vignette_rating(task, persona_prompt)
        
        # Extract elements shown in this vignette
        elements_shown_content = task.get('elements_shown_content', {})
        elements_shown = task.get('elements_shown', {})
        
        # Get only the elements that are actually shown (value = 1)
        shown_elements = []
        for key, value in elements_shown.items():
            if value == 1 and key in elements_shown_content:
                element_data = elements_shown_content[key]
                if element_data and isinstance(element_data, dict):
                    shown_elements.append(element_data)
        
        if not shown_elements:
            # No elements shown, return neutral rating
            return {
                'rating': 3,
                'reasoning': 'No elements shown in this vignette',
                'method': 'fallback'
            }
        
        # Build vignette description - separate text and images
        vignette_text_parts = []
        image_urls = []
        
        for element in shown_elements:
            category = element.get('category_name', 'Unknown')
            content = element.get('content', element.get('name', 'Unknown'))
            element_type = element.get('element_type', 'text')
            
            # Check if content is an image URL
            is_image_url = False
            if element_type == 'image':
                is_image_url = True
            elif isinstance(content, str) and (
                content.startswith('http://') or 
                content.startswith('https://')
            ):
                # Check if URL looks like an image (common image extensions)
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp']
                is_image_url = any(content.lower().endswith(ext) for ext in image_extensions)
            
            if is_image_url:
                # Store image URL for vision API
                image_urls.append({
                    'type': 'image_url',
                    'image_url': {'url': content}
                })
                vignette_text_parts.append(f"{category}: [Image]")
            else:
                # Regular text content
                vignette_text_parts.append(f"{category}: {content}")
        
        vignette_text = "\n".join(vignette_text_parts)
        
        # Build the user message content
        user_content = []
        
        # Build prompt based on whether there are images
        has_images = len(image_urls) > 0
        image_instruction = ""
        if has_images:
            image_instruction = """
IMPORTANT: This stimulus set contains IMAGES. Please carefully analyze each image visually:
- Look at the visual design, colors, composition, and overall aesthetic
- Consider how the images relate to the background and objectives
- Evaluate how well the images resonate with your role and the factors shaping your perspective
- Assess how the images work together as a cohesive visual experience
- Rate based on perceived sense-making from your role, NOT technical accuracy
"""
        else:
            image_instruction = ""
        
        # Add text prompt
        text_prompt = f"""{persona_prompt}

STIMULUS SET (evaluate as ONE combined proposition):
{vignette_text}
{image_instruction}

Respond ONLY with a JSON object in this exact format:
{{
    "rating": <number between 1 and 5>,
    "reasoning": "<brief explanation of why you gave this rating based on your role, the factors shaping your perspective, and how the stimulus set resonates with you>"
}}
"""
        user_content.append({"type": "text", "text": text_prompt})
        
        # Add images if any
        user_content.extend(image_urls)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a role-based evaluator providing ratings from within a defined human context. Your judgment reflects perceived sense-making, not expertise or factual knowledge. Always respond with valid JSON only."},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,  # Some variability but still consistent
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            rating = int(result.get('rating', 3))
            # Ensure rating is in valid range
            rating = max(1, min(5, rating))
            
            return {
                'rating': rating,
                'reasoning': result.get('reasoning', ''),
                'method': 'ai'
            }
        except Exception as e:
            print(f"Error calling AI API: {e}. Using fallback method.")
            return self._generate_fallback_vignette_rating(task, persona_prompt)
    
    def _generate_fallback_vignette_rating(self, task: Dict[str, Any], persona_prompt: str) -> Dict[str, Any]:
        """
        Generate a fallback rating for a vignette when AI is not available.
        
        Args:
            task: Task dictionary
            persona_prompt: Persona description (not used in fallback, but kept for consistency)
        
        Returns:
            Dictionary with rating and reasoning
        """
        # Base rating around 3 (neutral)
        rating = 3
        
        # Add some randomness
        rating += random.randint(-1, 1)
        
        # Ensure valid range
        rating = max(1, min(5, rating))
        
        return {
            'rating': rating,
            'reasoning': 'Fallback heuristic rating (AI not available)',
            'method': 'fallback'
        }


def generate_panelist_response_from_json(
    panelist_json: Dict[str, Any],
    tasks_json: Dict[str, List[Dict[str, Any]]],
    study_data: Dict[str, Any],
    openai_api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_vignette_workers: int = 10
) -> Dict[str, Any]:
    """
    Generate a complete survey response for a panelist using JSON objects directly.
    
    Designed for multithreading and Selenium automation with vignette-based tasks.
    This function rates entire vignettes (combinations of elements), not individual elements.
    """
    # Initialize AI respondent
    ai_respondent = AIRespondent(openai_api_key=openai_api_key, model=model)
    
    # Build persona prompt
    persona_prompt = ai_respondent.build_persona_prompt(panelist_json, study_data)
    
    # Get panelist number to find tasks
    panelist_number = panelist_json.get('panelist_number')
    if panelist_number is None:
        raise ValueError("panelist_json must contain 'panelist_number'")
    
    # Get tasks for this panelist
    panelist_tasks = tasks_json.get(str(panelist_number), [])
    
    if not panelist_tasks:
        raise ValueError(f"No tasks found for panelist number {panelist_number}")
    
    # Rate each vignette/task in parallel
    def rate_single_task(task_data: tuple) -> Dict[str, Any]:
        """Helper function to rate a single task/vignette."""
        idx, task = task_data
        task_id = task.get('task_id', f"{panelist_number}_{idx}")
        task_index = task.get('task_index', idx)
        
        # Rate the entire vignette (all elements combined together as one unit)
        rating_result = ai_respondent.rate_vignette_with_ai(task, persona_prompt, study_data)
        
        # Build vignette content string for output
        elements_shown_content = task.get('elements_shown_content', {})
        elements_shown = task.get('elements_shown', {})
        vignette_parts = []
        for key, value in elements_shown.items():
            if value == 1 and key in elements_shown_content:
                element_data = elements_shown_content[key]
                if element_data and isinstance(element_data, dict):
                    category = element_data.get('category_name', 'Unknown')
                    content = element_data.get('content', element_data.get('name', 'Unknown'))
                    vignette_parts.append(f"{category}: {content}")
        vignette_content = "\n".join(vignette_parts)
        
        # Get elements shown in this vignette (for reference/debugging)
        elements_shown_content = task.get('elements_shown_content', {})
        elements_shown = task.get('elements_shown', {})
        
        # Extract shown elements
        shown_elements = []
        for key, value in elements_shown.items():
            if value == 1 and key in elements_shown_content:
                element_data = elements_shown_content[key]
                if element_data and isinstance(element_data, dict):
                    shown_elements.append({
                        'key': key,
                        'element_id': element_data.get('element_id'),
                        'name': element_data.get('name'),
                        'content': element_data.get('content'),
                        'category_name': element_data.get('category_name'),
                        'element_type': element_data.get('element_type', 'text')
                    })
        
        return {
            'task_id': task_id,
            'task_index': task_index,
            'main_question': study_data.get('main_question', ''),
            'vignette_content': vignette_content,
            'rating': rating_result['rating'],
            'reasoning': rating_result.get('reasoning', ''),
            'elements_shown': shown_elements,
            'method': rating_result.get('method', 'unknown'),
            '_original_index': idx  # Preserve original order
        }
    
    # Process all tasks in parallel
    max_workers = min(max_vignette_workers, len(panelist_tasks))
    task_ratings = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(rate_single_task, (idx, task)): idx
            for idx, task in enumerate(panelist_tasks)
        }
        
        results = {}
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results[result['_original_index']] = result
            except Exception as e:
                idx = future_to_index[future]
                results[idx] = {
                    'task_id': f"{panelist_number}_{idx}",
                    'task_index': idx,
                    'main_question': study_data.get('main_question', ''),
                    'vignette_content': '',
                    'rating': 3,
                    'reasoning': f'Error during rating: {str(e)}',
                    'elements_shown': [],
                    'method': 'error',
                    '_original_index': idx
                }
        
        task_ratings = [results[i] for i in sorted(results.keys())]
        for tr in task_ratings:
            tr.pop('_original_index', None)
    
    # Extract classification answers in a clean format
    classification_answers = {}
    for q_id, answer_data in panelist_json.get('answers', {}).items():
        classification_answers[q_id] = {
            'question_id': q_id,
            'question_text': answer_data['question_text'],
            'answer': answer_data['answer'],
            'answer_index': answer_data['answer_index']
        }
    
    # Format for Selenium automation
    selenium_data = {
        'classification_answers': classification_answers,
        'task_ratings': [
            {
                'task_id': tr['task_id'],
                'task_index': tr['task_index'],
                'main_question': tr.get('main_question', ''),
                'vignette_content': tr.get('vignette_content', ''),
                'rating': tr['rating']
            }
            for tr in task_ratings
        ],
        'study_id': study_data.get('id'),
        'rating_scale': study_data.get('rating_scale', {'min_value': 1, 'max_value': 5})
    }
    
    return {
        'panelist_id': panelist_json.get('panelist_id'),
        'panelist_number': panelist_json.get('panelist_number'),
        'classification_answers': classification_answers,
        'task_ratings': task_ratings,
        'ready_for_selenium': selenium_data,
        'study_id': study_data.get('id'),
        'study_title': study_data.get('title'),
        'total_tasks': len(task_ratings),
        'generated_at': datetime.now(timezone.utc).isoformat()
    }


def process_panelist_response(
    panelist_json: Dict[str, Any],
    tasks_json: Dict[str, List[Dict[str, Any]]],
    study_data: Dict[str, Any],
    openai_api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_vignette_workers: int = 10
) -> Dict[str, Any]:
    """
    Main function to process a single panelist and generate survey response.
    
    This is a clean, simple interface that takes a panelist and their tasks,
    processes them, and returns the complete response ready for Selenium automation.
    """
    return generate_panelist_response_from_json(
        panelist_json=panelist_json,
        tasks_json=tasks_json,
        study_data=study_data,
        openai_api_key=openai_api_key,
        model=model,
        max_vignette_workers=max_vignette_workers
    )
