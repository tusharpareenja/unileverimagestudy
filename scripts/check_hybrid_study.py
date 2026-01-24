"""
Diagnostic script to check hybrid study configuration.
Run this to see:
1. Study's phase_order value
2. Category phase_type values
3. Sample tasks with their phase_type
4. Element content types (text vs image)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import UUID
from app.db.session import SessionLocal
from app.models.study_model import Study, StudyCategory, StudyElement

# Change this to your study ID
STUDY_ID = "8d7ce151-3885-4683-bd70-881ec5a7bdb4"

def main():
    db = SessionLocal()
    try:
        study = db.get(Study, UUID(STUDY_ID))
        if not study:
            print(f"Study {STUDY_ID} not found!")
            return
        
        print("=" * 60)
        print(f"STUDY: {study.title}")
        print(f"TYPE: {study.study_type}")
        print(f"PHASE_ORDER: {study.phase_order}")
        print("=" * 60)
        
        # Check categories with their phase_type
        print("\nCATEGORIES AND ELEMENTS:")
        categories = db.query(StudyCategory).filter(
            StudyCategory.study_id == study.id
        ).order_by(StudyCategory.order).all()
        
        grid_cats = []
        text_cats = []
        none_cats = []
        
        for cat in categories:
            phase = cat.phase_type
            if phase == 'grid':
                grid_cats.append(cat.name)
            elif phase == 'text':
                text_cats.append(cat.name)
            else:
                none_cats.append(cat.name)
            
            # Get elements for this category
            elements = db.query(StudyElement).filter(
                StudyElement.category_id == cat.id
            ).all()
            
            print(f"\n  [{cat.phase_type or 'None'}] {cat.name} ({len(elements)} elements):")
            for elem in elements[:3]:  # Show first 3 elements
                content_preview = (elem.content or '')[:50] + '...' if elem.content and len(elem.content) > 50 else elem.content
                print(f"      - {elem.name}: {content_preview}")
            if len(elements) > 3:
                print(f"      ... and {len(elements) - 3} more elements")
        
        print("\n" + "-" * 60)
        print(f"SUMMARY:")
        print(f"  Grid categories (phase_type='grid'): {len(grid_cats)} - {grid_cats}")
        print(f"  Text categories (phase_type='text'): {len(text_cats)} - {text_cats}")
        print(f"  Categories without phase_type: {len(none_cats)} - {none_cats}")
        
        if study.study_type == 'hybrid' and len(text_cats) == 0:
            print("\n  ⚠️  WARNING: This is a hybrid study but NO text categories found!")
            print("     Text phase will have 0 tasks. All tasks will be grid type.")
        
        # Check sample tasks
        if study.tasks:
            print("\n" + "-" * 60)
            print("SAMPLE TASKS:")
            
            # Find tasks for any available respondent
            for resp_key in ['1', '2', '3', '4', '5']:
                tasks = study.tasks.get(resp_key, [])
                if tasks:
                    print(f"\n  Respondent {resp_key} ({len(tasks)} total tasks):")
                    
                    # Count tasks by phase_type
                    phase_counts = {}
                    for t in tasks:
                        pt = t.get('phase_type', 'None')
                        phase_counts[pt] = phase_counts.get(pt, 0) + 1
                    
                    print(f"    Tasks by phase_type: {phase_counts}")
                    
                    # Show first 3 and last 3 tasks
                    print("\n    First 3 tasks:")
                    for i, t in enumerate(tasks[:3]):
                        esc = t.get('elements_shown_content', {})
                        has_images = any(v and isinstance(v, dict) and v.get('content', '').endswith(('.png', '.jpg', '.jpeg', '.gif'))
                                        for v in esc.values() if v)
                        has_text = any(v and isinstance(v, dict) and v.get('element_type') == 'text'
                                      for v in esc.values() if v)
                        content_type = "images" if has_images else ("text" if has_text else "unknown")
                        print(f"      Task {i}: task_id={t.get('task_id')}, phase_type={t.get('phase_type')}, content={content_type}")
                    
                    if len(tasks) > 6:
                        print(f"      ... ({len(tasks) - 6} middle tasks) ...")
                    
                    print("\n    Last 3 tasks:")
                    for i, t in enumerate(tasks[-3:], start=len(tasks)-3):
                        esc = t.get('elements_shown_content', {})
                        has_images = any(v and isinstance(v, dict) and v.get('content', '').endswith(('.png', '.jpg', '.jpeg', '.gif'))
                                        for v in esc.values() if v)
                        has_text = any(v and isinstance(v, dict) and v.get('element_type') == 'text'
                                      for v in esc.values() if v)
                        content_type = "images" if has_images else ("text" if has_text else "unknown")
                        print(f"      Task {i}: task_id={t.get('task_id')}, phase_type={t.get('phase_type')}, content={content_type}")
                    break
        else:
            print("\nNo tasks generated!")
        
        print("\n" + "=" * 60)
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
