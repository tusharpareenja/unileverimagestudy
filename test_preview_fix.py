#!/usr/bin/env python3
"""
Test script to verify the preview task generation fix
"""
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_preview_tasks():
    """Test the preview task generation with mock data"""
    print("🧪 Testing Preview Task Generation Fix...")
    
    # Mock payload structure
    class MockElement:
        def __init__(self, element_id, name, content, category_id, element_type, alt_text=None):
            self.element_id = element_id
            self.name = name
            self.content = content
            self.category_id = category_id
            self.element_type = element_type
            self.alt_text = alt_text
    
    class MockCategory:
        def __init__(self, category_id, name):
            self.category_id = category_id
            self.name = name
    
    class MockPayload:
        def __init__(self):
            self.study_type = 'grid'
            self.elements = [
                MockElement('elem1', 'Element 1', 'content1', 'cat1', 'image'),
                MockElement('elem2', 'Element 2', 'content2', 'cat1', 'image'),
                MockElement('elem3', 'Element 3', 'content3', 'cat2', 'image'),
            ]
            self.categories = [
                MockCategory('cat1', 'Category 1'),
                MockCategory('cat2', 'Category 2'),
            ]
    
    # Test the function
    try:
        from app.api.v1.study import _generate_preview_tasks
        
        payload = MockPayload()
        number_of_respondents = 100
        
        result = _generate_preview_tasks(payload, number_of_respondents)
        
        print(f"✅ Preview generation successful!")
        print(f"✅ Generated tasks for {len(result)} respondents")
        print(f"✅ Result type: {type(result)}")
        
        if result:
            print(f"✅ Sample task keys: {list(result.keys())[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preview_tasks()
    if success:
        print("🎉 Preview task generation fix is working!")
    else:
        print("💥 Preview task generation still has issues!")
