"""Test script to verify StudyOut schema correctly maps layers to study_layers"""

from uuid import uuid4
from datetime import datetime
from app.schemas.study_schema import StudyOut, StudyLayerOut, LayerImageOut, RatingScale, AudienceSegmentation

# Mock a study object with layers
class MockStudy:
    def __init__(self):
        self.id = uuid4()
        self.title = "Test Study"
        self.background = "Test background"
        self.language = "en"
        self.main_question = "Test question"
        self.orientation_text = "Test orientation"
        self.study_type = "layer"
        self.background_image_url = None
        self.aspect_ratio = "3:4"
        self.rating_scale = {"min_value": 1, "max_value": 5, "min_label": "Low", "max_label": "High"}
        self.audience_segmentation = {"number_of_respondents": 50}
        self.categories = []
        self.elements = []
        # Model has 'layers' attribute
        self.layers = [
            type('Layer', (), {
                'id': uuid4(),
                'layer_id': 'layer1',
                'name': 'Layer 1',
                'description': 'Test layer',
                'z_index': 1,
                'order': 1,
                'transform': None,
                'images': [
                    type('Image', (), {
                        'id': uuid4(),
                        'image_id': 'img1',
                        'name': 'Image 1',
                        'url': 'https://example.com/img1.jpg',
                        'alt_text': 'Test image',
                        'order': 1
                    })()
                ]
            })()
        ]
        self.classification_questions = []
        self.tasks = {}
        self.creator_id = uuid4()
        self.status = "draft"
        self.share_token = "test-token"
        self.share_url = "https://example.com/test"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.launched_at = None
        self.completed_at = None
        self.total_responses = 0
        self.completed_responses = 0
        self.abandoned_responses = 0

def test_schema_mapping():
    print("Testing StudyOut schema mapping...")

    mock_study = MockStudy()

    # Validate the model
    try:
        study_out = StudyOut.model_validate(mock_study)
        result = study_out.model_dump()

        print(f"\n[TEST] Checking 'study_layers' field in output...")
        if 'study_layers' in result:
            print(f"[PASS] 'study_layers' field exists in output")
            if result['study_layers']:
                print(f"[PASS] 'study_layers' has data: {len(result['study_layers'])} layer(s)")
                layer = result['study_layers'][0]
                if 'images' in layer and layer['images']:
                    print(f"[PASS] Layer has images: {len(layer['images'])} image(s)")
                    print(f"       Image data: {layer['images'][0]}")
                else:
                    print(f"[FAIL] Layer has no images")
            else:
                print(f"[FAIL] 'study_layers' is empty or null")
        else:
            print(f"[FAIL] 'study_layers' field NOT in output")
            print(f"       Available fields: {list(result.keys())}")

        print(f"\n[TEST] Checking if 'layers' field exists (should not)...")
        if 'layers' in result:
            print(f"[FAIL] 'layers' field exists (should be 'study_layers')")
        else:
            print(f"[PASS] 'layers' field does not exist (correct)")

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_schema_mapping()
