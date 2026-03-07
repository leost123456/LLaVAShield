import json

class Keyword:
    def __init__(self, text, image_id="", caption=""):
        self.text = text
        self.image_id = image_id
        self.caption = caption

    def to_dict(self):
        return {
            "text": self.text,
            "image_id": self.image_id,
            "caption": self.caption
        }

class Task:
    def __init__(self, id, dimension, subdimension, definition, example, malicious_intent, keywords):
        self.id = id
        self.dimension = dimension
        self.subdimension = subdimension
        self.definition = definition
        self.example = example
        self.malicious_intent = malicious_intent
        self.keywords = [Keyword(**k) if isinstance(k, dict) else k for k in keywords]

    def to_dict(self):
        return {
            "id": self.id,
            "dimension": self.dimension,
            "subdimension": self.subdimension,
            "definition": self.definition,
            "example": self.example,
            "malicious_intent": self.malicious_intent,
            "keywords": [k.to_dict() for k in self.keywords]
        }
    
    def get_keyword_texts(self):
        return [k.text for k in self.keywords]
    
    def get_image_ids(self):
        return [k.image_id for k in self.keywords]

    def get_captions(self):
        return [k.caption for k in self.keywords]

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [
        Task(
            id=item["id"],
            dimension=item["dimension"],
            subdimension=item["subdimension"],
            definition=item["definition"],
            example=item["example"],
            malicious_intent=item["malicious_intent"],
            keywords=item["keywords"]
        )
        for item in data
    ]
