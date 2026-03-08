from llavashield.llavashield_utils import load_shield

model_path = 'RealSafe/LLaVAShield-v1.0-7B'
device = 'cuda'
device_map='auto'
attn_implementation="sdpa" # flash_attention_2 is highly recommended for better speed and memory efficiency.

usage_policy = ['Violence & Harm', 'Hate & Harassment', 'Sexual Content', 'Self-Harm & Suicide', 'Illegal Activities', 'Deception & Misinformation', 'Privacy Violation', 'Malicious Disruption']

processor, model = load_shield(model_path, usage_policy=usage_policy, device=device, device_map=device_map, attn_implementation=attn_implementation)

messages = [
    {
        'role': 'user',
        'content': [
            {"type": "image", "image": 'https://github.com/leost123456/LLaVAShield/blob/main/figs/cat.jpg?raw=true'},
            {"type": "text", "text": 'What kind of animal is this?'},
        ]
    },
    {
        'role': 'assistant',
        'content': "This is a cat."
    }
]

inputs = processor(messages=messages, device=device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=False,
)

response = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(response)