''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate
import json
import base64
from io import BytesIO
from PIL import Image

prod = True

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'image_b64': {
        'type': str,
        'required': True,
        'default': None
    },
    'seg_b64': {
        'type': str,
        'required': True,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 20
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': "K-LMS",
        'constraints': lambda scheduler: scheduler in ["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                                                         "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM", "K-LMS"]
        },
    'swap': {
        'type': list,
        'required': True,
    }, 
    'output_format':{
        'type': str,
        'required': False,
        'default': "all-in-one",
        'constraints': lambda output_format: output_format in ["all-in-one", "instances"]
    },
    'delivery': {
        'type': str,
        'required': False,
        'default': "base64",
        'constraints': lambda delivery: delivery in ["base64", "s3"]
    }
}


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # b64 -> Image
    image_bytes = base64.b64decode(validated_input['image_b64'].encode('utf-8'))
    seg_bytes = base64.b64decode(validated_input['seg_b64'].encode('utf-8'))

    if not os.path.exists("input_objects"):
        os.mkdir("input_objects")
    
    image = Image.open(BytesIO(image_bytes)).save('input_objects/image.png')
    seg = Image.open(BytesIO(seg_bytes)).save('input_objects/seg.png')

    # Convert swap list to json
    swap = json.dumps(validated_input['swap'])
    
    img_paths = MODEL.predict(
        width=validated_input.get('width', 512),
        image= "input_objects/image.png",
        seg= "input_objects/seg.png",
        num_inference_steps=validated_input.get('num_inference_steps', 50),
        guidance_scale=validated_input['guidance_scale'],
        scheduler=validated_input.get('scheduler', "K-LMS"),
        output_format=validated_input.get('output_format', "all-in-one"),
        swap=swap
    )

    job_output = []

    for path in img_paths:
        buffered = BytesIO()
        
        if (validated_input.get('output_format', "all-in-one") == "all-in-one"):
            Image.open(path).save(buffered, format="JPEG")
        else:
            Image.open(path).save(buffered, format="PNG")
        
        output = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
        job_output.append({
            "image_b64": output,
            "token": path.split("/")[-1].split(".")[0]
        })
    
    # Remove downloaded input objects
    if prod:
        rp_cleanup.clean(['input_objects', 'tmp'])
    
    return job_output

if prod:
    runpod.serverless.start({"handler": run})
else:
    job = {}
    job['id'] = 'test'

    swap_list = [{"color": [11,102,255], "lora": "rosjf-05", "prompt": "a room with a sky blue rosjf sofa", 
                  "convex_hull": True, "output-format": "all-in-one"}]
    
    image = "room.jpg"
    seg = "seg.png"

    # to base64
    with open(image, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(seg, "rb") as seg_file:
        seg = base64.b64encode(seg_file.read()).decode('utf-8')

    job['input'] = { "image_b64": image, "seg_b64": seg, "swap": swap_list , "width": 256}
    
    run(job)