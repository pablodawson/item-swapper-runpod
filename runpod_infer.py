''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'image': {
        'type': str,
        'required': True,
        'default': None
    },
    'seg': {
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

    # Download input objects
    job_input['image'], job_input['seg'] = rp_download.download_input_objects(
        [job_input.get('image', None), job_input.get('seg', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking

    img_path = MODEL.predict(
        width=job_input.get('width', 512),
        image=job_input['image'],
        seg=job_input['seg'],
        num_inference_steps=job_input.get('num_inference_steps', 50),
        guidance_scale=job_input['guidance_scale'],
        scheduler=job_input.get('scheduler', "K-LMS"),
        swap=job_input['swap']
    )

    job_output = []
    image_url = rp_upload.upload_image(job['id'], img_path, 0)

    job_output.append({
        "image": image_url
    })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output

runpod.serverless.start({"handler": run})