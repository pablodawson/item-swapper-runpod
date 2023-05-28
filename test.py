import runpod
import base64
import time

runpod.api_key = "2ID7QJ6FO1PEUDBJQRA3GMLHHW3TZPSXF66VPP2T"
endpoint = runpod.Endpoint("630nkhjosg1478")

swap_list = [{"color": [11,102,255], "lora": "rosjf-05", "prompt": "a room with a sky blue rosjf sofa", "convex_hull": False}]

image = "room.jpg"
seg = "seg.png"

with open(image, "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode('utf-8')
with open(seg, "rb") as seg_file:
    seg = base64.b64encode(seg_file.read()).decode('utf-8')

model_input = { "image_b64": image, "seg_b64": seg, "swap": swap_list, "scheduler": "DPM-M", "width": 768, "guidance_scale": 12, "num_inference_steps": 16}

timestart = time.time()
run_request = endpoint.run(
    model_input
)

# Check the status of the endpoint run request
print(run_request.status())

# Get the output of the endpoint run request, blocking until the endpoint run is complete.
output = run_request.output()
print("Time to run: ", time.time() - timestart)

image = output[0]['image_b64']

# Decode the base64 image
image = base64.b64decode(image.encode('utf-8'))

# Save the image
with open("output.jpg", "wb") as f:
    f.write(image)
