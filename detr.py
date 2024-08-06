#from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import tempfile

from ray import serve
# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection



@serve.deployment()
class ObjectDetection:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        #self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        #self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Users can send HTTP requests with an image. The classifier will return
    # the top prediction.
    # Sample output: {"prediction":["n02099601","golden_retriever",0.17944198846817017]}
    
    async def __call__(self, http_request):
        request = await http_request.form()
        image_file = await request["image"].read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_file)
            temp_file.close()
            temp_file_path = temp_file.name
            image = Image.open(temp_file_path)
            #img = image.load_img(temp_file_path, target_size=(224, 224))

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        result_list = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            result_list.append(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

        return result_list


app = ObjectDetection.bind()
