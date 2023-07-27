import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

class AutomateLablingModel:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "alesanm/blip-image-captioning-base-fashionimages-finetuned-processor")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "alesanm/blip-image-captioning-base-fashionimages-finetuned")
        self.model.to(torch.device("cpu"))
        self.model.eval()

    def get_label(self, pil_image):

        inputs = self.processor(images=pil_image, return_tensors="pt").to(torch.device("cpu"))
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=1500)
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return description
