from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

class EnsembleModel:
    def __init__(self):
        self.processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
        self.queries = queries = [
            "Describe what type of clothes you see?",
            "Describe the overall style of the clothing in the picture?",
            "What are the dominant colors in the outfits?",
            "What type of occasion or event would these clothes be suitable for?",
            "Are there any specific details or embellishments on the garments that catch your attention?",
            "Are there any particular body types or figures that these clothes would flatter?",
            "Can you suggest the age group or demographic that these outfits would appeal to?"
        ]
        self.query_classes = [
            ['Casual', 'Formal', 'Sportswear', 'Ethnic', 'Beachwear', 'Business'],
            ['Classic', 'Bohemian', 'Trendy', 'Vintage', 'Minimalistic', 'Eclectic'],
            ['Monochrome', 'Earthy', 'Vibrant', 'Pastel', 'Colorful','Patterned'],
            ['Formal', 'Casual', 'Sports', 'Wedding','Business','Beach'],
            ['Embroidery', 'Sequins', 'Ruffles', 'Lace', 'Buttons', 'Trims'],
            ['Tall', 'Short','Plus-size'],
            ['Teenagers', 'Young adults', 'Middle-aged','Seniors']
        ]

        self.model_zhic = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor_zhic = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _get_basic_descr(self, image):
        # conditional image captioning
        text = "a photography of"
        inputs = self.processor_blip(image, text, return_tensors="pt").to("cpu")

        out = self.model_blip.generate(**inputs)
        descr = self.processor_blip.decode(out[0], skip_special_tokens=True)
        return descr

    def _generate_open_questions_descr(self, image):
        answers = []
        for idx, query in enumerate(self.queries):
            classes = self.query_classes[idx]
            inputs = self.processor_zhic(text=classes, images=image, return_tensors="pt", padding=True)
            outputs = self.model_zhic(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

            max = logits_per_image.argmax(-1).item()
            answers.append(query + classes[max] + '\n')
        return ' '.join(answers)

    def get_label(self, image):
            description = ""
            description += self._get_basic_descr(image)
            description += ' \n' + self._generate_open_questions_descr(image)
            return description
