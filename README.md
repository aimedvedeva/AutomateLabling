# AutomateLabling
Take photos from the Vogue runway and generate a description

Here you can find [Pre-project study](https://www.canva.com/design/DAFon6U_fVM/rdfINuKaYGVPYMjUJH09Gw/edit?utm_content=DAFon6U_fVM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Experiment 1:
1. Generate long descriptions using an ensemble of pre-trained models (BLIP for general description + Zero-Shot Image Classification)

   **Example:**<br />
   a photograph of a person in a black coat and black pants<br />
   The overall style or aesthetic of the clothing in the picture is formal<br />
   The dominant colors in the outfits are brown<br />
   The type of occasion or event these clothes be suitable for is a wedding<br />
   The most probable personality or vibe that these clothes convey is The most probable type of occasion or event would these clothes be suitable for is cute<br />
   The most probable key fashion trend or influence evident in these outfits is casual<br />
   The most probable particular body types or figures that these clothes would flatter are Victorian<br />
   The most probable age group or demographic that these outfits would appeal to is medium<br />
   The most probable notable designer brands or fashion houses associated with these garments are millennia<br />

2. Evaluate the quality of the generated descriptions using ground truth GPT3 descriptions <br />
[Test dataset](https://huggingface.co/datasets/alesanm/chanel_long_descriptions)

## Experiment 2:

1. generated short GPT3 descriptions for the train set <br />
   **Example:**<br />
   Clothes: Formal wear, evening gowns, cocktail dresses;<br />
   Style: Elegant, sophisticated;<br />
   Colors: Neutral tones, black, beige, ivory;<br />
   Occasion: Special events, galas, parties;<br />
   Details: Lace, sequins, high-neck, long-sleeves;<br />
   Trends: Feminine, romantic<br />
[Train dataset](https://huggingface.co/datasets/alesanm/balenciaga_short_descriptions)
2. fine-tune the BLIP model <br />
[Fine-tined model](https://huggingface.co/alesanm/blip-image-captioning-base-fashionimages-finetuned)
3. evaluate the quality of the generated descriptions with the help of fine-tuned BLIP using ground truth GPT3 descriptions <br />
[Test dataset](https://huggingface.co/datasets/alesanm/chanel_short_descriptions)

## Results

![Metrics](https://github.com/aimedvedeva/AutomateLabling/blob/main/log.jpg)
We used two metrics to compare our descriptions with ground truth. Suggested to rely on cosine similarity between BERT embeddings, because it takes into account the meaning of the text rather than a set of symbols as BLUE. Both approaches are almost similarly good in terms of cosine similarity metric, however, there are two important things that should be treated with attention: </br>
1. Ground truth descriptions were generated via the Openai GPT3 model. Although a good prompt was used, the quality of the output wasn't ideal. Sometimes it was too general. Since ground truth data is based on which we evaluate the quality of our models, probably, next time we should go to Toloka and ask real people to provide us with higher-quality descriptions.
2. Fashion photos sometimes do not differ from each other enough inside one brand, e.g. Chanel. Sometimes, it does, e.g. Balenciaga. That is why we decided in the second experiment to fine-tune the BLIP model with 140 Balenciaga fashion photos, rather than with Chanel to be sure that we introduce to the model a more comprehensive range of possible options.
3. Also we should mention that ideally for more confident evaluation metric experiments should be run numerous times to acquire much more data and evaluate confident interval for the man of the metric per each experiment.
4. For the first experiment we used an ensemble of pre-trained models such as general BLIP + One-Shot Image Classification. For the second model, we created a list of queries with possible answers. Then, the model gave us the most probable answer per each query from the suggested list.  The important thing is that this list and, especially, the answer options should be revised very attentively because the model's quality hugely depends on it.
