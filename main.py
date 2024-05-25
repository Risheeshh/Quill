import cv2
import tensorflow as tf
import numpy as np
from mltu.utils.text_utils import ctc_decoder, get_cer
import pandas as pd
import tqdm
import typing
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("handwriting_model/configs.yaml")

    model = ImageToWordModel(model_path="handwriting_model/model.onnx", char_list=configs.vocab)

    image_path=input("Enter the input image path \n")
    label=input("Enter label\n")
    image = cv2.imread(image_path.replace("\\", "/"))

    prediction_text = model.predict(image)
    cer=get_cer(prediction_text, label)
    print(f"Image: {image_path}, Prediction: {prediction_text}, CER: {cer}")

    # resize by 4x
    image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

