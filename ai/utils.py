import pickle
from PIL import Image
from io import BytesIO
import requests

def set_label_pickle(labels):
    with open("./tmp/label", "wb") as fp:
        pickle.dump(labels, fp)

def get_labels():
    with open("./tmp/label", "rb" ) as  fp:
        return pickle.load(fp)
    
def load_image_with_url(url):
    return Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    