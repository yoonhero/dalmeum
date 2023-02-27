from torchvision import transforms as T
import numpy as np
import pandas as pd
import random
from io import BytesIO
import base64
from PIL import Image

from img2vec import Img2Vec
import utils

img2vec = Img2Vec(True, transform=T.Compose([
        T.ToTensor(),
        T.Resize((224, 224))
    ]))
# labels = utils.get_labels()
embedding_db = np.load("./tmp/embedding.npy")
# player_db = pd.read_csv("./dataset/train_male.csv")
animal_db = pd.read_csv("./dataset/animal.csv")


def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

def inference(image):
    result_embedding = img2vec.get_vec(image, single_image=True).squeeze(0) 
    inference_embedding = result_embedding.detach().numpy()

    metrics = ["l2norm", "l1norm"]
    metric = random.choice(metrics)
    
    metric_func = lambda: None
    l2_norm = lambda x, y: np.sum((x-y)**2)
    l1_norm = lambda x, y: np.sum(np.abs(x-y))
    if metric == "l2norm":
        metric_func = l2_norm
    elif metric == "l1norm":
        metric_func = l1_norm

    embedding = [metric_func(emb, inference_embedding) for emb in embedding_db] 
    embedding = embedding - max(embedding) # prevent inf 
    count = np.exp(embedding)
    norm = count / np.sum(count)
    real_count = 1 - norm
    prob = real_count / np.sum(real_count)
    # result = np.argmin(embedding)
    result = np.argmax(np.random.multinomial(1, prob))
    confidence = prob[result] * 100

    # idx = labels[result] 
    # t_raw = player_db.iloc[idx, :]
    # player_name, player_img_url = t_raw["short_name"], t_raw["player_face_url"]
    animal_img_url = animal_db.iloc[result]["url"]
    bytes_img = img_to_base64_str(Image.open(animal_img_url))

    # category = animal_img_url.split("_")[-2]
    # filename = animal_img_url.split("\\")[-1]
    # url = f"""https://storage.googleapis.com/kagglesdsdata/datasets/667852/1176357/afhq/train/{category}/{filename}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230227%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230227T082435Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=126808cd1ea9ec382334f72357590f94413bc372a94bcfc8a748e41186f0da6546b38a62f31793dbf261801b8fbdb4816af57b993161d284f623a4a2f2b6ba99f61bac84b1341f3b1d7bae1da92c39ab72470a9c1a1007c9c344902e533c40419d667cb5794e5634d974497f9c5307f1179308f9f64be1ef468a205f8baf909a388b12bf4f9ed3963bbffcc74c18623c363b834c646bad8e98f8a729f54a442e13b140a0d4986fae3d91c79bbfe0933e0353c3e09a2511a39593bf16793b2e7ef0ad1759f22bc1067ae931f97fed95e4c0ab86bcd6179f875f05516f31510b234004f384d3b4f2fe7e81fb2ec9198c6b0add7efb8e014e0acb00e25848a810b4"""

    return bytes_img, confidence


if __name__ == "__main__":
    print(inference(Image.open("./test/test_img.JPG")))
    
    
    
