import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import time
from multiprocessing import Pool
import os

import utils

class Img2Vec():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    def __init__(self, cuda:bool, model:str="resnet18", layer:str="default", layer_output_size:int=512, transform=T.ToTensor()):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transform

    def get_vec(self, img, single_image=False):
        if not single_image:
            a = [self.transform(im) for im in img]
            images = torch.stack(a, 0).to(self.device)
            
        else:
            a = self.transform(img)
            # print(a.shape)
            images = a.unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(images.shape[0], self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(images)
        h.remove()

        return my_embedding

    def _get_model_and_layer(self, model_name, layer):
        model = getattr(models, model_name)(pretrained=True)
        if layer == "default":
            layer = model._modules.get("avgpool")
            self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
        else:
            layer = model._modules.get(layer)

        return model, layer

def load_internet_image(data):
    idx, url = set(data)
    if not isinstance(idx, int):
        idx, url = url, idx
    # idx += start_idx
    # print(start_idx)
    try:
        print(idx, url)
        image = utils.load_image_with_url(url)
        return (idx, image)
    except: return (idx, "")

def load_local_image(data):
    idx, url = set(data)
    if not isinstance(idx, int):
        idx, url = url, idx
    try:
        print(idx, url)
        image = Image.open(url).convert("RGB")
        return (idx, image)
    except: return (idx, "")

def make_embedding_db(dataset: str, download=True):
    df = pd.read_csv(dataset)
    batch_size = 1024
    nb_epoch = len(df) // 128
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224))
    ])
    img2vec = Img2Vec(cuda=True, transform=transform)

    image_embedding_result = np.array([])
    embedding_idxs = []

    for i in range(nb_epoch):
        start_idx = i*batch_size
        end_idx = (i+1) * batch_size
        
        # target = df.player_face_url[start_idx:end_idx]
        target = df.url[start_idx:end_idx]
        
        print("Loading Image start")
        start = time.time()
        # contents = [(idx, Image.open(BytesIO(requests.get(t).content)).convert("RGB")) for idx, t in enumerate(target)]

        pool = Pool()
        if download:
            contents = pool.map(load_internet_image, enumerate(target, start_idx))
        else:
            contents = pool.map(load_local_image, enumerate(target, start_idx))
        pool.close()
        pool.join()
        contents = list(filter(lambda x: x[1]!="", contents))

        idxs = [i for (i, _) in contents]
        embedding_idxs.extend(idxs)
        images = [image for (_, image) in contents]
        # for image_url in target:
        #     response = requests.get(image_url)
        #     img = Image.open(BytesIO(response.content)).convert("RGB")
        #     images.append(img)
        # tensor_images = torch.tensor(imagesa
        end = time.time()
        print(f"Loading within {(end - start):.2f}sec")
        
        img_tensor_vec = img2vec.get_vec(images)
        img_vec = img_tensor_vec.detach().numpy()
        # image_embedding_result.extend(img_vec)    
        if image_embedding_result.shape == (0, ):
            image_embedding_result = img_vec
        else:
            image_embedding_result = np.concatenate((image_embedding_result, img_vec), axis=0)
        
        print(image_embedding_result.shape)

    return image_embedding_result, embedding_idxs


if __name__ == "__main__":
    embedding, idxs = make_embedding_db("./dataset/animal.csv", download=False)
    # embedding, idxs = animal_make_embedding_db("./dataset/animal.csv")

    # df = pd.DataFrame([[i, j] for i, j in zip(idxs, embedding)], columns=["index", "embedding"])
    # df = pd.DataFrame(idxs, columns=["index"])
    os.makedirs("./tmp", exist_ok=True)
    np.save("./tmp/embedding.npy", embedding)
    import utils
    utils.set_label_pickle(idxs)
    # df.to_csv("./tmp/embedding_idx.csv")

