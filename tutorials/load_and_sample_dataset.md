*Generated from [load_and_sample_dataset.ipynb](load_and_sample_dataset.ipynb)*

# Tutorial: Load and Sample Datasets

In this tutorial you will learn what are the different ways to load datasets, subsample them.

To run all cells, you need the `annotated-trainval` and `synthetic` datasets unzipped in `data/docile` and `annotated-trainval.zip` file in `data/`.


```python
from pathlib import Path
from docile.dataset import CachingConfig, Dataset

DATASET_PATH = Path("/app/data/docile/")
DATASET_PATH_ZIP = Path("/app/data/annotated-trainval.zip")
```

## Load from folder with unzipped dataset


```python
val = Dataset("val", DATASET_PATH)
```

    Loading documents for docile:val: 100%|██████████| 500/500 [00:03<00:00, 145.04it/s]


## Load from zip

Dataset can be loaded directly from zip as well but image caching to disk must be turned off.


```python
val = Dataset("val", DATASET_PATH_ZIP, cache_images=CachingConfig.OFF)
```

    Loading documents for annotated-trainval.zip:val: 100%|██████████| 500/500 [00:01<00:00, 281.09it/s]


## Preloading document resources to memory and image caching

By default, dataset is loaded with these settings:

* annotations and pre-computed OCR are loaded to memory
* images generated from PDFs are cached to disk (for faster access of the images in future iterations)

Below you see options how to change this default behaviour, which is especially useful for large datasets.

**Only load annotations, not pre-computed OCR**


```python
train = Dataset("train", DATASET_PATH, load_ocr=False)
```

    Loading documents for docile:train: 100%|██████████| 5180/5180 [00:08<00:00, 634.19it/s]


**Postpone loading of document resources**


```python
# Do not preload document resources
synthetic = Dataset("synthetic", DATASET_PATH, load_annotations=False, load_ocr=False, cache_images=CachingConfig.OFF)
# You can load part of the dataset later
synthetic_sample = synthetic[:100].load()
# And release it from memory later
synthetic_sample.release()
```

    Initializing documents for docile:synthetic: 100%|██████████| 100000/100000 [00:03<00:00, 31701.69it/s]
    Loading documents for docile:synthetic[:100]: 100%|██████████| 100/100 [00:00<00:00, 220.34it/s]


**Cache images to both disk and memory**


```python
# Cache images also in memory. Make sure you have enough RAM memory to do this!
# Images are not loaded to memory right away but only after first
train = Dataset("train", DATASET_PATH, cache_images=CachingConfig.DISK_AND_MEMORY)
```

    Loading documents for docile:train: 100%|██████████| 5180/5180 [00:42<00:00, 122.97it/s]


## Sample and chunk documents

For experiments and to work with large datasets, it can be useful to take samples of the datasets.

For this, you can use slicing `[start:end:step]`, `.sample()`, `.get_cluster()` or `.from_documents()` methods.


```python
synthetic = Dataset("synthetic", DATASET_PATH, load_annotations=False, load_ocr=False, cache_images=CachingConfig.OFF)
trainval = Dataset("trainval", DATASET_PATH, load_annotations=False, load_ocr=False, cache_images=CachingConfig.OFF)
```

    Initializing documents for docile:synthetic: 100%|██████████| 100000/100000 [00:02<00:00, 41461.39it/s]


**Slicing**


```python
# Synthetic document has 100 chunks of 1000 documents from the same template document, so the
# following line selects 1 document for each template document:
synthetic_slice = synthetic[8::1000]
print(synthetic_slice.docids[:5])
```

    ['synthetic-02a50adccac54a569011f167-008', 'synthetic-04cb9c50d4c949598689ea6f-008', 'synthetic-062d94841d1649a5b5a4e720-008', 'synthetic-1239abc177b245dfae0145d9-008', 'synthetic-141c3120d0f54beeb771d411-008']


**Random sample**


```python
synthetic_sample = synthetic.sample(5)
print(synthetic_sample)
print(synthetic_sample.docids)
```

    Dataset(docile:synthetic[sample(5,seed=15980735705623844408)])
    ['synthetic-29639c44ca7f43ad8739bd62-825', 'synthetic-698c58c5ece74fc9971d2f64-769', 'synthetic-f079d766ab2841b4956aef79-064', 'synthetic-89679c3d35524159b431d16a-816', 'synthetic-ef9e2ae7d518402982dbcb99-384']


**Documents belonging to the same cluster**


```python
trainval_cluster = trainval.get_cluster(synthetic[0].annotation.cluster_id)
print(f"Found {len(trainval_cluster)} documents in {trainval_cluster}.")

from PIL import Image

print("Showing 10 images from the cluster:")
imgs = [doc.page_image(page=0, image_size=(None, 100)) for doc in trainval_cluster[:10]]
concat_img = Image.new("RGB", (sum(img.width for img in imgs), 100))
start_from = 0
for img in imgs:
    concat_img.paste(img, (start_from, 0))
    start_from += img.width
concat_img
```

    Found 29 documents in Dataset(docile:trainval[cluster_id=765]).
    Showing 10 images from the cluster:






![output_24_1](https://user-images.githubusercontent.com/1220288/215839029-2b7fb0ff-8c36-4201-a69d-60760d17a300.png)



**Using custom filter**


```python
trainval_ucsf = Dataset.from_documents("trainval-ucsf", [doc for doc in trainval if doc.annotation.source == "ucsf"])
trainval_pif = Dataset.from_documents("trainval-pif", [doc for doc in trainval if doc.annotation.source == "pif"])
print(f"{trainval_ucsf} with {len(trainval_ucsf)} documents")
print(f"{trainval_pif} with {len(trainval_pif)} documents")
```

    Dataset(docile:trainval-ucsf) with 2645 documents
    Dataset(docile:trainval-pif) with 3035 documents


## Chunk dataset into parts with the same number of pages

Create dataset chunks that have a limited number of pages. This can be especially useful for large datasets, such as the unlabeled dataset.


```python
from typing import Iterable

def chunk_dataset(dataset: Dataset, max_pages_per_chunk: int) -> Iterable[Dataset]:
    start_doc_i = 0
    pages = 0
    for doc_i, document in enumerate(dataset.documents):
        documents = doc_i - start_doc_i + 1
        pages += document.page_count
        if doc_i > start_doc_i and pages > max_pages_per_chunk:
            yield dataset[start_doc_i:doc_i]
            start_doc_i = doc_i
            pages = document.page_count
    yield dataset[start_doc_i:]
```


```python
trainval = Dataset("trainval", DATASET_PATH, load_annotations=False, load_ocr=False, cache_images=CachingConfig.OFF)

max_pages_per_chunk = 2000

for chunk in chunk_dataset(trainval, max_pages_per_chunk):
    print(f"{chunk}, pages: {chunk.total_page_count()}")
    chunk.load(annotations=True, ocr=False)
    # ... work with the chunk here ...
    chunk.release() # don't forget to free up the memory
```

    Dataset(docile:trainval[0:1530]), pages: 1999


    Loading documents for docile:trainval[0:1530]: 100%|██████████| 1530/1530 [00:02<00:00, 660.47it/s]


    Dataset(docile:trainval[1530:3051]), pages: 2000


    Loading documents for docile:trainval[1530:3051]: 100%|██████████| 1521/1521 [00:02<00:00, 665.22it/s]


    Dataset(docile:trainval[3051:4584]), pages: 2000


    Loading documents for docile:trainval[3051:4584]: 100%|██████████| 1533/1533 [00:02<00:00, 654.72it/s]


    Dataset(docile:trainval[4584:]), pages: 1395


    Loading documents for docile:trainval[4584:]: 100%|██████████| 1096/1096 [00:01<00:00, 613.10it/s]
