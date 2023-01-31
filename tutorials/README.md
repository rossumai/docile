# DocILE tutorials

This folder contains a few examples/tutorials to get familiar with the repository and the dataset:

* [Quickstart](quickstart.md): Showcase the basic functionality of the repo -- load dataset, create dummy predictions, run & visualize evaluation.
* [Load and Sample Dataset](load_and_sample_dataset.md): Useful overview of how to load the dataset, working efficiently with small & large datasets and their subsets.

Also check [Dataset Browser Notebook](../docile/tools/dataset_browser.ipynb) to browse through the data.

## Editing the tutorials

To edit the tutorials follow these steps (to prevent committing large files in git):

* Make the changes in the jupyter notebook and run all cells.
* Export the markdown file from jupyter lab with `File -> Save and Export Notebook As... -> Markdown`
* Add `*Generated from [ntb.ipynb](ntb.ipynb)*` at the top of the markdown file. Limit other changes in the markdown output to minimum so that future edits of the tutorials remain simple.
* Do not add images to git. Instead create a pull request and edit the markdown files directly on github, uploading the images there.
* Clear all outputs in jupyter notebook before adding it to git.
