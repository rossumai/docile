from typing import List, Optional, Sequence, Tuple

from docile.dataset import CachingConfig, Dataset

NamedRange = Tuple[str, Tuple[int, Optional[int]]]


def size_in_range(size: int, size_range: Tuple[int, Optional[int]]) -> bool:
    """
    Test if the cluster size lies in the given range.

    Parameters
    ----------
    size
        Size of the cluster.
    size_range
        A range [start, end], start and end is inclusive. If end is None, only start is tested.
    """
    return size_range[0] <= size and (size_range[1] is None or size <= size_range[1])


def get_x_shot_subsets(
    test: Dataset, train: Dataset, named_ranges: Sequence[NamedRange]
) -> List[Dataset]:
    """
    Find subsets of test corresponding to x-shot clusters.

    For each given range, find a subset of `test` documents from clusters with `x` samples in
    `train` where `x` lies in that range.

    Parameters
    ----------
    test
        Dataset used for evaluation, find subsets of this dataset.
    train
        Dataset that contains all the documents seen during training. Then `x`-shot cluster is
        defined as a cluster that has `x` documents in `train`. Notice that in docile, `trainval`
        is considered to be the training set for `test` as both `train` and `val` splits can be
        used during training (even if just for validation).
    named_ranges
        Sequence of tuples of names and ranges representing the cluster sizes in `train` to fall in
        the corresponding subset.

    Returns
    -------
    A sequence of datasets, one for each named range, that are subsets of `test` and whose clusters
    have the correct number of documents in `train`.
    """
    # First parse the range names to raise an exception early if they are not valid.
    test_cluster_ids = {doc.annotation.cluster_id for doc in test}
    range_name_to_documents = {range_name: [] for range_name, _ in named_ranges}
    for cluster_id in test_cluster_ids:
        train_documents = [doc for doc in train if doc.annotation.cluster_id == cluster_id]
        test_documents = test.get_cluster(cluster_id).documents
        for range_name, size_range in named_ranges:
            if size_in_range(len(train_documents), size_range):
                range_name_to_documents[range_name].extend(test_documents)
    return [
        Dataset.from_documents(f"{test.split_name}-{range_name}-shot", documents)
        for range_name, documents in range_name_to_documents.items()
    ]


def get_synthetic_subset(test: Dataset, synthetic_sources: Dataset) -> Optional[Dataset]:
    """
    Get subset of test corresponding to clusters with synthetic data available.

    Returns
    -------
    Subset with documents in clusters that have synthetic data available. Returns None if there are
    no such documents in `test`.
    """
    synthetic_cluster_ids = {doc.annotation.cluster_id for doc in synthetic_sources}
    documents_synth = [doc for doc in test if doc.annotation.cluster_id in synthetic_cluster_ids]
    if len(documents_synth) == 0:
        return None
    return Dataset.from_documents(f"{test.split_name}-synth-clusters-only", documents_synth)


def get_evaluation_subsets(
    test: Dataset, named_ranges: Sequence[NamedRange], synthetic: bool
) -> List[Dataset]:
    """
    Find subsets corresponding to x-shot and/or synthetic clusters.

    When named_ranges is given, finds x-shot clusters with respect to `trainval` for `test` and
    w.r.t. to `train` for `val`. When synthetic is true, for each subset a variant is added that
    includes only documents in clusters with synthetic data available.

    Parameters
    ----------
    named_ranges
        Sequence of tuples of names and ranges representing the cluster sizes in `train` to fall in
        the corresponding subset.
    synthetic
        If true, generate subsets of documents belonging to clusters with synthetic data.

    Returns
    -------
    List of dataset subsets, without the full `test` dataset.
    """
    if len(named_ranges) == 0 and not synthetic:
        return []

    # Add the full dataset to the list so that the synth version is generated below. It is removed
    # from the output at the end.
    subsets = [test]
    if len(named_ranges) > 0:
        if test.split_name == "test":
            train_split_name = "trainval"
        elif test.split_name == "val":
            train_split_name = "train"
        else:
            raise ValueError(f"No default corresponding train dataset for {test}")

        train = Dataset(
            train_split_name, test.data_paths, load_ocr=False, cache_images=CachingConfig.OFF
        )
        subsets.extend(get_x_shot_subsets(test, train, named_ranges))

    if synthetic:
        new_subsets = []
        synthetic_sources = Dataset(
            "synthetic-sources", test.data_paths, load_ocr=False, cache_images=CachingConfig.OFF
        )
        for subset in subsets:
            new_subsets.append(subset)
            synthetic_subset = get_synthetic_subset(subset, synthetic_sources)
            if synthetic_subset is not None:
                new_subsets.append(synthetic_subset)
        subsets = new_subsets

    # Remove the full test set from the output
    return subsets[1:]
