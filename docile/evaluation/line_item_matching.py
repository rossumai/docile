import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx

from docile.dataset import PCC, BBox, Field
from docile.evaluation.pcc_field_matching import FieldMatching, MatchedPair, get_matches


def _get_line_item_id(field: Field) -> int:
    if field.line_item_id is None:
        raise ValueError(f"No line item ID specified for LIR field {field}")
    return field.line_item_id


def _place_bbox_in_document(bbox: BBox, page: int) -> BBox:
    """
    Return a bbox where y coordinates are in range [page, page+1].

    This way it is possible to work with bboxes from different pages on the same document.
    """
    return BBox(left=bbox.left, top=bbox.top + page, right=bbox.right, bottom=bbox.bottom + page)


def _get_covering_bbox(bboxes: Iterable[BBox]) -> BBox:
    """
    Return the minimum bbox covering all input bboxes.

    Raises an exception if there are no bboxes on input.
    """
    lefts, tops, rights, bottoms = zip(*(bbox.to_tuple() for bbox in bboxes))
    return BBox(min(lefts), min(tops), max(rights), max(bottoms))


def get_lir_matches(
    predictions: Sequence[Field],
    annotations: Sequence[Field],
    pccs: Sequence[PCC],
    iou_threshold: float = 1,
) -> Tuple[FieldMatching, Dict[int, int]]:
    """
    Get matching of line item fields in the document.

    This is similar to pcc_field_matching.get_matches but first corresponding line items are found
    with maximum matching while optimizing the total number of matched predictions, irrespective of
    their score.

    Returns
    -------
    Matching of line item fields and used maximum matching between line item ids (prediction to gold).
    """
    if len(predictions) == 0 or len(annotations) == 0:
        return FieldMatching(matches=[], extra=predictions, misses=annotations), {}

    pred_line_items = {
        li_i: list(group)
        for li_i, group in itertools.groupby(
            sorted(predictions, key=_get_line_item_id), key=_get_line_item_id
        )
    }
    gold_line_items = {
        li_i: list(group)
        for li_i, group in itertools.groupby(
            sorted(annotations, key=_get_line_item_id), key=_get_line_item_id
        )
    }

    pred_li_bbox = {
        li_i: _get_covering_bbox(_place_bbox_in_document(f.bbox, f.page) for f in fields)
        for li_i, fields in pred_line_items.items()
    }
    gold_li_bbox = {
        li_i: _get_covering_bbox(_place_bbox_in_document(f.bbox, f.page) for f in fields)
        for li_i, fields in gold_line_items.items()
    }

    # Construct complete bipartite graph between pred and gold line items. Weight of edges is
    # (minus) number of the matched pairs. Then we can use minimum weight full matching to find the
    # maximum matching.
    G = networkx.Graph()
    pred_nodes = [(0, i) for i in pred_line_items.keys()]
    gold_nodes = [(1, i) for i in gold_line_items.keys()]
    G.add_nodes_from(pred_nodes)
    G.add_nodes_from(gold_nodes)
    for pred_li_i, preds in pred_line_items.items():
        for gold_li_i, golds in gold_line_items.items():
            # If the bboxes covering the line items are disjoint, there cannot be any field matches
            if not pred_li_bbox[pred_li_i].intersects(gold_li_bbox[gold_li_i]):
                field_matching = FieldMatching(matches=[], extra=preds, misses=golds)
            else:
                field_matching = get_matches(
                    predictions=preds, annotations=golds, pccs=pccs, iou_threshold=iou_threshold
                )

            G.add_edge(
                (0, pred_li_i),
                (1, gold_li_i),
                weight=-len(field_matching.matches),
                field_matching=field_matching,
            )

    maximum_matching = networkx.algorithms.bipartite.minimum_weight_full_matching(G, pred_nodes)
    maximum_pred_matching = {
        pred_node[1]: gold_node[1]
        for pred_node, gold_node in maximum_matching.items()
        if pred_node[0] == 0 and G.edges[pred_node, gold_node]["weight"] != 0
    }

    matched_pairs: List[MatchedPair] = []
    extra: List[Field] = []
    misses: List[Field] = []
    for pred_li_i, preds in pred_line_items.items():
        if pred_li_i not in maximum_pred_matching:
            extra.extend(preds)
            continue

        gold_li_i = maximum_pred_matching[pred_li_i]
        field_matching = G.edges[(0, pred_li_i), (1, gold_li_i)]["field_matching"]
        matched_pairs.extend(field_matching.matches)
        extra.extend(field_matching.extra)
        misses.extend(field_matching.misses)

    for gold_li_i, golds in gold_line_items.items():
        if gold_li_i not in maximum_pred_matching.values():
            misses.extend(golds)

    lir_field_matching = FieldMatching(
        matches=matched_pairs,
        extra=extra,
        misses=misses,
    )
    return lir_field_matching, maximum_pred_matching
