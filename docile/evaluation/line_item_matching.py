from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx

from docile.dataset import BBox, Field
from docile.evaluation.pcc import PCCSet
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches


class LineItemsGraph:
    """
    Class representing the bipartite graph between prediction and gold line items.

    Each edge holds the information about the field matching between the two line items. The graph
    is used to find the maximum matching between line items that is maximizing the overall number
    of matched fields (after excluding predictions with flag `use_only_for_ap`).
    """

    def __init__(
        self, pred_line_item_ids: Sequence[int], gold_line_item_ids: Sequence[int]
    ) -> None:
        self.G = networkx.Graph()
        self.pred_nodes = [(0, i) for i in pred_line_item_ids]
        self.gold_nodes = [(1, i) for i in gold_line_item_ids]
        self.G.add_nodes_from(self.pred_nodes)
        self.G.add_nodes_from(self.gold_nodes)

    def add_edge(self, pred_li_i: int, gold_li_i: int, field_matching: FieldMatching) -> None:
        # Only count predictions without the `use_only_for_ap` flag.
        main_prediction_matches = len(field_matching.filter(exclude_only_for_ap=True).matches)
        self.G.add_edge(
            (0, pred_li_i),
            (1, gold_li_i),
            weight=-main_prediction_matches,
            field_matching=field_matching,
        )

    def get_pair_field_matching(self, pred_li_i: int, gold_li_i: int) -> FieldMatching:
        return self.G.edges[(0, pred_li_i), (1, gold_li_i)]["field_matching"]

    def get_maximum_matching(self) -> Dict[int, int]:
        """
        Return the maximum matching between the prediction and gold line items.

        Returns
        -------
        Mapping from pred line item ids to gold line item ids. Only pairs that have non-empty field
        matching are returned.
        """
        maximum_matching = networkx.algorithms.bipartite.minimum_weight_full_matching(
            self.G, self.pred_nodes
        )
        # Each node in the graph is identified as (0, i), resp. (1, i) based on which side of
        # bipartition the node is in (p=0 .. prediction, p=1 .. gold).
        return {
            pred_node[1]: gold_node[1]  # remove the bipartition id part
            for pred_node, gold_node in maximum_matching.items()
            # keep only edges from pred to gold and if they have non-empty field matching
            if pred_node[0] == 0
            and len(self.get_pair_field_matching(pred_node[1], gold_node[1]).matches) != 0
        }


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
    pcc_set: PCCSet,
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
        return (FieldMatching.empty(predictions, annotations), {})

    pred_line_items = defaultdict(list)
    pred_i_to_index_in_li = {}
    for pred_i, pred in enumerate(predictions):
        li_i = _get_line_item_id(pred)
        pred_i_to_index_in_li[pred_i] = len(pred_line_items[li_i])
        pred_line_items[li_i].append(pred)

    gold_line_items = defaultdict(list)
    for gold in annotations:
        li_i = _get_line_item_id(gold)
        gold_line_items[li_i].append(gold)

    # We precompute the covering bbox of each line item. This is used to speedup the computation
    # since prediction/gold line items that are completely disjoint cannot have any matches.
    pred_li_bbox = {
        li_i: _get_covering_bbox(_place_bbox_in_document(f.bbox, f.page) for f in fields)
        for li_i, fields in pred_line_items.items()
    }
    gold_li_bbox = {
        li_i: _get_covering_bbox(_place_bbox_in_document(f.bbox, f.page) for f in fields)
        for li_i, fields in gold_line_items.items()
    }

    # Construct complete bipartite graph between pred and gold line items.
    line_items_graph = LineItemsGraph(list(pred_line_items.keys()), list(gold_line_items.keys()))
    for pred_li_i, preds in pred_line_items.items():
        for gold_li_i, golds in gold_line_items.items():
            # If the bboxes covering the line items are disjoint, there cannot be any field matches
            if not pred_li_bbox[pred_li_i].intersects(gold_li_bbox[gold_li_i]):
                field_matching = FieldMatching.empty(preds, golds)
            else:
                field_matching = get_matches(
                    predictions=preds,
                    annotations=golds,
                    pcc_set=pcc_set,
                    iou_threshold=iou_threshold,
                )

            line_items_graph.add_edge(
                pred_li_i=pred_li_i, gold_li_i=gold_li_i, field_matching=field_matching
            )

    maximum_matching = line_items_graph.get_maximum_matching()

    # Construct matching on the field level from the line item matching.
    ordered_predictions_with_match: List[Tuple[Field, Optional[Field]]] = []
    for pred_i, pred in enumerate(predictions):
        pred_li_i = _get_line_item_id(pred)
        if pred_li_i not in maximum_matching:
            ordered_predictions_with_match.append((pred, None))
            continue

        gold_li_i = maximum_matching[pred_li_i]
        field_matching = line_items_graph.get_pair_field_matching(
            pred_li_i=pred_li_i, gold_li_i=gold_li_i
        )
        pred_i_in_li = pred_i_to_index_in_li[pred_i]
        ordered_predictions_with_match.append(
            field_matching.ordered_predictions_with_match[pred_i_in_li]
        )

    false_negatives: List[Field] = []
    maximum_matching_gold_to_pred = {v: k for k, v in maximum_matching.items()}
    for gold_li_i, golds in gold_line_items.items():
        if gold_li_i in maximum_matching_gold_to_pred:
            pred_li_i = maximum_matching_gold_to_pred[gold_li_i]
            field_matching = line_items_graph.get_pair_field_matching(
                pred_li_i=pred_li_i, gold_li_i=gold_li_i
            )
            false_negatives.extend(field_matching.false_negatives)
        else:
            false_negatives.extend(golds)

    lir_field_matching = FieldMatching(ordered_predictions_with_match, false_negatives)
    return lir_field_matching, maximum_matching
