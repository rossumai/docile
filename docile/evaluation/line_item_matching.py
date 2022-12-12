import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx

from docile.dataset import BBox, Field
from docile.evaluation.pcc import PCCSet
from docile.evaluation.pcc_field_matching import FieldMatching, MatchedPair, get_matches


class LineItemsGraph:
    """
    Class representing the bipartite graph between prediction and gold line items.

    Each edge holds the information about the field matching between the two line items. The graph
    is used to find the maximum matching between line items that is maximizing the overall number
    of matched fields.
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
        self.G.add_edge(
            (0, pred_li_i),
            (1, gold_li_i),
            weight=-len(field_matching.matches),
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
            if pred_node[0] == 0 and self.G.edges[pred_node, gold_node]["weight"] != 0
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
        return (
            FieldMatching(matches=[], false_positives=predictions, false_negatives=annotations),
            {},
        )

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
                field_matching = FieldMatching(
                    matches=[], false_positives=preds, false_negatives=golds
                )
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
    matched_pairs: List[MatchedPair] = []
    false_positives: List[Field] = []
    false_negatives: List[Field] = []
    for pred_li_i, preds in pred_line_items.items():
        if pred_li_i not in maximum_matching:
            false_positives.extend(preds)
            continue

        gold_li_i = maximum_matching[pred_li_i]
        field_matching = line_items_graph.get_pair_field_matching(
            pred_li_i=pred_li_i, gold_li_i=gold_li_i
        )
        matched_pairs.extend(field_matching.matches)
        false_positives.extend(field_matching.false_positives)
        false_negatives.extend(field_matching.false_negatives)

    for gold_li_i, golds in gold_line_items.items():
        if gold_li_i not in maximum_matching.values():
            false_negatives.extend(golds)

    lir_field_matching = FieldMatching(
        matches=matched_pairs,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    return lir_field_matching, maximum_matching
