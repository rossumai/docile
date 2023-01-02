import itertools
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from docile.dataset import BBox, Document, Field
from docile.dataset.types import OptionalImageSize
from docile.evaluation import PCC, get_document_pccs
from docile.evaluation.line_item_matching import get_lir_matches
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches

RGBA = Tuple[int, int, int, int]
RED = (255, 0, 0, 255)
GREEN = (0, 128, 0, 255)
BLUE = (0, 0, 255, 255)


def draw_field(
    draw: ImageDraw.ImageDraw, field: Field, outline: RGBA, width: int, fill: Optional[RGBA] = None
) -> None:
    scaled_bbox = field.bbox.to_absolute_coords(*draw.im.size)  # type: ignore
    draw.rectangle(scaled_bbox.to_tuple(), outline=outline, width=width, fill=fill)


def draw_pcc(
    draw: ImageDraw.ImageDraw, pcc: PCC, radius: int, outline: RGBA, width: int, fill: RGBA
) -> None:
    img_width, img_height = draw.im.size  # type: ignore
    x, y = (pcc.x * img_width, pcc.y * img_height)
    ellipse = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(ellipse, outline=outline, width=width, fill=fill)


def page_image_with_fields(
    page_img: Image.Image,
    kile_fields: Sequence[Field],
    li_fields: Sequence[Field],
    ocr_words: Sequence[Field],
) -> Image.Image:
    """Return page image with bboxes representing fields."""
    draw_img = page_img.copy()
    draw = ImageDraw.Draw(draw_img, "RGBA")

    for fields, outline, width in [
        (kile_fields, GREEN, 2),
        (li_fields, BLUE, 2),
        (ocr_words, RED, 1),
    ]:
        for field in fields:
            draw_field(draw, field, outline, width)

    return draw_img


def page_image_with_all_fields(
    document: Document,
    page: int,
    image_size: OptionalImageSize = (None, None),
    show_ocr_words: bool = False,
    snapped: bool = True,
) -> Image.Image:
    kile_fields = [f for f in document.annotation.fields if f.page == page]
    li_fields = [f for f in document.annotation.li_fields if f.page == page]
    return page_image_with_fields(
        page_img=document.page_image(page, image_size),
        kile_fields=kile_fields,
        li_fields=li_fields,
        ocr_words=document.ocr.get_all_words(page, snapped=snapped) if show_ocr_words else [],
    )


def page_image_with_matching(
    document: Document,
    page: int,
    field_matching: FieldMatching,
    image_size: OptionalImageSize = (None, None),
) -> Image.Image:
    page_img = document.page_image(page, image_size)
    draw_img = page_img.copy()
    draw = ImageDraw.Draw(draw_img, "RGBA")

    for fields, outline, width, fill in [
        ([match.pred for match in field_matching.matches], GREEN, 3, (0, 0, 255, 96)),
        ([match.gold for match in field_matching.matches], GREEN, 3, (255, 255, 0, 96)),
        (field_matching.false_negatives, RED, 3, (0, 0, 255, 96)),
        (field_matching.false_positives, RED, 3, (255, 255, 0, 96)),
    ]:
        for field in fields:
            draw_field(draw, field, outline, width, fill)

    pcc_radius = 2
    pcc_width = 2
    pcc_outline = (0, 0, 0, 128)

    for pcc in get_document_pccs(document):
        if pcc.page != page:
            continue

        for match in field_matching.matches:
            if match.pred.page != page:
                continue
            if match.pred.bbox.intersects(BBox(left=pcc.x, top=pcc.y, right=pcc.x, bottom=pcc.y)):
                draw_pcc(
                    draw=draw,
                    pcc=pcc,
                    radius=pcc_radius,
                    outline=pcc_outline,
                    width=pcc_width,
                    fill=(0, 255, 0, 64),
                )
                break
        else:
            for field in itertools.chain(
                field_matching.false_negatives, field_matching.false_positives
            ):
                if field.page != page:
                    continue
                if field.bbox.intersects(BBox(left=pcc.x, top=pcc.y, right=pcc.x, bottom=pcc.y)):
                    draw_pcc(
                        draw=draw,
                        pcc=pcc,
                        radius=2,
                        outline=pcc_outline,
                        width=pcc_width,
                        fill=(255, 0, 0, 64),
                    )
                    break
            else:
                draw_pcc(
                    draw=draw,
                    pcc=pcc,
                    radius=2,
                    outline=pcc_outline,
                    width=pcc_width,
                    fill=(0, 0, 255, 64),
                )

    return draw_img


def page_image_with_predictions(
    document: Document,
    page: int,
    kile_predictions: Sequence[Field],
    lir_predictions: Sequence[Field],
    iou_threshold: float = 1.0,
    image_size: OptionalImageSize = (None, None),
) -> Image.Image:
    if sum(len(preds) > 0 for preds in [kile_predictions, lir_predictions]) != 1:
        raise ValueError("Exactly one of kile_predictions/lir_predictions should be non-empty")

    pcc_set = get_document_pccs(document)

    if kile_predictions:
        field_matching = get_matches(
            predictions=kile_predictions,
            annotations=document.annotation.fields,
            pcc_set=pcc_set,
            iou_threshold=iou_threshold,
        )
    else:
        field_matching, _line_item_id_matching = get_lir_matches(
            predictions=lir_predictions,
            annotations=document.annotation.li_fields,
            pcc_set=pcc_set,
            iou_threshold=iou_threshold,
        )

    return page_image_with_matching(document, page, field_matching, image_size)
