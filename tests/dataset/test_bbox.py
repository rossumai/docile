from docile.dataset.bbox import BBox


def test_bbox_to_absolute_coords() -> None:
    b_float = BBox(left=0.1, top=0.2, right=0.3, bottom=0.4)
    b_abs_w10_h100 = BBox(left=1, top=20, right=3, bottom=40)
    assert b_float.to_absolute_coords(width=10, height=100) == b_abs_w10_h100


def test_bbox_intersects() -> None:
    b0022 = BBox(0, 0, 2, 2)
    b1122 = BBox(1, 1, 2, 2)
    b2233 = BBox(2, 2, 3, 3)
    b3344 = BBox(3, 3, 4, 4)
    assert b0022.intersects(b1122)
    assert b0022.intersects(b2233)
    assert not b0022.intersects(b3344)
    assert b1122.intersects(b2233)
    assert not b1122.intersects(b3344)
    assert b2233.intersects(b3344)
