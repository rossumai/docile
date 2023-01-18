import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

from docile.dataset import Field


@dataclass(frozen=True)
class FieldWithGroups(Field):
    groups: Optional[Sequence[str]] = None


def show_summary(args: argparse.Namespace, filename: str):
    """Helper function showing the summary of surgery experiment instance given by runtime
    arguments

    Parameters
    ----------
    args : argparse.Namespace
        input arguments
    """ """"""
    # Helper function showing the summary of surgery experiment instance given by runtime
    # arguments
    # """
    print("-" * 50)
    print(f"{filename}")
    print("-" * 10)
    [print(f"{k.upper()}: {v}") for k, v in vars(args).items()]
    print("-" * 50)


def bbox_str(bbox):
    if bbox:
        try:
            return f"{bbox[0]:>#04.1f}, {bbox[1]:>#04.1f}, {bbox[2]:>#04.1f}, {bbox[3]:>#04.1f}"
        except Exception:
            bbox = bbox.to_tuple()
            return f"{bbox[0]:>#04.1f}, {bbox[1]:>#04.1f}, {bbox[2]:>#04.1f}, {bbox[3]:>#04.1f}"
    else:
        return "<NONE>"


def print_docile_fields(fields, fieldtype=None, ft_width=65):
    for i, ft in enumerate(fields):
        if ft:
            if (fieldtype and ft.fieldtype == fieldtype) or fieldtype is None:
                if ft.text:
                    text = repr(ft.text) if isinstance(ft.text, str) else ft.text
                else:
                    text = "<NONE>"
                if isinstance(ft.fieldtype, list):
                    fieldtype_1 = ";".join(ft.fieldtype) if ft.fieldtype else "None"
                else:
                    fieldtype_1 = ft.fieldtype if ft.fieldtype else "None"
                # NOTE (michal.uricar): add page
                score = f"{ft.score:.2f}" if ft.score else "None"
                print(
                    f"{i:05d}: ",
                    f"ft='{fieldtype_1:<{ft_width}}' |"
                    f"page='{ft.page:<3}' |"
                    f"'{text:<30}' |"
                    f"{bbox_str(ft.bbox):<30} |"
                    f"{ft.groups} |"
                    f"{ft.line_item_id} |"
                    f"score={score:<5} | ",
                )
        else:
            print("None")
