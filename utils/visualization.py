import cv2


def annotate(result, show_masks=True):
    """Return annotated image as RGB."""
    return cv2.cvtColor(result.plot(masks=show_masks and result.masks is not None, line_width=2), cv2.COLOR_BGR2RGB)
