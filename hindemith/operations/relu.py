from hindemith.operations.core import Map
# TODO: Add negative slope


class ReluForward(Map):
    """
    top = ReluForward(bottom)
    """
    def fn(bottom_elt):
        return bottom_elt if bottom_elt > 0 else 0


class ReluBackward(Map):
    """
    bottom_diff = ReluBackward(bottom, top_diff)
    """
    def fn(bottom_elt, top_diff_elt):
        return top_diff_elt * (bottom_elt > 0)
